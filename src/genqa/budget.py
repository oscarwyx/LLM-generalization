"""训练 token 预算：固定样本先计数，再决定能否开始完整训练。

input 包含 prompt、答案和 EOS；target 是其中受监督的答案与 EOS 子集，不能
再加到 input 上。每轮重新出现的 token 都重新计数，不是独立 token 数，也不
包括底座预训练费用。padding 单独报告；FP16 重试也消耗实际 input 预算。
"""
from __future__ import annotations

import hashlib
import json


DEFAULT_MAX_TRAIN_TOKENS = 1_000_000
TOKEN_DEFINITION = ('Non-padding prompt + first reference answer + EOS for each training microbatch presentation, including retries; '
                    'excludes internal activation-checkpoint recomputation and frozen-base pretraining.')


class TokenBudgetExceeded(ValueError):
    """完整训练计划或下一次实际前向计算将超过预算；不得静默缩短训练。"""


class TrainingSequenceTooLong(ValueError):
    """完整 prompt + answer + EOS 超过训练单题长度限制。"""


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} 必须是正整数')


def check_token_budget(nonpadding_input_tokens, max_train_tokens, *, stage='训练计划'):
    _positive_integer(max_train_tokens, 'max_train_tokens')
    if isinstance(nonpadding_input_tokens, bool) or not isinstance(nonpadding_input_tokens, int) or nonpadding_input_tokens < 0:
        raise ValueError('nonpadding_input_tokens 必须是非负整数')
    if nonpadding_input_tokens > max_train_tokens:
        raise TokenBudgetExceeded(
            f'{stage}需要 {nonpadding_input_tokens:,} 个非 padding input tokens，'
            f'超过上限 {max_train_tokens:,}。已停止；不会换更短题、少训或缩减轮数。')


def batch_token_counts(rows):
    """一个实际 micro-batch 的 token 数；padded_input 是矩形张量槽位数。

    这是 token-slot 统计，不是 FLOPs 或注意力二次复杂度的精确替代量。
    监督数按 next-token 预测实际可见的 labels[1:] 计数。
    """
    if not rows:
        raise ValueError('不能统计空 batch')
    nonpadding, targets, lengths = 0, 0, []
    for row in rows:
        ids, labels = row['input_ids'], row['labels']
        if not ids or len(ids) != len(labels) or labels[0] != -100:
            raise ValueError('tokenized 样本长度不一致，或首位置不应被监督')
        count = sum(value != -100 for value in labels[1:])
        if count < 1 or any(label != -100 and label != token for label, token in zip(labels, ids)):
            raise ValueError('监督标签必须来自 input_ids 的答案位置')
        if 'target_tokens' in row and count != row['target_tokens']:
            raise ValueError('target_tokens 与实际监督 labels 不一致')
        nonpadding += len(ids)
        targets += count
        lengths.append(len(ids))
    padded = len(rows) * max(lengths)
    return {'nonpadding_input_tokens': nonpadding, 'supervised_target_tokens': targets,
            'padded_input_tokens': padded, 'padding_tokens': padded - nonpadding}


def step_token_counts(rows, micro_batch_size):
    """effective batch 分成多个 micro-batch，逐块计算 padding 后再求和。"""
    _positive_integer(micro_batch_size, 'micro_batch_size')
    if not rows:
        raise ValueError('不能统计空 step')
    total = dict.fromkeys(('nonpadding_input_tokens', 'supervised_target_tokens', 'padded_input_tokens', 'padding_tokens'), 0)
    for start in range(0, len(rows), micro_batch_size):
        for name, value in batch_token_counts(rows[start:start + micro_batch_size]).items():
            total[name] += value
    return total


def plan_training_budget(tokenized_rows, *, epochs=2, max_train_tokens=DEFAULT_MAX_TRAIN_TOKENS):
    """不含重试的完整 epoch 计划；数据顺序和 token IDs 也纳入指纹。"""
    _positive_integer(epochs, 'epochs')
    counts = batch_token_counts(tokenized_rows)
    total = counts['nonpadding_input_tokens'] * epochs
    check_token_budget(total, max_train_tokens)
    tokenized_hash = hashlib.sha256(json.dumps(tokenized_rows, ensure_ascii=False,
        sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
    return {
        'n': len(tokenized_rows), 'epochs': epochs, 'max_train_tokens': max_train_tokens,
        'per_epoch_nonpadding_input_tokens': counts['nonpadding_input_tokens'],
        'per_epoch_supervised_target_tokens': counts['supervised_target_tokens'],
        'planned_nonpadding_input_tokens': total,
        'planned_supervised_target_tokens': counts['supervised_target_tokens'] * epochs,
        'planned_examples_presented': len(tokenized_rows) * epochs,
        'tokenized_rows_sha256': tokenized_hash,
        'token_definition': TOKEN_DEFINITION,
        'padding_counted_in_budget': False,
        'retries_counted_in_actual_budget': True,
        'padding_note': 'Actual per-micro-batch padded token slots are logged separately; not a FLOP count.',
    }


def selection_rank(row_id, seed=42):
    """公开固定排名，不读取答案、问题文本、模型预测或评分。"""
    return hashlib.sha256(f'train-selection:{seed}:{row_id}'.encode('utf-8')).hexdigest()


def select_training_rows(pool, tokenizer, prompt, n=2048, seed=42, maxseq=768,
                         max_train_tokens=DEFAULT_MAX_TRAIN_TOKENS, epochs=2):
    """按固定 ID 哈希取首 n 条标签有效且全长 <= maxseq 的样本。

    只有输入 ID 决定排名。答案仅用于监督合法性及总 token 长度检查，不用内容
    类型或模型得分选择“容易题”。选满 n 后即固定集合；计划超预算直接报错，
    不继续扫描以替换为更短样本。返回原始 rows 和可写入 JSON 的预算审计。
    """
    from .engine import tokenize_training_rows
    _positive_integer(n, 'n')
    _positive_integer(maxseq, 'maxseq')
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError('seed 必须是整数')
    if len({row['id'] for row in pool}) != len(pool):
        raise ValueError('训练候选池 ID 重复')
    ranked = sorted(pool, key=lambda row: (selection_rank(row['id'], seed), row['id']))
    selected, tokenized = [], []
    invalid_labels, overlength, considered = 0, 0, 0
    for row in ranked:
        considered += 1
        answers = row.get('answers')
        if (not isinstance(answers, list) or not answers
                or any(not isinstance(answer, str) or not answer.strip() for answer in answers)):
            invalid_labels += 1
            continue
        try:
            item = tokenize_training_rows([row], tokenizer, prompt, maxseq)[0]
        except TrainingSequenceTooLong:
            overlength += 1
            continue
        selected.append(row)
        tokenized.append(item)
        if len(selected) == n:
            break
    if len(selected) != n:
        raise ValueError(f'只有 {len(selected)} 条合法且不超长样本，未达到固定要求 {n}')
    budget = plan_training_budget(tokenized, epochs=epochs, max_train_tokens=max_train_tokens)
    budget.update(selection_seed=seed, selection_hash='SHA256(train-selection:{seed}:{id})',
        max_seq_length=maxseq, pool_n=len(pool), considered_n=considered,
        skipped_invalid_labels_n=invalid_labels, skipped_overlength_n=overlength,
        selected_ids=[row['id'] for row in selected],
        selection_policy='First n valid length-eligible rows in fixed ID-hash order; no score filtering or budget-fitting replacement.')
    return selected, budget
