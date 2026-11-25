"""纯 CPU 测试：预算拒绝、token 口径、固定 ID 排名与 padding 计算。"""
from unittest.mock import patch

import unittest

from genqa import engine
from genqa.budget import (TokenBudgetExceeded, batch_token_counts, check_token_budget,
                         plan_training_budget, select_training_rows, selection_rank,
                         step_token_counts)


PROMPT = {'instruction': 'Answer.', 'target_template': '{context} {question}',
          'demonstrations': [], 'enable_thinking': False}


class TinyTokenizer:
    """每个空白分隔词一个 token，另有一个 chat 前缀 token；不下载任何模型。"""
    eos_token_id = 2
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [10 + i for i, _ in enumerate(text.split())]

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize and add_generation_prompt and not enable_thinking
        return [1] + self.encode(messages[-1]['content'])


def raw(row_id, context='one two', answer='answer'):
    return {'id': row_id, 'context': context, 'question': 'Which?', 'answers': [answer]}


def tokenized(row_id, length, targets=2):
    ids = list(range(10, 10 + length))
    return {'id': row_id, 'input_ids': ids,
            'labels': [-100] * (length - targets) + ids[-targets:], 'target_tokens': targets}


class BudgetTests(unittest.TestCase):
    def test_budget_counts_repeated_input_and_target_subset(self):
        rows = [tokenized('a', 6, 2), tokenized('b', 9, 3)]
        plan = plan_training_budget(rows, epochs=2, max_train_tokens=30)
        assert plan['per_epoch_nonpadding_input_tokens'] == 15
        assert plan['planned_nonpadding_input_tokens'] == 30
        assert plan['planned_supervised_target_tokens'] == 10
        assert plan['planned_examples_presented'] == 4
        assert plan['padding_counted_in_budget'] is False
        # 监督 target 已经包含在 input 中，不是 30+10，也不是只数答案的 10。
        with self.assertRaises(TokenBudgetExceeded):
            plan_training_budget(rows, epochs=2, max_train_tokens=29)


    def test_padding_counts_actual_micro_batches_not_effective_batch_width(self):
        rows = [tokenized('a', 3), tokenized('b', 8), tokenized('c', 4), tokenized('d', 5)]
        assert batch_token_counts(rows[:2]) == {
            'nonpadding_input_tokens': 11, 'supervised_target_tokens': 4,
            'padded_input_tokens': 16, 'padding_tokens': 5}
        counts = step_token_counts(rows, micro_batch_size=2)
        assert counts['nonpadding_input_tokens'] == 20
        assert counts['padded_input_tokens'] == 16 + 10
        assert counts['padding_tokens'] == 6
        assert counts['padded_input_tokens'] != len(rows) * 8


    def test_invalid_label_metadata_is_rejected(self):
        row = tokenized('a', 5)
        row['target_tokens'] = 999
        with self.assertRaisesRegex(ValueError, 'target_tokens'):
            batch_token_counts([row])


    def test_selection_is_fixed_id_rank_independent_of_pool_order(self):
        pool = [raw(str(i)) for i in range(12)]
        expected = sorted(pool, key=lambda r: (selection_rank(r['id'], 42), r['id']))[:4]
        one, budget = select_training_rows(pool, TinyTokenizer(), PROMPT, n=4, maxseq=8)
        two, _ = select_training_rows(list(reversed(pool)), TinyTokenizer(), PROMPT, n=4, maxseq=8)
        assert [r['id'] for r in one] == [r['id'] for r in two] == [r['id'] for r in expected]
        assert budget['planned_nonpadding_input_tokens'] == 4 * 6 * 2
        assert budget['planned_supervised_target_tokens'] == 4 * 2 * 2


    def test_selection_skips_only_invalid_or_overlength(self):
        pool = [raw(str(i)) for i in range(8)]
        ranked = sorted(pool, key=lambda r: (selection_rank(r['id']), r['id']))
        ranked[0]['context'] = 'too ' * 40
        ranked[1]['answers'] = []
        rows, budget = select_training_rows(pool, TinyTokenizer(), PROMPT, n=3, maxseq=8)
        assert [r['id'] for r in rows] == [r['id'] for r in ranked[2:5]]
        assert budget['skipped_invalid_labels_n'] == budget['skipped_overlength_n'] == 1


    def test_over_budget_does_not_replace_first_ranked_with_shorter_rows(self):
        pool = [raw(str(i)) for i in range(8)]
        ranked = sorted(pool, key=lambda r: (selection_rank(r['id']), r['id']))
        for row in ranked[:2]:
            row['context'] = 'long ' * 10
        # 前2条各14 tokens，2轮需56；后面的短题可以预算内，但不准偷偷替换。
        with self.assertRaisesRegex(TokenBudgetExceeded, '不会换更短题'):
            select_training_rows(pool, TinyTokenizer(), PROMPT, n=2, maxseq=20,
                                 max_train_tokens=30, epochs=2)


    def test_engine_rejects_budget_before_gpu_or_output_creation(self):
        with patch.object(engine, 'load_tokenizer', return_value=TinyTokenizer()), \
                patch.object(engine, 'resolve_precision') as gpu, \
                patch.object(engine, '_new_output_directory') as output:
            with self.assertRaises(TokenBudgetExceeded):
                engine.train_lora([raw('x')], PROMPT, 'should_not_exist',
                                  max_train_tokens=11, training_config={'epochs': 2})
            gpu.assert_not_called()
            output.assert_not_called()


    def test_retry_consumption_cannot_exceed_cap(self):
        # 完整计划已预检通过，但重试仍可能耗尽实际上限，下一次前向必须拒绝。
        check_token_budget(30, 30)
        with self.assertRaises(TokenBudgetExceeded):
            check_token_budget(30 + 6, 30, stage='下一次训练前向（含重试）')


    def test_input_includes_chat_prompt_first_answer_and_eos(self):
        rows = engine.tokenize_training_rows([raw('a')], TinyTokenizer(), PROMPT)
        item = rows[0]
        assert len(item['input_ids']) == 6  # chat前缀 + 3个prompt词 + 答案 + EOS
        assert item['labels'] == [-100] * 4 + [10, TinyTokenizer.eos_token_id]
        assert item['prompt_tokens'] == 4 and item['target_tokens'] == 2


if __name__ == "__main__":
    unittest.main()
