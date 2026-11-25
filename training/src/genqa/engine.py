"""SQuAD 1.1 生成式 QA：新数据流程的训练与评测核心。

本文件只使用传入的路径，不依赖作者电脑、工作目录布局或 __file__。
baseline/改进 prompting 不更新权重；LoRA 只训练适配器。所有指标来自本次
实际运行。SQuAD 1.1 只有有答案题，不评测拒答能力。
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import re
import statistics
import string
import sys
import time

from .budget import (DEFAULT_MAX_TRAIN_TOKENS, TrainingSequenceTooLong,
                     batch_token_counts, check_token_budget, plan_training_budget,
                     step_token_counts)


# 固定底座的仓库提交；不能用会随时间改变的 main 代替这个 revision。
MODEL_ID = 'Qwen/Qwen3-0.6B'
MODEL_REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'
DEFAULT_TRAINING = {
    'epochs': 2, 'micro_batch_size': 4, 'gradient_accumulation_steps': 4,
    'max_seq_length': 768, 'learning_rate': 2e-4, 'weight_decay': .01,
    'max_grad_norm': 1.0, 'warmup_ratio': .1,
    'lora_r': 8, 'lora_alpha': 16, 'lora_dropout': .05,
    'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                       'gate_proj', 'up_proj', 'down_proj'],
    'gradient_checkpointing': True,
}
DEFAULT_SEEDS = (17, 123, 2026)


def load_json(path):
    """读取配置/指标；路径由 Notebook 明确传入。"""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()
            if line.strip()]


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def save_jsonl(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(v, ensure_ascii=False) + '\n' for v in values),
                    encoding='utf-8')


def file_sha256(path):
    """大模型文件分块读，避免一次性占用主机内存。"""
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def object_sha256(value):
    """为内存中的配置/数据建立稳定指纹，记录顺序也属于实验配置。"""
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _new_output_directory(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f'输出目录已有文件，请使用新的运行目录，避免混入旧结果：{path}')
    path.mkdir(parents=True, exist_ok=True)
    return path


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def free_memory():
    """先在调用处 del model，再调用此函数；它不会删除仍被变量引用的模型。"""
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def resolve_precision(precision='auto'):
    """Ampere/更新 GPU 优先 bf16；Colab T4 使用 fp16，训练时必须配合 GradScaler。

    不自动退到 CPU：0.6B 的训练/完整生成在 CPU 上会非常慢。跨 GPU、CUDA、
    dtype 的结果可能略有不同，固定 seed 不能保证跨硬件逐位复现。
    """
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('未检测到 CUDA GPU。请在 Colab/Kaggle 中启用 GPU 运行时后重试；不会静默使用 CPU 训练。')
    if precision not in ('auto', 'bf16', 'fp16'):
        raise ValueError("precision 必须是 'auto'、'bf16' 或 'fp16'")
    # 只认可硬件原生 BF16：默认 including_emulation=True 可能把 T4/P100 的
    # “能分配 BF16 张量”误当作原生支持，进而跳过 FP16 所需的 GradScaler。
    native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    chosen = ('bf16' if native_bf16 else 'fp16') if precision == 'auto' else precision
    if chosen == 'bf16' and not native_bf16:
        raise RuntimeError("这块 GPU 不支持 bf16；T4 请使用 precision='fp16' 或 'auto'。")
    return torch.bfloat16 if chosen == 'bf16' else torch.float16


def _model_kwargs(model_name_or_path, local_files_only):
    """远程下载固定提交；本地快照若带 manifest，也检查其 revision。"""
    source = str(model_name_or_path)
    path = Path(source)
    if path.is_dir():
        manifest = path/'download_manifest.json'
        if manifest.exists():
            info = load_json(manifest)
            if info.get('model_id') != MODEL_ID or info.get('revision') != MODEL_REVISION:
                raise ValueError('本地缓存的模型身份/revision 与本实验不一致')
        return {'pretrained_model_name_or_path': source, 'local_files_only': True}
    if source != MODEL_ID:
        raise ValueError('此复现模块只接受固定 Qwen3-0.6B 仓库或明确的本地快照目录')
    return {'pretrained_model_name_or_path': source, 'revision': MODEL_REVISION,
            'local_files_only': local_files_only}


def load_tokenizer(model_name_or_path=MODEL_ID, *, local_files_only=False):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(**_model_kwargs(model_name_or_path, local_files_only),
                                               padding_side='left')
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError('本实验要求 tokenizer 明确定义 padding 和 EOS token')
    return tokenizer


def load_model(model_name_or_path=MODEL_ID, *, precision='auto', adapter_dir=None,
               local_files_only=False):
    """加载用于评测的底座或底座＋LoRA；不 merge adapter，不允许评测更新权重。"""
    from transformers import AutoModelForCausalLM
    dtype = resolve_precision(precision)
    model = AutoModelForCausalLM.from_pretrained(
        **_model_kwargs(model_name_or_path, local_files_only),
        torch_dtype=dtype, attn_implementation='sdpa').to('cuda')
    if adapter_dir is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False,
                                         autocast_adapter_dtype=True)
    model.requires_grad_(False)
    model.eval()
    model.config.use_cache = True
    return model


def normalize_answer(value):
    """SQuAD 1.1 / MRQA 官方归一化：小写、删除ASCII标点/英文冠词、合并空格。"""
    value = value.lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', value).split())


def score_answer(prediction, answers):
    """多参考 EM/F1，各自取最优匹配。返回 0–100，不是 0–1。

    这是有答案任务。空参考列表直接报错，不把它误当成 SQuAD 2.0 拒答题。
    precision/recall 是最高 F1 参考对应的诊断量，不是官方独立指标。
    """
    if not isinstance(prediction, str) or isinstance(answers, str) or not answers:
        raise ValueError('prediction 必须为字符串；SQuAD1.1 answers 必须为非空字符串列表')
    values = []
    for answer in answers:
        p, a = normalize_answer(prediction), normalize_answer(answer)
        pw, aw = p.split(), a.split()
        common = sum((Counter(pw) & Counter(aw)).values())
        precision = common/len(pw) if pw else 0.
        recall = common/len(aw) if aw else 0.
        f1 = 2*common/(len(pw)+len(aw)) if pw and aw else 0.
        values.append({'em':100.*(p == a), 'f1':100*f1,
                       'precision':100*precision, 'recall':100*recall})
    best = dict(max(values, key=lambda r:r['f1']))
    best['em'] = max(r['em'] for r in values)
    return best


def prepare_input(row, tokenizer, prompt_config, max_input_tokens=8192):
    """逐 token 复用旧 run_prompt 的消息模板，包括示例、换行和非思考模式。

    构造输入只读取 context/question 及训练集 demonstration，不读取当前题的 gold。
    评测沿用历史策略：超长时只从 context 尾部截断，保留完整问题和所有示例；
    不按答案位置裁剪。覆盖 gold 的诊断是在输入固定以后计算，绝不反馈给输入。
    本包固定验证/测试集正常均可完整输入，context_truncated 会如实记录。
    """
    if prompt_config.get('enable_thinking', False) is not False:
        raise ValueError('此实验固定 enable_thinking=False')
    instruction = prompt_config['instruction']
    template = prompt_config['target_template']
    prefix = []
    for demo in prompt_config['demonstrations']:
        prefix += [{'role':'user','content':template.format(instruction=instruction,
                    context=demo['context'],question=demo['question'])},
                   {'role':'assistant','content':demo['answer']}]

    def render(context):
        messages = prefix + [{'role':'user','content':template.format(
            instruction=instruction,context=context,question=row['question'])}]
        return tokenizer.apply_chat_template(messages, tokenize=True,
            add_generation_prompt=True, enable_thinking=False)

    context = row['context']
    ids = render(context)
    original_length = len(ids)
    if len(ids) > max_input_tokens:
        context_ids = tokenizer.encode(context, add_special_tokens=False)
        if len(render('')) > max_input_tokens:
            raise ValueError('示例＋问题本身超出输入预算，不能静默截断问题')
        low, high = 0, min(max_input_tokens, len(context_ids))
        while low < high:
            mid = (low+high+1)//2
            if len(render(tokenizer.decode(context_ids[:mid]))) <= max_input_tokens:
                low = mid
            else:
                high = mid-1
        context = tokenizer.decode(context_ids[:low])
        ids = render(context)
    covered = any(normalize_answer(a) and normalize_answer(a) in normalize_answer(context)
                  for a in row.get('answers', []))
    return {'input_ids':ids, 'original_input_tokens':original_length,
            'input_tokens':len(ids), 'context_truncated':original_length>max_input_tokens,
            'gold_present_in_visible_context_normalized':bool(covered)}


def tokenize_training_rows(rows, tokenizer, prompt_config, max_seq_length=768):
    """固定选出的训练题重新 token 化；监督首参考＋EOS，prompt labels 全为-100。

    训练不做 silent drop 或截断：任何一题超出768就停止，让使用者检查配置。
    选择与训练共用此函数，训练前重新核算完整输入，不按答案位置裁剪。
    """
    output, seen = [], set()
    for row in rows:
        if row['id'] in seen:
            raise ValueError('训练 ID 重复')
        seen.add(row['id'])
        if not row['answers']:
            raise ValueError('SQuAD 1.1 训练题必须有答案')
        # 此处只在 CPU 上 tokenize，先完整构造后拒绝超长，绝不截断训练正文。
        prepared = prepare_input(row, tokenizer, prompt_config, 10**9)
        target = tokenizer.encode(row['answers'][0], add_special_tokens=False)+[tokenizer.eos_token_id]
        ids = prepared['input_ids']+target
        if prepared['context_truncated'] or len(ids)>max_seq_length:
            raise TrainingSequenceTooLong(f"训练题 {row['id']} 长度 {len(ids)} 超出 {max_seq_length}；固定样本不允许静默丢弃")
        output.append({'id':row['id'], 'input_ids':ids,
                       'labels':[-100]*prepared['input_tokens']+target,
                       'prompt_tokens':prepared['input_tokens'], 'target_tokens':len(target)})
    if not output:
        raise ValueError('训练数据为空')
    return output


def collate_rows(rows, pad_token_id, device='cpu'):
    """训练右侧padding；padding以及prompt均不参加答案loss。"""
    import torch
    width = max(len(r['input_ids']) for r in rows)
    inputs = torch.full((len(rows),width),pad_token_id,dtype=torch.long,device=device)
    attention = torch.zeros_like(inputs)
    labels = torch.full_like(inputs,-100)
    for i,row in enumerate(rows):
        length=len(row['input_ids'])
        inputs[i,:length]=torch.tensor(row['input_ids'],dtype=torch.long,device=device)
        attention[i,:length]=1
        labels[i,:length]=torch.tensor(row['labels'],dtype=torch.long,device=device)
    return {'input_ids':inputs,'attention_mask':attention,'labels':labels}


def answer_loss_sum(model, batch):
    """答案 next-token CE 总和，以及 teacher-forced 正确 token 数。

    位置j的logits预测labels[j+1]。只计算各样本需要的logit位置并集，减少大词表
    投影占用；-100仍忽略其他样本的prompt和padding。sum后再用整个effective batch
    的答案token总数归一化，不能简单平均每个micro batch的mean loss。
    """
    import torch
    import torch.nn.functional as F
    labels=batch['labels']
    positions=torch.where((labels[:,1:]!=-100).any(dim=0))[0]
    if positions.numel()==0:
        raise ValueError('Batch没有监督答案token')
    chosen_labels=labels[:,positions+1]
    result=model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],
                 use_cache=False,logits_to_keep=positions)
    logits=result.logits
    loss=F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),chosen_labels.reshape(-1),
                         ignore_index=-100,reduction='sum')
    with torch.no_grad():
        mask=chosen_labels.ne(-100)
        correct=((logits.argmax(-1)==chosen_labels)&mask).sum()
        count=mask.sum()
    return loss,correct,count


def _tensor_digest(parameters):
    """训练前后全量校验冻结底座；只读取参数，不导出底座副本。"""
    import torch
    digest=hashlib.sha256()
    for name,value in parameters:
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def train_lora(rows, prompt_config, output_dir, *, seed=17,
               model_name_or_path=MODEL_ID, precision='auto', training_config=None,
               local_files_only=False, max_train_tokens=DEFAULT_MAX_TRAIN_TOKENS):
    """从原始底座新建LoRA，训练两轮并保存epoch_1/epoch_2。

    默认2048题、micro4×accum4=effective16，共128步/轮、256次更新。支持显式
    training_config覆盖做独立小实验。完整计划超出max_train_tokens会在加载模型前
    报错；实际前向含FP16重试同样受上限约束，不以少训/减少轮数凑预算。
    seed同时控制LoRA初始化、dropout及每轮shuffle，不根据测试选择seed。
    """
    hp={**DEFAULT_TRAINING, **(training_config or {})}
    for key in ('epochs','micro_batch_size','gradient_accumulation_steps','max_seq_length','lora_r'):
        if not isinstance(hp[key],int) or hp[key]<1:
            raise ValueError(f'非法训练配置：{key}')
    if prompt_config['demonstrations']:
        raise ValueError('LoRA训练使用与原baseline完全相同的零样例提示，不能混入few-shot')
    tokenizer=load_tokenizer(model_name_or_path,local_files_only=local_files_only)
    tokenized=tokenize_training_rows(rows,tokenizer,prompt_config,hp['max_seq_length'])
    budget=plan_training_budget(tokenized,epochs=hp['epochs'],max_train_tokens=max_train_tokens)
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM
    dtype=resolve_precision(precision)
    output=_new_output_directory(output_dir)
    save_json(output/'budget_plan.json',budget)
    torch.set_num_threads(4)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    save_jsonl(output/'train_tokenized.jsonl',tokenized)
    save_json(output/'prompt_config.json',prompt_config)
    base=AutoModelForCausalLM.from_pretrained(**_model_kwargs(model_name_or_path,local_files_only),
        torch_dtype=dtype,attn_implementation='sdpa').to('cuda')
    base.config.use_cache=False
    model=get_peft_model(base,LoraConfig(task_type=TaskType.CAUSAL_LM,r=hp['lora_r'],
        lora_alpha=hp['lora_alpha'],lora_dropout=hp['lora_dropout'],
        target_modules=hp['target_modules'],bias='none'),autocast_adapter_dtype=True)
    if hp['gradient_checkpointing']:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.enable_input_require_grads()
    model.train()
    trainable=[(name,p) for name,p in model.named_parameters() if p.requires_grad]
    frozen=[(name,p) for name,p in model.named_parameters() if not p.requires_grad]
    if not trainable or any('lora_' not in name or p.dtype!=torch.float32 for name,p in trainable):
        raise RuntimeError('只应有float32 LoRA adapter参数参与训练')
    parameters=[p for _,p in trainable]
    frozen_before,adapter_before=_tensor_digest(frozen),_tensor_digest(trainable)
    effective_batch=hp['micro_batch_size']*hp['gradient_accumulation_steps']
    steps_per_epoch=math.ceil(len(tokenized)/effective_batch)
    total_steps=hp['epochs']*steps_per_epoch
    warmup=max(1,math.ceil(hp['warmup_ratio']*total_steps)) if hp['warmup_ratio']>0 else 0
    optimizer=torch.optim.AdamW(parameters,lr=hp['learning_rate'],weight_decay=hp['weight_decay'])

    def lr_multiplier(step):
        if warmup and step<warmup:
            return step/warmup
        progress=(step-warmup)/max(1,total_steps-warmup)
        return .5*(1.+math.cos(math.pi*min(1.,max(0.,progress))))

    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_multiplier)
    # BF16指数范围大，不启用loss scaling。T4使用FP16，必须缩放梯度并在裁剪前unscale。
    scaler=torch.amp.GradScaler('cuda',enabled=dtype==torch.float16)
    config={'created_utc':_utc_now(),'model_id':MODEL_ID,'model_revision':MODEL_REVISION,
        'model_source':str(model_name_or_path),'seed':seed,'train_hparams':hp,
        'train_rows':len(rows),'training_rows_sha256':object_sha256(rows),
        'token_budget':budget,'max_train_tokens':max_train_tokens,
        'tokenized_sha256':object_sha256(tokenized),'prompt_sha256':object_sha256(prompt_config),
        'base_dtype':str(dtype),'adapter_dtype':'torch.float32','device':'cuda',
        'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,
        'versions':{p:importlib.metadata.version(p) for p in ('torch','transformers','peft')},
        'effective_batch_size':effective_batch,'steps_per_epoch':steps_per_epoch,
        'planned_optimizer_steps':total_steps,'warmup_steps':warmup,
        'scheduler':'linear warmup then cosine to zero','fresh_adapter':True,'resume_from':None,
        'trainable_parameters':sum(p.numel() for p in parameters),
        'base_parameter_sha256_before':frozen_before,'adapter_sha256_before':adapter_before,
        'loss':'Answer+EOS token-summed CE / all supervised tokens in the effective batch',
        'accuracy':'Teacher-forced next-token accuracy; not free-generation QA EM',
        'fp16_overflow_policy':'Retry same effective batch and same RNG state at reduced GradScaler scale; maximum 8 retries',
        'cross_hardware_note':'Seeds do not guarantee bit-identical outputs across dtype/GPU/CUDA/software',
        'complete':False}
    save_json(output/'run_config.json',config)
    torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    logs,epoch_summaries,global_step,overflow_retries=[],[],0,0
    consumed=dict.fromkeys(('nonpadding_input_tokens','supervised_target_tokens','padded_input_tokens','padding_tokens'),0)
    completed_training=False
    try:
        for epoch in range(hp['epochs']):
            epoch_started=time.perf_counter()
            order=list(range(len(tokenized)))
            random.Random(seed+epoch).shuffle(order)
            epoch_logs=[]
            for group_index in range(steps_per_epoch):
                group=[tokenized[i] for i in order[group_index*effective_batch:(group_index+1)*effective_batch]]
                group_counts=step_token_counts(group,hp['micro_batch_size'])
                total_targets=sum(r['target_tokens'] for r in group)
                lr_used=optimizer.param_groups[0]['lr']
                # FP16溢出时不跳过训练数据/调度步，而是降低scale后重试这一组。
                # 恢复RNG保证dropout掩码不因重试而改变，避免悄悄改变实验预算。
                cpu_rng=torch.get_rng_state()
                cuda_rng=torch.cuda.get_rng_state_all()
                for attempt in range(9):
                    check_token_budget(consumed['nonpadding_input_tokens']+group_counts['nonpadding_input_tokens'],
                        max_train_tokens,stage='下一次训练前向（含重试）')
                    if attempt:
                        torch.set_rng_state(cpu_rng)
                        torch.cuda.set_rng_state_all(cuda_rng)
                    optimizer.zero_grad(set_to_none=True)
                    loss_sum_value,correct_value,counted_targets=0.,0,0
                    for offset in range(0,len(group),hp['micro_batch_size']):
                        micro_rows=group[offset:offset+hp['micro_batch_size']]
                        batch=collate_rows(micro_rows,tokenizer.pad_token_id,'cuda')
                        for key,value in batch_token_counts(micro_rows).items():
                            consumed[key]+=value
                        with torch.autocast(device_type='cuda',dtype=dtype):
                            loss_sum,correct,count=answer_loss_sum(model,batch)
                        if not torch.isfinite(loss_sum):
                            raise FloatingPointError('出现非有限loss；已停止，不能将此运行当作有效结果')
                        scaler.scale(loss_sum/total_targets).backward()
                        loss_sum_value+=loss_sum.detach().item()
                        correct_value+=int(correct.item())
                        counted_targets+=int(count.item())
                        del batch,loss_sum,correct,count
                    if counted_targets!=total_targets:
                        raise RuntimeError('梯度累积的答案token计数不一致')
                    if any(p.grad is not None for _,p in frozen):
                        raise RuntimeError('冻结底座意外产生梯度')
                    scaler.unscale_(optimizer)
                    grad_norm=torch.nn.utils.clip_grad_norm_(parameters,hp['max_grad_norm'],
                        error_if_nonfinite=not scaler.is_enabled()).item()
                    old_scale=scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.is_enabled() and scaler.get_scale()<old_scale:
                        overflow_retries+=1
                        if attempt==8:
                            raise FloatingPointError('FP16连续溢出超过重试上限；请保留失败日志并检查环境')
                        continue
                    break
                scheduler.step()
                torch.cuda.synchronize()
                global_step+=1
                entry={'epoch':epoch+1,'step_in_epoch':group_index+1,'global_step':global_step,
                    'examples':len(group),'input_tokens':sum(len(r['input_ids']) for r in group),
                    **group_counts,
                    **{f'consumed_{key}':value*(attempt+1) for key,value in group_counts.items()},
                    'cumulative_consumed_nonpadding_input_tokens':consumed['nonpadding_input_tokens'],
                    'max_train_tokens':max_train_tokens,
                    'supervised_tokens':total_targets,'train_nll':loss_sum_value/total_targets,
                    'teacher_forced_token_accuracy':correct_value/total_targets,
                    'learning_rate':lr_used,'next_learning_rate':optimizer.param_groups[0]['lr'],
                    'grad_norm_before_clip':grad_norm,'elapsed_seconds':time.perf_counter()-started,
                    'effective_batch_size':len(group),'overflow_retries_this_step':attempt,
                    'grad_scaler_scale':scaler.get_scale(),
                    'mode':'formal' if len(rows)==2048 and hp['epochs']==2 else 'explicit_small_run'}
                logs.append(entry)
                epoch_logs.append(entry)
                with (output/'train_log.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(entry,ensure_ascii=False)+'\n')
                if global_step==1 or global_step%16==0 or global_step==total_steps:
                    print(f"seed={seed} epoch={epoch+1} step={global_step}/{total_steps} "
                          f"loss={entry['train_nll']:.4f} TF-token-acc={100*entry['teacher_forced_token_accuracy']:.2f}%",flush=True)
            checkpoint=output/f'epoch_{epoch+1}'
            checkpoint.mkdir()
            model.save_pretrained(checkpoint,safe_serialization=True,save_embedding_layers=False)
            targets=sum(e['supervised_tokens'] for e in epoch_logs)
            record={'epoch':epoch+1,'optimizer_steps':len(epoch_logs),'examples_presented':sum(e['examples'] for e in epoch_logs),
                    'supervised_tokens':targets,'input_tokens':sum(e['input_tokens'] for e in epoch_logs),
                    **{key:sum(e[key] for e in epoch_logs) for key in group_counts},
                    **{f'consumed_{key}':sum(e[f'consumed_{key}'] for e in epoch_logs) for key in group_counts},
                    'train_nll':sum(e['train_nll']*e['supervised_tokens'] for e in epoch_logs)/targets,
                    'teacher_forced_token_accuracy_pct':100*sum(e['teacher_forced_token_accuracy']*e['supervised_tokens'] for e in epoch_logs)/targets,
                    'elapsed_including_checkpoint_seconds':time.perf_counter()-epoch_started}
            epoch_summaries.append(record)
            save_json(checkpoint/'training_state.json',{'seed':seed,'completed_epochs':epoch+1,
                'global_step':global_step,'training_config_sha256':object_sha256(config),
                'training_rows_sha256':config['training_rows_sha256'],'epoch_metrics':record})
        frozen_after,adapter_after=_tensor_digest(frozen),_tensor_digest(trainable)
        if frozen_after!=frozen_before:
            raise RuntimeError('冻结底座参数发生变化')
        if global_step>1 and adapter_after==adapter_before:
            raise RuntimeError('LoRA参数未发生变化')
        targets=sum(e['supervised_tokens'] for e in logs)
        if (sum(e['input_tokens'] for e in logs)!=budget['planned_nonpadding_input_tokens']
                or targets!=budget['planned_supervised_target_tokens']
                or global_step!=total_steps or len(epoch_summaries)!=hp['epochs']):
            raise RuntimeError('实际完成的训练数据/监督token/步数/轮数与完整预算计划不一致')
        check_token_budget(consumed['nonpadding_input_tokens'],max_train_tokens,stage='已消耗训练前向')
        summary={'complete':True,'formal_training_complete':len(rows)==2048 and hp['epochs']==2,
            'seed':seed,'completed_epochs':hp['epochs'],'requested_epochs':hp['epochs'],
            'global_step':global_step,'examples_presented':sum(e['examples'] for e in logs),
            'input_tokens':sum(e['input_tokens'] for e in logs),'supervised_tokens':targets,
            **{key:sum(e[key] for e in logs) for key in group_counts},
            **{f'completed_{key}':sum(e[key] for e in logs) for key in group_counts},
            **{f'consumed_{key}':value for key,value in consumed.items()},
            'max_train_tokens':max_train_tokens,'budget_plan':budget,'budget_satisfied':True,
            'train_nll':sum(e['train_nll']*e['supervised_tokens'] for e in logs)/targets,
            'teacher_forced_token_accuracy':sum(e['teacher_forced_token_accuracy']*e['supervised_tokens'] for e in logs)/targets,
            'epochs':epoch_summaries,'fp16_overflow_retries':overflow_retries,
            'base_parameters_unchanged':True,'adapter_parameters_changed':adapter_after!=adapter_before,
            'base_parameter_sha256_after':frozen_after,'adapter_sha256_after':adapter_after,
            'trainable_parameters':config['trainable_parameters'],
            'elapsed_including_checkpoint_seconds':time.perf_counter()-started,
            'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved(),
            'final_adapter_dir':str(output/f'epoch_{hp["epochs"]}'),
            'training_token_accuracy_scale':'0-1 in step logs and summary; 0-100 in epoch *_pct',
            'notes':'Training token accuracy uses gold answer prefixes; run evaluate separately for free-generation EM/F1.'}
        save_json(output/'training_summary.json',summary)
        config['complete']=True
        config['completed_utc']=_utc_now()
        save_json(output/'run_config.json',config)
        completed_training=True
        return summary
    finally:
        failure_in_flight=sys.exc_info()[1]
        try:
            save_json(output/'budget_usage.json',{
                'complete':completed_training,'completed_epochs':len(epoch_summaries),
                'requested_epochs':hp['epochs'],'max_train_tokens':max_train_tokens,
                'completed_nonpadding_input_tokens':sum(e['input_tokens'] for e in logs),
                'completed_supervised_target_tokens':sum(e['supervised_tokens'] for e in logs),
                'completed_padded_input_tokens':sum(e['padded_input_tokens'] for e in logs),
                'completed_padding_tokens':sum(e['padding_tokens'] for e in logs),
                **{f'consumed_{key}':value for key,value in consumed.items()},
                'token_definition':budget['token_definition'],
                'padding_counted_in_budget':False,'retries_counted_in_actual_budget':True,
                'budget_satisfied':consumed['nonpadding_input_tokens']<=max_train_tokens})
        except Exception as audit_error:
            if failure_in_flight is None:
                raise
            failure_in_flight.add_note(f'保存失败运行的预算审计也失败：{audit_error}')
        finally:
            del model,base,optimizer,scheduler,scaler,parameters,trainable,frozen
            free_memory()


def aggregate_metrics(rows):
    """生成答案评分与形式诊断；所有分数/比例为百分数，token长度是原始生成长度。

    “长于最长参考”只是形式诊断，不等于人工语义错误。多个split不应混成一个
    泛化结论，因此同时输出by_split；overall只是题目加权描述。
    """
    if not rows:
        raise ValueError('没有预测可评分')
    def summarize(values):
        scores=[score_answer(r['prediction'],r['answers']) for r in values]
        n=len(values)
        lengths=[len(normalize_answer(r['prediction']).split()) for r in values]
        refs=[max(len(normalize_answer(a).split()) for a in r['answers']) for r in values]
        counts={'empty_prediction_n':sum(not r['prediction'].strip() for r in values),
                'generation_capped_n':sum(bool(r.get('generation_capped',False)) for r in values),
                'context_truncated_n':sum(bool(r.get('context_truncated',False)) for r in values),
                'prediction_longer_than_longest_reference_n':sum(p>g for p,g in zip(lengths,refs)),
                'exact_match_n':sum(s['em']==100 for s in scores),'zero_f1_n':sum(s['f1']==0 for s in scores)}
        result={'n':n,**{k:statistics.mean(s[k] for s in scores) for k in ('em','f1','precision','recall')},
                **counts,'mean_normalized_prediction_words':statistics.mean(lengths),
                'mean_longest_reference_words':statistics.mean(refs)}
        if all('output_tokens' in r for r in values):
            result.update(mean_output_tokens=statistics.mean(r['output_tokens'] for r in values),
                          median_output_tokens=statistics.median(r['output_tokens'] for r in values))
        for key,value in counts.items():
            result[key.removesuffix('_n')+'_rate_pct']=100*value/n
        if all('article_title' in r for r in values):
            result['n_articles']=len({r['article_title'] for r in values})
        return result
    groups=defaultdict(list)
    for row in rows:
        groups[row.get('split','evaluation')].append(row)
    return {'scale':'0-100 (percent)','scorer':'SQuAD1.1/MRQA official normalization and multi-reference max',
            'overall':summarize(rows),'overall_note':'Question-weighted diagnostic; report each split separately',
            'by_split':[{'split':split,**summarize(values)} for split,values in sorted(groups.items())]}


def evaluate(model, tokenizer, rows, prompt_config, output_dir, *, batch_size=4,
             max_input_tokens=8192, max_new_tokens=64, loss_examples_per_split=64,
             seed=42, run_label=''):
    """自由生成＋固定hash子样本的答案NLL/PPL/TF-token-accuracy；不反向传播。

    为公平比较，同一split的所有方法使用同样题目、greedy预算和loss子样本。
    prediction只strip两端空白，不借gold抽取短答案、不模糊修正输出。
    要评训练子集的生成EM/F1，可传入该子集并用新的output_dir；这是真实重跑，
    不应把训练teacher-forced accuracy冒充自由生成准确率。
    """
    import torch
    from transformers import GenerationConfig
    if batch_size<1 or max_input_tokens+max_new_tokens>32768 or loss_examples_per_split<0:
        raise ValueError('无效评测预算')
    if not torch.cuda.is_available() or next(model.parameters()).device.type!='cuda':
        raise RuntimeError('本实验评测需要已加载到CUDA的模型；请启用GPU')
    if not rows or len({(r.get('split','evaluation'),r['id']) for r in rows})!=len(rows):
        raise ValueError('评测数据为空或split/ID重复')
    if any(not r['answers'] for r in rows):
        raise ValueError('本模块是SQuAD1.1有答案任务')
    output=_new_output_directory(output_dir)
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model.eval()
    model.requires_grad_(False)
    model.config.use_cache=True
    tokenizer.padding_side='left'
    generation=GenerationConfig(do_sample=False,num_beams=1,max_new_tokens=max_new_tokens,
        use_cache=True,eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,repetition_penalty=1.0)
    prepared=[{**row,**prepare_input(row,tokenizer,prompt_config,max_input_tokens)} for row in rows]
    # 排序只为减少padding。与旧引擎相同，确定batch组成才能比较数值路径。
    prepared.sort(key=lambda r:(r['input_tokens'],r.get('dataset','SQuAD'),r['id']))
    config={'created_utc':_utc_now(),'run_label':run_label,'model_id':MODEL_ID,'model_revision':MODEL_REVISION,
        'seed':seed,'dtype':str(next(model.parameters()).dtype),'gpu':torch.cuda.get_device_name(0),
        'input_rows_sha256':object_sha256(rows),'n':len(rows),'prompt_config':prompt_config,
        'prompt_sha256':object_sha256(prompt_config),'generation':generation.to_dict(),
        'enable_thinking':False,'use_model_defaults':False,'prediction_postprocessing':'strip only',
        'batch_size':batch_size,'max_input_tokens':max_input_tokens,'max_new_tokens':max_new_tokens,
        'loss_examples_per_split':loss_examples_per_split,'parameters_updated_during_evaluation':0,
        'context_policy':'Full input if possible; otherwise deterministic head-only context truncation, never gold-conditioned',
        'loss_policy':'First reference plus EOS; fixed SHA256(loss:seed:id) subset; token-weighted mean',
        'versions':{p:importlib.metadata.version(p) for p in ('torch','transformers','peft')},
        'python':platform.python_version(),'complete':False}
    save_json(output/'run_config.json',config)
    save_json(output/'prompt_config.json',prompt_config)
    results,durations,loss_rows=[],[],[]
    eos_ids=generation.eos_token_id if isinstance(generation.eos_token_id,list) else [generation.eos_token_id]
    with torch.inference_mode():
        warm=torch.tensor([prepared[0]['input_ids']],device='cuda')
        model.generate(warm,attention_mask=torch.ones_like(warm),generation_config=generation,
                       use_model_defaults=False,do_sample=False,max_new_tokens=4)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        with (output/'predictions.jsonl').open('w',encoding='utf-8') as stream:
            for offset in range(0,len(prepared),batch_size):
                batch=prepared[offset:offset+batch_size]
                encoded=tokenizer.pad([{'input_ids':r['input_ids']} for r in batch],padding=True,return_tensors='pt').to('cuda')
                torch.cuda.synchronize()
                started=time.perf_counter()
                generated=model.generate(**encoded,generation_config=generation,use_model_defaults=False,do_sample=False)
                torch.cuda.synchronize()
                elapsed=time.perf_counter()-started
                durations.append(elapsed)
                for row,tokens in zip(batch,generated[:,encoded['input_ids'].shape[1]:].tolist()):
                    stop=next((i for i,t in enumerate(tokens) if t in eos_ids),len(tokens))
                    raw=tokenizer.decode(tokens[:stop],skip_special_tokens=True)
                    result={k:v for k,v in row.items() if k!='input_ids'}
                    result.update(raw_output=raw,prediction=raw.strip(),output_tokens=min(stop+1,len(tokens)),
                        generation_capped=stop==len(tokens) and len(tokens)>=max_new_tokens,
                        batch_size=len(batch),batch_generate_seconds=elapsed,batch_amortized_seconds=elapsed/len(batch))
                    stream.write(json.dumps(result,ensure_ascii=False)+'\n')
                    results.append(result)
                stream.flush()
                if offset==0 or (offset+len(batch))%64==0 or offset+len(batch)==len(prepared):
                    print(f'{run_label} 生成 {offset+len(batch)}/{len(prepared)}',flush=True)
        generation_peak=torch.cuda.max_memory_allocated()
        by_split=defaultdict(list)
        for row in prepared:
            by_split[row.get('split','evaluation')].append(row)
        for split,candidates in by_split.items():
            candidates.sort(key=lambda r:hashlib.sha256(f'loss:{seed}:{r["id"]}'.encode()).hexdigest())
            for row in candidates[:loss_examples_per_split]:
                targets=tokenizer.encode(row['answers'][0],add_special_tokens=False)+[tokenizer.eos_token_id]
                if len(row['input_ids'])+len(targets)>32768:
                    raise ValueError('held-out参考答案超出模型上下文预算')
                ids=torch.tensor([row['input_ids']+targets],device='cuda')
                result=model(ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=len(targets)+1)
                logits=result.logits[:,:-1,:].float()
                labels=torch.tensor([targets],device='cuda')
                loss=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels.reshape(-1),reduction='sum')
                if not torch.isfinite(loss):
                    raise FloatingPointError('held-out loss非有限值')
                loss_rows.append({'id':row['id'],'split':split,'reference':row['answers'][0],
                    'target_tokens':len(targets),'nll_sum':loss.item(),
                    'teacher_forced_correct_tokens':int((logits.argmax(-1)==labels).sum().item())})
                del result,logits,labels,ids,loss
    save_jsonl(output/'heldout_loss.jsonl',loss_rows)
    metrics=aggregate_metrics(results)
    heldout={}
    for split in by_split:
        selected=[r for r in loss_rows if r['split']==split]
        count=sum(r['target_tokens'] for r in selected)
        nll=sum(r['nll_sum'] for r in selected)/count if count else None
        heldout[split]={'n':len(selected),'target_tokens':count,'answer_nll':nll,
            'answer_perplexity':math.exp(nll) if nll is not None else None,
            'teacher_forced_token_accuracy':100*sum(r['teacher_forced_correct_tokens'] for r in selected)/count if count else None}
    elapsed=sum(durations)
    metrics.update(complete=True,n_predictions=len(results),heldout_loss=heldout,
        parameters_updated_during_evaluation=0,
        training={'performed_during_evaluation':False,'training_loss':None,'training_accuracy':None},
        runtime={'generation_seconds':elapsed,'examples_per_second':len(results)/elapsed,
            'output_tokens_per_second':sum(r['output_tokens'] for r in results)/elapsed,
            'median_batch_seconds':statistics.median(durations),
            'generation_peak_allocated_gib':generation_peak/2**30,
            'all_stages_peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30},
        output_token_definition='Generated tokens including EOS when present; excludes padding',
        teacher_forced_accuracy_scale='0-100 in heldout_loss',
        artifact_hashes={name:file_sha256(output/name) for name in ('predictions.jsonl','heldout_loss.jsonl')})
    save_json(output/'metrics.json',metrics)
    config['complete']=True
    config['completed_utc']=_utc_now()
    config['artifact_hashes']={name:file_sha256(output/name) for name in ('predictions.jsonl','heldout_loss.jsonl','metrics.json')}
    save_json(output/'run_config.json',config)
    return metrics
