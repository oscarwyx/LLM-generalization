"""公开原料 → 数据划分 → baseline → few-shot → LoRA → 对比报告。

Notebook 与命令行共用这些函数。每次 run 建立独立目录，不读取旧实验成绩。
"""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from . import engine
from .budget import select_training_rows
from .data import prepare_data
from .runtime import configure_reproducibility, resolve_model, environment_info


def read_config(path='configs/experiment.json'):
    path = Path(path).expanduser().resolve()
    config = engine.load_json(path)
    if config['schema_version'] != 1:
        raise ValueError('不支持的配置版本')
    if config['model']['model_id'] != engine.MODEL_ID or config['model']['revision'] != engine.MODEL_REVISION:
        raise ValueError('本项目固定 Qwen3-0.6B 及提交版本')
    root = path.parent.parent
    canonical_model = engine.load_json(root/'configs/experiment.json')['model']
    if config['model'] != canonical_model:
        raise ValueError('备用配置必须保留固定模型及文件校验清单；只修改训练/数据/运行参数')
    return root, config


def resolve_path(root, value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(root)/path).resolve()


def new_run(root, config, smoke=False):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    path = resolve_path(root, config['paths']['runs']) / (('smoke_' if smoke else '') + stamp)
    path.mkdir(parents=True, exist_ok=False)
    engine.save_json(path/'status.json', {'status':'running', 'smoke':smoke})
    return path


def choose_demonstrations(train, prompt, seed=42):
    """从实际训练集固定选择三篇不同文章的示例，不看验证/测试答案或模型分数。"""
    candidates = sorted(train, key=lambda r: hashlib.sha256(f'few-shot:{seed}:{r["id"]}'.encode()).hexdigest())
    demos, articles = [], set()
    for row in candidates:
        if row['article_title'] in articles:
            continue
        articles.add(row['article_title'])
        demos.append({k:row[k] for k in ('id','context','question','article_title')})
        demos[-1]['answer'] = row['answers'][0]
        if len(demos) == 3:
            break
    if len(demos) != 3:
        raise ValueError('至少需要三篇训练文章才能构建固定 three-shot 提示')
    result = deepcopy(prompt)
    result['demonstrations'] = demos
    return result


def prepare_experiment(root, config, *, raw_dir=None, model_dir=None, output_dir=None, smoke=False):
    """真正重做清理/文章划分，再按固定排名选择训练样本并做训练预算预检。"""
    root = Path(root).resolve()
    output = resolve_path(root, output_dir or config['paths']['prepared'])
    raw = resolve_path(root, raw_dir or config['paths']['raw'])
    manifest = prepare_data(raw, output, **config['data'])
    model_source = model_dir or os.environ.get('QA_MODEL_DIR') or config['paths']['model']
    model_path = resolve_model(root, model_source)
    tokenizer = engine.load_tokenizer(str(model_path), local_files_only=True)
    baseline = engine.load_json(root/'configs/baseline.json')
    training = config['training']
    train, budget = select_training_rows(
        engine.load_jsonl(output/'train.jsonl'), tokenizer, baseline,
        n=16 if smoke else training['train_examples'], seed=config['data']['seed'],
        maxseq=training['max_seq_length'], epochs=training['epochs'],
        max_train_tokens=training['max_train_tokens'])
    splits = {'train':train, 'validation':engine.load_jsonl(output/'validation.jsonl'),
              'test':engine.load_jsonl(output/'squad_test.jsonl')}
    if smoke:
        splits['validation'] = splits['validation'][:8]
        splits['test'] = splits['test'][:8]
    prompting = choose_demonstrations(train, engine.load_json(root/'configs/prompting.json'), config['data']['seed'])
    engine.save_jsonl(output/'train_selected.jsonl', train)
    engine.save_json(output/'budget_plan.json', budget)
    engine.save_json(output/'prompting_selected.json', prompting)
    engine.save_json(output/'sample_ids.json', {s:[r['id'] for r in rows] for s,rows in splits.items()})
    print(f"数据：train={len(train)} / validation={len(splits['validation'])} / test={len(splits['test'])}", flush=True)
    print(f"计划训练 input tokens={budget['planned_nonpadding_input_tokens']:,} / 上限={training['max_train_tokens']:,}；"
          f"监督子集={budget['planned_supervised_target_tokens']:,}", flush=True)
    return {'root':root, 'config':deepcopy(config), 'directory':output, 'model_dir':model_path,
            'tokenizer':tokenizer, 'splits':splits, 'baseline':baseline, 'prompting':prompting,
            'budget':budget, 'data_manifest':manifest, 'smoke':smoke}


def setup_reproducibility(config):
    # CUBLAS 环境配置必须在第一次 CUDA 运算之前；命令行和 Notebook 都会提前设置。
    return configure_reproducibility(config['training']['seed'], config['runtime']['strict_determinism'])


def check_cuda(config):
    """实际进行矩阵计算，而不只检查能否识别GPU。"""
    import torch
    dtype = engine.resolve_precision(config['runtime']['precision'])
    probe = torch.ones((8,8),device='cuda',dtype=dtype)
    try:
        if (probe @ probe)[0,0].item() != 8:
            raise RuntimeError('CUDA矩阵运算自检失败')
    finally:
        del probe
        engine.free_memory()
    return {'gpu':torch.cuda.get_device_name(0),'dtype':str(dtype),'torch':torch.__version__}


def record_run(run_dir, prepared, reproducibility, dtype):
    """保存实际配置、输入/代码指纹、样本 ID 和环境；不依赖作者目录或历史结果。"""
    root, config = prepared['root'], prepared['config']
    code = {}
    for path in sorted([*root.joinpath('src').rglob('*.py'), *root.joinpath('notebooks').glob('*.ipynb'),
                        root/'requirements.txt',root/'pyproject.toml']):
        if path.suffix == '.ipynb':
            notebook = engine.load_json(path)
            contents = [{'cell_type':c['cell_type'],'source':c['source']} for c in notebook['cells']]
            code[path.relative_to(root).as_posix()] = engine.object_sha256(contents)
        else:
            code[path.relative_to(root).as_posix()] = engine.file_sha256(path)
    model_spec = config['model']
    engine.save_json(Path(run_dir)/'run_manifest.json', {
        'schema_version':1, 'smoke':prepared['smoke'], 'formal_run':not prepared['smoke'],
        'config':config, 'budget_plan':prepared['budget'], 'model_files_verified':model_spec,
        'reproducibility':reproducibility, 'environment':environment_info(dtype), 'code_sha256':code,
        'data_manifest':prepared['data_manifest'],
        'active_counts':{s:len(rows) for s,rows in prepared['splits'].items()},
        'active_rows_sha256':{s:engine.object_sha256(rows) for s,rows in prepared['splits'].items()},
        'prompt_sha256':{s:engine.object_sha256(prepared[s]) for s in ('baseline','prompting')},
    })
    engine.save_json(Path(run_dir)/'sample_ids.json', {s:[r['id'] for r in rows] for s,rows in prepared['splits'].items()})


def evaluate_one(model, prepared, run_dir, method, phase, *, epoch=None):
    """各集合真实生成答案；EM/F1覆盖全体，NLL等覆盖预先固定的最多64题。"""
    config = prepared['config']
    folder = f'lora_epoch_{epoch}' if method == 'LoRA' else method.lower()
    prompt = prepared['prompting'] if method == 'Prompting' else prepared['baseline']
    metrics = engine.evaluate(model, prepared['tokenizer'], prepared['splits'][phase], prompt,
        Path(run_dir)/folder/phase, run_label=f'{method} {phase}' + (f' epoch {epoch}' if epoch else ''),
        **config['evaluation'])
    score = metrics['by_split'][0]
    loss = metrics['heldout_loss'][score['split']]
    row = {'method':method,'phase':phase,'epoch':epoch,'n':score['n'], 'EM (%)':score['em'],
           'F1 (%)':score['f1'],'mean_output_tokens':score['mean_output_tokens'],
           'loss_n':loss['n'],'answer_NLL':loss['answer_nll'],'answer_PPL':loss['answer_perplexity'],
           'TF_token_accuracy (%)':loss['teacher_forced_token_accuracy']}
    engine.save_json(Path(run_dir)/folder/phase/'table_row.json', row)
    print(json.dumps(row,ensure_ascii=False),flush=True)
    return row


def evaluate_method(model, prepared, run_dir, method):
    return [evaluate_one(model, prepared, run_dir, method, phase) for phase in ('train','validation','test')]


def train(prepared, run_dir):
    config, settings = prepared['config'], prepared['config']['training']
    hp = {key:settings[key] for key in engine.DEFAULT_TRAINING}
    return engine.train_lora(prepared['splits']['train'], prepared['baseline'], Path(run_dir)/'lora_training',
        seed=settings['seed'], model_name_or_path=str(prepared['model_dir']),
        precision=config['runtime']['precision'], training_config=hp,
        max_train_tokens=settings['max_train_tokens'], local_files_only=True)


def evaluate_lora(prepared, run_dir):
    """训练结束后独立载入每轮adapter验证；预先固定最终轮测试，不按test挑选。"""
    epochs = prepared['config']['training']['epochs']
    rows = []
    for epoch in range(1, epochs+1):
        model = engine.load_model(str(prepared['model_dir']), local_files_only=True,
            precision=prepared['config']['runtime']['precision'], adapter_dir=Path(run_dir)/f'lora_training/epoch_{epoch}')
        try:
            rows.append(evaluate_one(model,prepared,run_dir,'LoRA','validation',epoch=epoch))
            if epoch == epochs:
                rows += [evaluate_one(model,prepared,run_dir,'LoRA',phase,epoch=epoch) for phase in ('train','test')]
        finally:
            del model
            engine.free_memory()
    return rows


def run_experiment(config_path='configs/experiment.json', *, raw_dir=None, model_dir=None, smoke=False):
    from .report import make_report
    root, config = read_config(config_path)
    repro = setup_reproducibility(config)
    # 首先验证CUDA架构可实际运算，避免错误环境等到下载/准备后才报错。
    print(check_cuda(config),flush=True)
    run_dir = new_run(root,config,smoke)
    print('新运行目录：',run_dir,flush=True)
    try:
        prepared = prepare_experiment(root,config,raw_dir=raw_dir,model_dir=model_dir,
                                      output_dir=run_dir/'data',smoke=smoke)
        model = engine.load_model(str(prepared['model_dir']), precision=config['runtime']['precision'],local_files_only=True)
        try:
            record_run(run_dir,prepared,repro,next(model.parameters()).dtype)
            evaluate_method(model,prepared,run_dir,'Baseline')
            evaluate_method(model,prepared,run_dir,'Prompting')
        finally:
            del model
            engine.free_memory()
        train(prepared,run_dir)
        evaluate_lora(prepared,run_dir)
        make_report(run_dir)
        engine.save_json(run_dir/'status.json',{'status':'completed','smoke':smoke,'formal_run':not smoke})
    except Exception as error:
        engine.save_json(run_dir/'status.json',{'status':'failed','smoke':smoke,'error_type':type(error).__name__,'error':str(error)})
        raise
    print('全部阶段完成：',run_dir,flush=True)
    return run_dir
