"""校验、加载和重新评测公开的 LoRA 权重；不会重新训练或读取参考预测冒充输出。

发布清单中的路径都相对于发布目录。原运行记录可能包含作者缓存路径，
这些字符串仅作为历史记录保存；模型始终由本机固定 revision 的缓存显式加载。
本模块和 verify_published 的默认路径仅依赖标准库，不导入 torch、不联网。
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import re
import traceback

from . import engine, runtime


DEFAULT_PUBLICATION = 'published/seed17_epoch2'
EXPECTED_COUNTS = {'train': 2048, 'validation': 256, 'test': 900}


def _inside(folder, value):
    """禁止清单中的绝对路径、Windows 盘符、目录穿越和逃出目录的符号链接。"""
    if not isinstance(value, str) or not value or '\\' in value or ':' in value:
        raise ValueError(f'发布清单必须使用普通相对 POSIX 路径：{value!r}')
    relative = PurePosixPath(value)
    if relative.is_absolute() or '..' in relative.parts or relative.as_posix() != value:
        raise ValueError(f'发布清单路径不合法：{value!r}')
    target = (folder / value).resolve()
    if not target.is_relative_to(folder) or target == folder:
        raise ValueError(f'发布清单路径超出发布目录：{value!r}')
    return target


def _publication(root, published_dir):
    root = Path(root).expanduser().resolve()
    selected = Path(published_dir).expanduser()
    folder = (selected if selected.is_absolute() else root / selected).resolve()
    return root, folder


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _verified_bundle(root, published_dir, model_dir=None):
    root, folder = _publication(root, published_dir)
    manifest_path = folder / 'manifest.json'
    spec = engine.load_json(manifest_path)
    _require(spec.get('schema_version') == 1 and spec.get('kind') == 'genqa_lora_publication',
             '不支持的发布清单格式')
    _, canonical_model = runtime._model_spec(root)
    _require(spec['model'] == canonical_model, '发布权重的底座模型/revision/文件清单不匹配')
    files = spec['files']
    _require(isinstance(files, dict) and bool(files), '发布清单 files 不能为空')
    checked = {}
    for name, record in files.items():
        _require(name != 'manifest.json', 'manifest 本身不列入 files，避免循环哈希')
        path = _inside(folder, name)
        _require(isinstance(record, dict), f'文件记录格式错误：{name}')
        digest, size = record.get('sha256'), record.get('bytes')
        _require(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest) is not None,
                 f'文件 SHA256 格式错误：{name}')
        _require(type(size) is int and size > 0, f'文件字节数错误：{name}')
        _require(path.is_file(), f'发布文件不存在：{name}')
        _require(path.stat().st_size == size and engine.file_sha256(path) == digest,
                 f'发布文件 SHA256/大小不符：{name}')
        checked[name] = path

    def listed(name):
        _inside(folder, name)
        _require(name in checked, f'必需文件没有列入 SHA256 清单：{name}')
        return checked[name]

    source = spec['source_run']
    _require(isinstance(source.get('name'), str) and bool(source['name']), '缺少原始运行名称')
    _require(re.fullmatch('[0-9a-f]{40}', source.get('source_commit', '')) is not None,
             'source_commit 必须是完整 Git 提交 SHA')
    original = engine.load_json(listed(source['run_manifest']))
    training_config = engine.load_json(listed(source['training_run_config']))
    summary = engine.load_json(listed(source['training_summary']))
    _require(original.get('formal_run') is True and original.get('smoke') is False,
             '只允许发布完成正式训练的结果，不能将 smoke 当作正式模型')
    config = original['config']
    _require(config['model'] == spec['model'], '原始运行模型与发布清单不一致')
    _require(spec['evaluation'] == config['evaluation'] and spec['runtime'] == config['runtime'],
             '发布评测/运行配置与原始运行不一致')
    _require(set(spec['evaluation']) == {'seed', 'batch_size', 'max_input_tokens',
                                       'max_new_tokens', 'loss_examples_per_split'},
             '发布评测参数与当前评测接口不兼容')
    _require(config['runtime'].get('precision') in ('auto', 'bf16', 'fp16')
             and type(config['runtime'].get('strict_determinism')) is bool,
             '运行精度或严格确定性配置无效')
    declared_training = {'seed': config['training']['seed'], 'epoch': config['training']['epochs'],
                         'train_examples': config['training']['train_examples']}
    _require(spec['training'] == declared_training == {'seed': 17, 'epoch': 2, 'train_examples': 2048},
             '此发布接口要求 seed17、2048 条训练数据、固定 epoch2')
    hp = {key: config['training'][key] for key in engine.DEFAULT_TRAINING}
    _require(training_config.get('complete') is True and summary.get('complete') is True
             and summary.get('formal_training_complete') is True,
             '原始训练尚未完整完成')
    _require(training_config['train_hparams'] == hp and training_config['seed'] == 17
             and training_config['model_id'] == engine.MODEL_ID
             and training_config['model_revision'] == engine.MODEL_REVISION,
             '原始训练的模型或超参数不一致')
    _require(training_config.get('fresh_adapter') is True and training_config.get('resume_from') is None,
             '发布记录必须是从底座新建 adapter 的正式训练')
    _require(summary['seed'] == 17 and summary['completed_epochs'] == 2
             and summary['examples_presented'] == 4096 and summary['global_step'] == 256,
             '原始训练轮次、样本呈现数或优化步数不完整')
    _require(summary.get('base_parameters_unchanged') is True
             and summary.get('adapter_parameters_changed') is True,
             '缺少底座冻结且 adapter 实际更新的证据')
    _require(summary.get('budget_satisfied') is True
             and 0 < summary['consumed_nonpadding_input_tokens'] <= config['training']['max_train_tokens'],
             '原始训练未满足实际 token 预算')

    adapter_dir = _inside(folder, spec['adapter_dir'])
    adapter_config_path = listed(spec['adapter_dir'] + '/adapter_config.json')
    listed(spec['adapter_dir'] + '/adapter_model.safetensors')
    _require(adapter_dir.is_dir(), 'adapter 目录不存在')
    adapter = engine.load_json(adapter_config_path)
    _require(adapter.get('base_model_name_or_path') == engine.MODEL_ID
             and adapter.get('revision') == engine.MODEL_REVISION,
             '发布版 adapter 必须绑定固定的模型 ID 和 revision，不能保留作者缓存路径')
    _require(adapter.get('peft_type') == 'LORA' and adapter.get('task_type') == 'CAUSAL_LM'
             and adapter.get('bias') == 'none' and adapter.get('r') == hp['lora_r']
             and adapter.get('lora_alpha') == hp['lora_alpha']
             and adapter.get('lora_dropout') == hp['lora_dropout']
             and set(adapter.get('target_modules', [])) == set(hp['target_modules'])
             and not adapter.get('use_dora', False) and not adapter.get('use_rslora', False)
             and not adapter.get('modules_to_save'), 'adapter 身份或 LoRA 配置与训练记录不符')
    # 即使 adapter 已固定远程标识，本接口也使用校验过的本机底座，避免隐式另下版本。
    prompt = engine.load_json(listed(spec['prompt']))
    _require(not prompt.get('demonstrations') and prompt.get('enable_thinking') is False,
             'LoRA 评测必须使用训练时固定的 non-thinking 零样例提示')
    _require(engine.object_sha256(prompt) == original['prompt_sha256']['baseline']
             == training_config['prompt_sha256'], '发布提示词与正式训练/评测提示不一致')
    if 'configs/prompting.json' in checked:
        prompting = engine.load_json(checked['configs/prompting.json'])
        _require(engine.object_sha256(prompting) == original['prompt_sha256']['prompting'],
                 '附带的 few-shot 提示与原运行不一致')

    _require(set(spec['data']) == set(EXPECTED_COUNTS), '必须提供 train/validation/test 三份固定数据')
    rows = {}
    for split, expected_n in EXPECTED_COUNTS.items():
        record = spec['data'][split]
        rows[split] = engine.load_jsonl(listed(record['path']))
        selected = rows[split]
        _require(len(selected) == record['n'] == original['active_counts'][split] == expected_n,
                 f'{split} 样本数不符合正式发布配置')
        _require(len({r['id'] for r in selected}) == expected_n, f'{split} 有重复 ID')
        _require(all(isinstance(r.get('context'), str) and isinstance(r.get('question'), str)
                     and isinstance(r.get('article_title'), str) and isinstance(r.get('answers'), list)
                     and bool(r['answers']) and all(isinstance(a, str) for a in r['answers'])
                     for r in selected), f'{split} 不是完整的 SQuAD 1.1 有答案数据')
        _require(engine.object_sha256(selected) == record['rows_sha256']
                 == original['active_rows_sha256'][split], f'{split} 数据内容/顺序与原运行不一致')
    _require(engine.object_sha256(rows['train']) == training_config['training_rows_sha256'],
             '发布训练数据与实际训练样本不一致')
    for first, second in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        for key in ('id', 'article_title'):
            _require(not ({r[key] for r in rows[first]} & {r[key] for r in rows[second]}),
                     f'{first}/{second} 存在跨集合重复 {key}')

    verified_model = None
    if model_dir is not None:
        # 离线验证：不调用 resolve_model，因为后者可能下载缺失文件。
        local = Path(model_dir).expanduser()
        local = (local if local.is_absolute() else root / local).resolve()
        runtime._checked_model_files(local, spec['model'])
        verified_model = str(local)
    verification = {
        'status': 'passed', 'publication_dir': str(folder), 'manifest_sha256': engine.file_sha256(manifest_path),
        'verified_files': len(checked), 'counts': {s: len(r) for s, r in rows.items()},
        'source_run': source['name'], 'source_commit': source['source_commit'],
        'model_revision': spec['model']['revision'], 'model_files_checked': model_dir is not None,
        'model_dir': verified_model, 'gpu_used': False, 'network_used': False,
    }
    return {'root': root, 'publication_dir': folder, 'spec': spec, 'config': config,
            'rows': rows, 'prompt': prompt, 'adapter_dir': adapter_dir, 'verification': verification}


def verify_published(root='.', published_dir=DEFAULT_PUBLICATION, *, model_dir=None):
    """仅用 CPU 校验发布包；显式传 model_dir 才额外离线校验底座的 8 个文件。"""
    return _verified_bundle(root, published_dir, model_dir)['verification']


def load_published_model(root='.', published_dir=DEFAULT_PUBLICATION, *, model_dir=None):
    """校验后加载底座+LoRA，返回含 model/tokenizer/spec/runtime_info 的字典。

    默认底座位置为 root/cache/model。仅模型缺失文件会从固定 revision 下载。
    此接口需要可实际运行的 NVIDIA CUDA GPU；不会调用训练函数。
    调用者用完后应删除返回字典中的 model，并调用 engine.free_memory()。
    """
    from .workflow import check_cuda, setup_reproducibility
    loaded = _verified_bundle(root, published_dir)
    config = loaded['config']
    reproducibility = setup_reproducibility(config)
    cuda_probe = check_cuda(config)
    model_path = runtime.resolve_model(loaded['root'], model_dir if model_dir is not None else 'cache/model')
    tokenizer = engine.load_tokenizer(str(model_path), local_files_only=True)
    try:
        model = engine.load_model(str(model_path), precision=config['runtime']['precision'],
                                  adapter_dir=loaded['adapter_dir'], local_files_only=True)
    except BaseException as error:
        # PEFT 可能在底座已上 CUDA 后失败；Notebook 会保留 traceback。
        # 清除已经退出的加载帧局部引用，才能让 free_memory 真正回收底座。
        # 当前执行帧不会被 clear_frames 清除，原异常及栈位置仍原样抛出。
        traceback.clear_frames(error.__traceback__)
        engine.free_memory()
        raise
    try:
        environment = runtime.environment_info(next(model.parameters()).dtype)
    except BaseException:
        del model
        engine.free_memory()
        raise
    loaded.update(model=model, tokenizer=tokenizer, model_dir=model_path,
                  runtime_info={'reproducibility': reproducibility, 'cuda_probe': cuda_probe,
                                'environment': environment})
    return loaded


def evaluate_published(root='.', published_dir=DEFAULT_PUBLICATION, *, split='test', model_dir=None,
                       output_dir=None):
    """固定原实验配置，重新生成所选集合的答案并评分，输出到全新目录。

    split 仅允许 train/validation/test。train 的生成指标不是训练 token accuracy。
    参考预测不作为模型输入，也不作为此次推理结果；每次必须实际调用 evaluate。
    """
    if split not in EXPECTED_COUNTS:
        raise ValueError('split 只能是 train、validation 或 test')
    root, _ = _publication(root, published_dir)
    # 在创建运行目录及任何 GPU 调用前，拒绝损坏或不兼容的发布包。
    verification = verify_published(root, published_dir)
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        destination = root / 'runs' / ('pretrained_' + stamp)
    else:
        path = Path(output_dir).expanduser()
        destination = (path if path.is_absolute() else root / path).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    engine.save_json(destination / 'status.json', {'status': 'running', 'training_performed': False, 'split': split})
    loaded = None
    try:
        loaded = load_published_model(root, published_dir, model_dir=model_dir)
        engine.save_json(destination / 'run_manifest.json', {
            'schema_version': 1, 'kind': 'published_lora_evaluation', 'training_performed': False,
            'split': split, 'publication': verification, 'config': loaded['config'],
            'runtime_info': loaded['runtime_info'], 'model_dir': str(loaded['model_dir']),
            'active_rows_sha256': engine.object_sha256(loaded['rows'][split]),
            'prompt_sha256': engine.object_sha256(loaded['prompt']),
            'code_sha256': {str(path.relative_to(root)).replace('\\', '/'): engine.file_sha256(path)
                            for path in sorted((root / 'src' / 'genqa').glob('*.py'))},
        })
        metrics = engine.evaluate(loaded['model'], loaded['tokenizer'], loaded['rows'][split],
            loaded['prompt'], destination / split, run_label=f'Published LoRA {split}',
            **loaded['spec']['evaluation'])
        engine.save_json(destination / 'status.json', {'status': 'completed', 'training_performed': False, 'split': split})
        return {'run_dir': str(destination), 'metrics': metrics, 'verification': verification,
                'runtime_info': loaded['runtime_info']}
    except BaseException as error:
        engine.save_json(destination / 'status.json', {'status': 'failed', 'training_performed': False,
                                                      'split': split, 'error_type': type(error).__name__,
                                                      'error': str(error)})
        raise
    finally:
        if loaded is not None:
            del loaded['model']
            engine.free_memory()
