"""可迁移实验的模型缓存、随机性设置和运行清单；不读取历史实验成绩。

本文件只有标准库顶层依赖。Notebook 应在 import torch 之前显式设置
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'，再调用下面的函数。
所有相对路径均以传入的 package_dir 为基准，不依赖某台电脑的盘符。
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import random
import sys
import warnings


MODEL_ID = 'Qwen/Qwen3-0.6B'
MODEL_REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'
MODEL_FILENAMES = (
    'README.md', 'config.json', 'generation_config.json', 'merges.txt',
    'model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json',
)
REPRODUCIBILITY_LIMIT = (
    '固定数据、模型修订、种子及确定性算子可改善重复计算的一致性；'
    '不同 GPU、dtype、CUDA、PyTorch/Transformers 版本仍不保证逐位相同。'
)


def _sha256(path):
    """流式读入：模型权重约 1.5 GB，不能把整个文件一次装入内存。"""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _package_path(package_dir):
    root = Path(package_dir).expanduser().resolve()
    if not (root / 'configs' / 'experiment.json').is_file():
        raise FileNotFoundError(f'请传入解压后的实验目录，缺少：{root / "configs/experiment.json"}')
    return root


def _model_spec(package_dir):
    root = _package_path(package_dir)
    spec = json.loads((root / 'configs' / 'experiment.json').read_text(encoding='utf-8'))['model']
    if spec.get('model_id') != MODEL_ID or spec.get('revision') != MODEL_REVISION:
        raise ValueError('模型清单的仓库或 revision 不符合本实验的固定版本')
    files = spec.get('files', [])
    if len(files) != len(MODEL_FILENAMES) or {r.get('name') for r in files} != set(MODEL_FILENAMES):
        raise ValueError('模型清单必须恰好列出本实验的 8 个模型/分词器文件')
    for record in files:
        name = PurePosixPath(record['name'])
        if name.is_absolute() or '..' in name.parts:
            raise ValueError('模型清单不能包含目录穿越路径')
        digest = record.get('sha256', '')
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError(f'模型文件 SHA256 格式错误：{name}')
        if not isinstance(record.get('bytes'), int) or record['bytes'] < 1:
            raise ValueError(f'模型文件大小记录错误：{name}')
    return root, spec


def _checked_model_files(model_dir, spec, *, allow_missing=False):
    verified, missing = [], []
    for record in spec['files']:
        path = model_dir / record['name']
        if not path.exists():
            missing.append(record['name'])
            continue
        if not path.is_file():
            raise ValueError(f'预期是模型文件，实际不是普通文件：{path}')
        size = path.stat().st_size
        if size != record['bytes']:
            raise ValueError(f'模型文件大小不符，不自动覆盖已有文件：{path}')
        actual = _sha256(path)
        if actual != record['sha256']:
            raise ValueError(f'模型 SHA256 不符，不自动覆盖已有文件：{path}')
        verified.append({'name': record['name'], 'bytes': size, 'sha256': actual})
    if missing and not allow_missing:
        raise FileNotFoundError(f'模型文件不完整：{missing}；目录：{model_dir}')
    return verified, missing


def resolve_model(package_dir, model_dir=None):
    """定位并校验模型；仅缺失文件会从 Hugging Face 固定修订下载。

    优先级：显式 model_dir > QA_MODEL_DIR 环境变量 > 包内 cache/model。
    用户提供相对路径时，也相对于 package_dir 解释。已有文件内容不符会报错，
    不把同名但不同版本的权重当作本实验模型，也不自动覆盖用户文件。
    首次下载需联网；已有全部 8 个正确文件时完全离线，不调用下载 API。
    """
    root, spec = _model_spec(package_dir)
    selected = model_dir if model_dir is not None else os.environ.get('QA_MODEL_DIR', '').strip()
    target = Path(selected or 'cache/model').expanduser()
    if not target.is_absolute():
        target = root / target
    target = target.resolve()
    if target.exists() and not target.is_dir():
        raise ValueError(f'模型缓存路径不是目录：{target}')
    _, missing = _checked_model_files(target, spec, allow_missing=True)
    if missing:
        from huggingface_hub import snapshot_download
        target.mkdir(parents=True, exist_ok=True)
        print(f'模型缓存缺少 {len(missing)} 个文件；下载固定修订 {MODEL_REVISION} 到 {target}', flush=True)
        snapshot_download(
            repo_id=MODEL_ID, repo_type='model', revision=MODEL_REVISION,
            local_dir=str(target), allow_patterns=missing,
        )
        _checked_model_files(target, spec)
    print(f'模型 8 个文件 SHA256 校验通过：{target}', flush=True)
    return target


def configure_reproducibility(seed, strict=True):
    """设定 Python/NumPy/PyTorch 随机种子及实际确定性选项，返回可保存记录。

    strict=True：遇到已知非确定性算子时抛错；False：警告后继续，不能把结果
    描述为严格确定性。此函数应在加载模型或开始 CUDA 运算之前调用。
    PYTHONHASHSEED 只能在启动 Python 前生效；这里记录现状，不假装运行时可改。
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError('seed 应是 [0, 2**32) 范围内的整数')
    if not isinstance(strict, bool):
        raise TypeError('strict 应为 bool')
    loaded_torch = sys.modules.get('torch')
    cuda_initialized = bool(loaded_torch is not None and loaded_torch.cuda.is_initialized())
    workspace = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    if workspace is None and not cuda_initialized:
        # Notebook 仍应可见地在 import torch 前设置；这里为直接调用 .py 的用户兜底。
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        workspace = ':4096:8'
    if workspace not in (':4096:8', ':16:8'):
        message = ('CUBLAS_WORKSPACE_CONFIG 未正确设置。请重启内核，并在 import torch 前'
                   "设置 os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'。")
        if strict:
            raise RuntimeError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    import numpy as np
    import torch
    torch.set_num_threads(4)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True, warn_only=not strict)
    return {
        'seed': seed, 'strict': strict,
        'python_random_seeded': True, 'numpy_random_seeded': True,
        'torch_seeded': True, 'cuda_all_devices_seeded': bool(torch.cuda.is_available()),
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'float32_matmul_precision': torch.get_float32_matmul_precision(),
        'torch_num_threads': torch.get_num_threads(),
        'CUBLAS_WORKSPACE_CONFIG': workspace,
        'cuda_initialized_before_configuration': cuda_initialized,
        'PYTHONHASHSEED': os.environ.get('PYTHONHASHSEED'),
        'python_hash_seed_note': '只记录进程启动配置；运行时设置不能改变当前进程的哈希随机化。',
        'limitations': REPRODUCIBILITY_LIMIT,
    }


def environment_info(dtype):
    """记录实际运行环境；不做矩阵计算，也不以 GPU 名称推断数值等价。"""
    import torch
    distributions = ('numpy', 'torch', 'transformers', 'peft', 'accelerate',
                     'huggingface-hub', 'tokenizers', 'safetensors')
    versions = {}
    for name in distributions:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    gpu = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            gpu.append({'index': index, 'name': props.name,
                        'compute_capability': [props.major, props.minor],
                        'total_memory_bytes': props.total_memory})
    return {
        'python': platform.python_version(), 'python_executable': sys.executable,
        'platform': platform.platform(), 'packages': versions,
        'cuda_build': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
        'cuda_available': torch.cuda.is_available(), 'gpus': gpu,
        'dtype': str(dtype) if dtype is not None else None,
        'torch_initial_seed_at_recording': torch.initial_seed(),
        'deterministic_algorithms_at_recording': torch.are_deterministic_algorithms_enabled(),
        'deterministic_warn_only_at_recording': torch.is_deterministic_algorithms_warn_only_enabled(),
        'cudnn_benchmark_at_recording': torch.backends.cudnn.benchmark,
        'cudnn_deterministic_at_recording': torch.backends.cudnn.deterministic,
        'cuda_matmul_allow_tf32_at_recording': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_allow_tf32_at_recording': torch.backends.cudnn.allow_tf32,
        'torch_num_threads_at_recording': torch.get_num_threads(),
        'attention_implementation': 'sdpa',
        # 保留引擎 attn_implementation='sdpa' 的自动选择，不强制 math 后端。
        # enabled 只表示后端允许使用，并不声称每次前向实际选择了哪个实现。
        'sdpa_backend_enabled': {
            'flash': torch.backends.cuda.flash_sdp_enabled(),
            'memory_efficient': torch.backends.cuda.mem_efficient_sdp_enabled(),
            'math': torch.backends.cuda.math_sdp_enabled(),
            'cudnn': torch.backends.cuda.cudnn_sdp_enabled(),
        },
        'sdpa_backend_note': '允许的后端开关；实际选择由 PyTorch、硬件、dtype 和张量形状决定。',
    }

