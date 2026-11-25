"""发布模型接口的 CPU/mock 测试：哈希、来源一致性和真正调用评测的参数。

测试不下载权重，不导入 CUDA 库，不训练；小型假权重仅用于文件校验测试，
不能冒充对真实模型推理的验证。真实发布权重另由 verify_published 校验。
"""
from copy import deepcopy
import builtins
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import weakref
from unittest.mock import patch

from genqa import engine, pretrained, runtime


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


class PublicationFixture:
    def __init__(self, root):
        self.root = root
        self.folder = root / pretrained.DEFAULT_PUBLICATION
        self.folder.mkdir(parents=True)
        self.model = {'model_id': runtime.MODEL_ID, 'revision': runtime.MODEL_REVISION, 'files': []}
        for name in runtime.MODEL_FILENAMES:
            content = ('test model file: ' + name).encode()
            local = root / 'cache/model' / name
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(content)
            self.model['files'].append({'name': name, 'bytes': len(content),
                                       'sha256': hashlib.sha256(content).hexdigest()})
        self.config = {
            'model': self.model, 'training': {**deepcopy(engine.DEFAULT_TRAINING),
                'seed': 17, 'train_examples': 2048, 'max_train_tokens': 1_000_000},
            'evaluation': {'seed': 42, 'batch_size': 4, 'max_input_tokens': 8192,
                           'max_new_tokens': 64, 'loss_examples_per_split': 64},
            'runtime': {'precision': 'auto', 'strict_determinism': True},
        }
        write_json(root / 'configs/experiment.json', self.config)
        self.prompt = {'name': 'baseline', 'instruction': 'Return a short answer.',
                       'target_template': '{instruction}\n{context}\n{question}',
                       'demonstrations': [], 'enable_thinking': False}
        write_json(self.folder / 'configs/baseline.json', self.prompt)
        self.rows = {}
        data = {}
        for split, n in pretrained.EXPECTED_COUNTS.items():
            selected = [{'id': f'{split}-{i}', 'context': f'Context {split} {i}.',
                         'question': f'Question {split} {i}?', 'answers': [str(i)],
                         'article_title': f'Article {split} {i % 45}', 'split': split}
                        for i in range(n)]
            filename = 'data/' + {'train': 'train_selected', 'validation': 'validation', 'test': 'squad_test'}[split] + '.jsonl'
            engine.save_jsonl(self.folder / filename, selected)
            self.rows[split] = selected
            data[split] = {'path': filename, 'n': n, 'rows_sha256': engine.object_sha256(selected)}
        self.original = {'formal_run': True, 'smoke': False, 'config': self.config,
                         'active_counts': deepcopy(pretrained.EXPECTED_COUNTS),
                         'active_rows_sha256': {s: engine.object_sha256(r) for s, r in self.rows.items()},
                         'prompt_sha256': {'baseline': engine.object_sha256(self.prompt)}}
        write_json(self.folder / 'reference/run_manifest.json', self.original)
        self.training_config = {'complete': True, 'train_hparams': deepcopy(engine.DEFAULT_TRAINING),
            'seed': 17, 'model_id': engine.MODEL_ID, 'model_revision': engine.MODEL_REVISION,
            'fresh_adapter': True, 'resume_from': None,
            'training_rows_sha256': engine.object_sha256(self.rows['train']),
            'prompt_sha256': engine.object_sha256(self.prompt)}
        write_json(self.folder / 'reference/training_run_config.json', self.training_config)
        self.summary = {'complete': True, 'formal_training_complete': True, 'seed': 17,
            'completed_epochs': 2, 'examples_presented': 4096, 'global_step': 256,
            'base_parameters_unchanged': True, 'adapter_parameters_changed': True,
            'budget_satisfied': True, 'consumed_nonpadding_input_tokens': 939880}
        write_json(self.folder / 'reference/training_summary.json', self.summary)
        self.adapter = {'peft_type': 'LORA', 'task_type': 'CAUSAL_LM', 'bias': 'none',
            'r': 8, 'lora_alpha': 16, 'lora_dropout': .05,
            'target_modules': deepcopy(engine.DEFAULT_TRAINING['target_modules']),
            'base_model_name_or_path': engine.MODEL_ID, 'revision': engine.MODEL_REVISION}
        write_json(self.folder / 'adapter/adapter_config.json', self.adapter)
        (self.folder / 'adapter/adapter_model.safetensors').write_bytes(b'fixture-only-not-real-weights')
        self.manifest = {'schema_version': 1, 'kind': 'genqa_lora_publication', 'model': self.model,
            'training': {'seed': 17, 'epoch': 2, 'train_examples': 2048},
            'runtime': self.config['runtime'], 'evaluation': self.config['evaluation'],
            'adapter_dir': 'adapter', 'prompt': 'configs/baseline.json', 'data': data,
            'source_run': {'name': 'formal_fixture', 'source_commit': 'a' * 40,
                          'run_manifest': 'reference/run_manifest.json',
                          'training_run_config': 'reference/training_run_config.json',
                          'training_summary': 'reference/training_summary.json'}, 'files': {}}
        self.refresh()

    def refresh(self):
        self.manifest['files'] = {p.relative_to(self.folder).as_posix():
            {'bytes': p.stat().st_size, 'sha256': engine.file_sha256(p)}
            for p in sorted(self.folder.rglob('*')) if p.is_file() and p.name != 'manifest.json'}
        self.write_manifest()

    def write_manifest(self):
        write_json(self.folder / 'manifest.json', self.manifest)


class PretrainedTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fx = PublicationFixture(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def test_verify_uses_no_gpu_imports_or_downloads(self):
        original_import = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in {'torch', 'transformers', 'peft', 'huggingface_hub'}:
                raise AssertionError('verify-only attempted GPU/network library import')
            return original_import(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=guarded), patch.object(runtime, 'resolve_model') as download:
            result = pretrained.verify_published(self.root)
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['counts'], pretrained.EXPECTED_COUNTS)
        self.assertFalse(result['model_files_checked'])
        download.assert_not_called()

    def test_explicit_model_check_is_offline_and_detects_bad_cache(self):
        with patch.object(runtime, 'resolve_model') as download:
            result = pretrained.verify_published(self.root, model_dir='cache/model')
            self.assertTrue(result['model_files_checked'])
            (self.root / 'cache/model/model.safetensors').unlink()
            with self.assertRaises(FileNotFoundError):
                pretrained.verify_published(self.root, model_dir='cache/model')
            download.assert_not_called()

    def test_weight_tampering_and_missing_required_checksum_are_rejected(self):
        path = self.fx.folder / 'adapter/adapter_model.safetensors'
        before = path.read_bytes()
        path.write_bytes(b'x' * len(before))
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            pretrained.verify_published(self.root)
        path.write_bytes(before)
        del self.fx.manifest['files']['adapter/adapter_model.safetensors']
        self.fx.write_manifest()
        with self.assertRaisesRegex(ValueError, '没有列入'):
            pretrained.verify_published(self.root)

    def test_cross_platform_path_escape_is_rejected(self):
        for name in ('../outside', '/absolute', 'C:/outside', 'adapter\\outside', './adapter'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                pretrained._inside(self.fx.folder, name)

    def test_config_and_data_cannot_silently_drift(self):
        self.fx.manifest['evaluation'] = {**self.fx.manifest['evaluation'], 'max_new_tokens': 128}
        self.fx.write_manifest()
        with self.assertRaisesRegex(ValueError, '配置与原始运行'):
            pretrained.verify_published(self.root)
        self.fx.manifest['evaluation'] = deepcopy(self.fx.config['evaluation'])
        selected = self.fx.rows['test']
        selected[0]['question'] = 'Changed question'
        engine.save_jsonl(self.fx.folder / 'data/squad_test.jsonl', selected)
        self.fx.manifest['data']['test']['rows_sha256'] = engine.object_sha256(selected)
        self.fx.refresh()
        with self.assertRaisesRegex(ValueError, '数据内容/顺序与原运行'):
            pretrained.verify_published(self.root)

    def test_unfinished_training_and_wrong_adapter_are_rejected(self):
        self.fx.summary['formal_training_complete'] = False
        write_json(self.fx.folder / 'reference/training_summary.json', self.fx.summary)
        self.fx.refresh()
        with self.assertRaisesRegex(ValueError, '尚未完整完成'):
            pretrained.verify_published(self.root)
        self.fx.summary['formal_training_complete'] = True
        write_json(self.fx.folder / 'reference/training_summary.json', self.fx.summary)
        self.fx.adapter['r'] = 16
        write_json(self.fx.folder / 'adapter/adapter_config.json', self.fx.adapter)
        self.fx.refresh()
        with self.assertRaisesRegex(ValueError, 'adapter 身份'):
            pretrained.verify_published(self.root)

    def test_adapter_must_pin_public_base_id_and_revision(self):
        for field, wrong_value in [('base_model_name_or_path', 'C:/author/old-cache'),
                                   ('revision', None), ('revision', 'main')]:
            with self.subTest(field=field, value=wrong_value):
                adapter = {**self.fx.adapter, field: wrong_value}
                write_json(self.fx.folder / 'adapter/adapter_config.json', adapter)
                self.fx.refresh()
                with self.assertRaisesRegex(ValueError, '固定的模型 ID 和 revision'):
                    pretrained.verify_published(self.root)

    def test_loading_passes_verified_local_base_and_adapter(self):
        from genqa import workflow
        fake_model = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(dtype='mock-bf16')]))
        with patch.object(workflow, 'setup_reproducibility', return_value={'seed': 17}) as repro, \
             patch.object(workflow, 'check_cuda', return_value={'passed': True}), \
             patch.object(runtime, 'resolve_model', return_value=self.root / 'elsewhere') as resolve, \
             patch.object(engine, 'load_tokenizer', return_value='tokenizer') as tokenizer, \
             patch.object(engine, 'load_model', return_value=fake_model) as model, \
             patch.object(runtime, 'environment_info', return_value={'gpu': 'mock'}):
            loaded = pretrained.load_published_model(self.root, model_dir='elsewhere')
        repro.assert_called_once_with(self.fx.config)
        resolve.assert_called_once_with(self.root, 'elsewhere')
        tokenizer.assert_called_once_with(str(self.root / 'elsewhere'), local_files_only=True)
        model.assert_called_once_with(str(self.root / 'elsewhere'), precision='auto',
            adapter_dir=self.fx.folder / 'adapter', local_files_only=True)
        self.assertIs(loaded['model'], fake_model)
        self.assertEqual(loaded['runtime_info']['reproducibility']['seed'], 17)

    def mocked_loaded(self):
        loaded = pretrained._verified_bundle(self.root, pretrained.DEFAULT_PUBLICATION)
        loaded.update(model=object(), tokenizer='tokenizer', model_dir=self.root / 'cache/model',
                      runtime_info={'environment': {'gpu': 'mock'}})
        return loaded

    def test_failed_adapter_load_releases_frame_references_and_preserves_exception(self):
        from genqa import workflow
        class FakeCudaResource:
            pass
        resource_refs = []
        original_error = RuntimeError('adapter shape mismatch')
        def failed_loader(*args, **kwargs):
            cuda_model = FakeCudaResource()
            resource_refs.append(weakref.ref(cuda_model))
            raise original_error
        def checked_cleanup():
            # 即使原异常仍被 Notebook/测试持有，已退出的加载帧也不应继续持有模型。
            self.assertEqual(len(resource_refs), 1)
            self.assertIsNone(resource_refs[0]())
        with patch.object(workflow, 'setup_reproducibility', return_value={}), \
             patch.object(workflow, 'check_cuda', return_value={}), \
             patch.object(runtime, 'resolve_model', return_value=self.root / 'cache/model'), \
             patch.object(engine, 'load_tokenizer', return_value='tokenizer'), \
             patch.object(engine, 'load_model', side_effect=failed_loader), \
             patch.object(engine, 'free_memory', side_effect=checked_cleanup) as cleanup:
            with self.assertRaises(RuntimeError) as captured:
                pretrained.load_published_model(self.root)
        self.assertIs(captured.exception, original_error)
        self.assertEqual(str(captured.exception), 'adapter shape mismatch')
        cleanup.assert_called_once()

    def test_eval_calls_actual_engine_with_all_frozen_rows_and_records_new_run(self):
        loaded = self.mocked_loaded()
        with patch.object(pretrained, 'load_published_model', return_value=loaded), \
             patch.object(engine, 'evaluate', return_value={'newly_generated': True}) as evaluate, \
             patch.object(engine, 'free_memory') as cleanup, \
             patch.object(engine, 'train_lora') as train:
            result = pretrained.evaluate_published(self.root, split='test')
        args, kwargs = evaluate.call_args
        self.assertEqual(len(args[2]), 900)
        self.assertEqual(engine.object_sha256(args[2]), self.fx.original['active_rows_sha256']['test'])
        self.assertEqual(args[3], self.fx.prompt)
        self.assertEqual({k: kwargs[k] for k in self.fx.config['evaluation']}, self.fx.config['evaluation'])
        self.assertTrue(Path(result['run_dir']).name.startswith('pretrained_'))
        self.assertEqual(engine.load_json(Path(result['run_dir']) / 'status.json')['status'], 'completed')
        self.assertFalse(engine.load_json(Path(result['run_dir']) / 'run_manifest.json')['training_performed'])
        self.assertNotIn('model', loaded)
        train.assert_not_called()
        cleanup.assert_called_once()

    def test_failed_eval_is_not_marked_complete_and_existing_output_is_never_reused(self):
        destination = self.root / 'runs/failure'
        with patch.object(pretrained, 'load_published_model', return_value=self.mocked_loaded()), \
             patch.object(engine, 'evaluate', side_effect=RuntimeError('mock generation failure')), \
             patch.object(engine, 'free_memory') as cleanup:
            with self.assertRaisesRegex(RuntimeError, 'mock generation failure'):
                pretrained.evaluate_published(self.root, output_dir=destination)
        self.assertEqual(engine.load_json(destination / 'status.json')['status'], 'failed')
        cleanup.assert_called_once()
        with patch.object(pretrained, 'load_published_model') as load:
            with self.assertRaises(FileExistsError):
                pretrained.evaluate_published(self.root, output_dir=destination)
            load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
