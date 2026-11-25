# Published epoch-2 LoRA checkpoint and full run

Source run: `20261115T182055_377596Z`.

| Method | Test EM (%) | Test F1 (%) |
|---|---:|---:|
| Baseline | 22.1111 | 44.5315 |
| Prompting | 24.4444 | 49.1689 |
| LoRA | 77.1111 | 86.1260 |

From the repository root, install dependencies as described in the main README, then run:

```bash
python -m genqa pretrained --verify-only
python -m genqa pretrained --split test
```

`adapter/` contains the two epoch-2 LoRA weight/configuration files. `data/` contains the fixed training, validation, and test rows; `configs/` contains the original prompts; `reference/` contains predictions, metrics, logs, and provenance. The pinned Qwen3-0.6B base model is downloaded on first inference.

Re-evaluation writes a fresh `runs/pretrained_<timestamp>/` directory. It neither overwrites this published run nor retrains the model. Historical local paths in metadata were replaced with `<source_project>` solely for provenance and are not used for loading. The adapter configuration pins the public base model ID and revision; the adapter weight bytes were not changed.
