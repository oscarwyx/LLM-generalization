# Qwen3-0.6B LLM course project

This repository brings together four project parts. The first three contain completed work; `tiny_transformer/` is reserved for the remaining team component. The training source history and the inference teammate's source history are preserved in Git.

| Part | Contents | Start here |
|---|---|---|
| **Training** | Qwen3-0.6B baseline, prompting, LoRA training, fixed 900-question test set, and the published epoch-2 adapter | [training/README.md](training/README.md) |
| **Generalization** | Prepended irrelevant-context experiments, 90/810 results, analysis code, report, and live demo | [generalization/README.md](generalization/README.md) |
| **Inference** | Teammate's KV-cache, batch-size, and decoding benchmarks with saved results | [inference/README.md](inference/README.md) and [integration notes](inference/INTEGRATION.md) |
| **Tiny Transformer** | Future team component | [tiny_transformer/README.md](tiny_transformer/README.md) |

## Checkpoint and quick demo

Open the [unified Colab project demo](https://colab.research.google.com/github/oscarwyx/LLM-generalization/blob/main/project_demo.ipynb) for training, generalization, inference, and the future tiny Transformer section in one notebook. It reads saved results and runs on a CPU runtime. The separate generalization notebook below provides optional live GPU inference.

The [LoRA adapter](training/published/seed17_epoch2/adapter/) is included as ordinary Git files. The fixed Qwen3-0.6B base model (about 1.5 GB) downloads on first GPU inference; its revision is pinned in the [publication manifest](training/published/seed17_epoch2/manifest.json). The archived training checkpoint passes the manifest's file and data checks.

To verify the checkpoint without a GPU or download:

```bash
cd training
PYTHONPATH=src python -m genqa pretrained --verify-only
```

To reproduce the generalization statistics without inference:

```bash
cd generalization/repro
python reproduce.py
```

For a short live presentation, use the [Colab generalization demo](https://colab.research.google.com/github/oscarwyx/LLM-generalization/blob/main/generalization/notebooks/generalization_presentation_demo.ipynb) and its [runbook](generalization/presentation/demo_runbook.md). Run setup and model loading before the talk; the live cell generates only four example answers. The saved inference benchmark figures can be displayed without rerunning the full benchmark.

## Key findings

The published training run reached **77.11% EM** and **86.13% F1** for LoRA on the fixed 900-question SQuAD test set. In the 810-question generalization validation, the LoRA checkpoint's EM decline under 1,024 prepended irrelevant tokens was **3.40 percentage points smaller** than the base model's decline (95% article-cluster bootstrap CI **+0.93 to +5.80**). This is a within-SQuAD robustness result on questions from the same 45 articles.

The inference teammate measured a KV-cache throughput increase from **2.06 to 2.51 questions/s** (about 22%), a batch-size 1-to-16 increase from **1.27 to 5.70 questions/s**, and lower EM/F1 with beam widths 2 and 4 than with greedy decoding in that run. See the saved CSVs and provenance notes in [`inference/`](inference/).

## Sources and reproducibility

- [Training source](https://github.com/sunweiyi2026-ship-it/LLM-project) at commit `2359fd5d9ea34bb9d1845fe1a6732d657343361f`.
- [Inference source](https://github.com/ZZZeet/qwen3-inference-experiments) at commit `6560a4ef24d00c240d69754dc333c8b4f1745ce2`.
- [Generalization prediction-level bundle](generalization/repro/README.md), including frozen manifests and checksums.
- [Final generalization report section](generalization/report/generalization_final_section.md).

Run each component from its own directory as described in its README. The two archived result sets retain their original configurations and raw measurements.
