# SQuAD 1.1 generalization study: Qwen3-0.6B + LoRA

This repository combines a published group training run with a follow-up generalization study. The training code, fixed test set, and seed-17 epoch-2 LoRA adapter come from [sunweiyi2026-ship-it/LLM-project](https://github.com/sunweiyi2026-ship-it/LLM-project), commit `2359fd5d9ea34bb9d1845fe1a6732d657343361f`. Its Git history is preserved. The generalization notebooks, prediction-level results, analysis scripts, figures, and report section are the subsequent additions.

## Main finding

On 810 held-out questions from the same 45 SQuAD articles, prepending approximately 1,024 irrelevant tokens lowered exact match (EM) by 8.89 percentage points for the base model and 5.49 points for LoRA. The paired difference in declines was **+3.40 points** (95% article-cluster bootstrap CI **+0.93 to +5.80**). This supports a smaller EM decline for this checkpoint in this SQuAD condition. It does not establish transfer to unseen articles or domains. The initial 90-question study and its unmet original prediction remain reported separately.

## Start here

- [Generalization reproducibility guide](submission/generalization_repro/README.md): recompute both studies from compressed prediction-level CSV files without a GPU; includes fixed checksums and plotting code.
- [Final report section](submission/report/generalization_final_section.md): research question, design, statistical results, error audit, and limitations.
- [Generalization notebooks](notebooks/): original 90-question study and 810-question validation. They pin the original group commit and reuse the published checkpoint; neither retrains the model.
- [Presentation source](submission/presentation/overleaf/main.tex): includes both experiment figures and speaker notes.
- [Colab presentation demo](notebooks/generalization_presentation_demo.ipynb): shows saved results and runs one fixed live Base-versus-LoRA example without retraining. Follow the [demo runbook](submission/presentation/demo_runbook.md).
- [Published checkpoint](published/seed17_epoch2/adapter/): the 20 MB LoRA adapter is tracked in Git. The pinned Qwen3-0.6B base model (approximately 1.5 GB) is downloaded on first inference. See [manifest.json](published/seed17_epoch2/manifest.json) for its revision and file hashes.

## Training environment

The original training workflow uses Python 3.12, Linux or Windows, and an NVIDIA GPU with a compatible CUDA build. The reference hardware was an 8 GB RTX 5060. Training is not supported by this implementation on CPU or macOS. The CUDA 12.8 installation command below applies only to compatible drivers and GPUs; the program performs an actual CUDA computation check.

```bash
git clone https://github.com/oscarwyx/LLM-generalization.git
cd LLM-generalization
conda create -n genqa python=3.12 -y
conda activate genqa
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install -e .
```

The first run downloads public source data and the pinned base model. It also needs disk space for data, cache, predictions, and the adapter.

## Use the checkpoint without retraining

The adapter consists of [adapter_config.json](published/seed17_epoch2/adapter/adapter_config.json) and [adapter_model.safetensors](published/seed17_epoch2/adapter/adapter_model.safetensors). These are LoRA weights, not the full base model. They are ordinary Git files; Git LFS is not needed.

Verify the published files without a GPU or base-model download, then generate and score the fixed 900 test questions:

```bash
python -m genqa pretrained --verify-only
python -m genqa pretrained --split test
```

The second command downloads the pinned base model if needed, performs inference without training, and writes a new `runs/pretrained_<timestamp>/test/` directory. You can replace `test` with `validation` or `train`. To reuse a cached snapshot, append `--model-dir /path/to/qwen-snapshot`.

For a custom inference demo, load the model object directly:

```python
from genqa.pretrained import load_published_model
bundle = load_published_model(root=".")  # Run from the repository root; model_dir is optional.
model, tokenizer = bundle["model"], bundle["tokenizer"]
prompt = bundle["prompt"]
test_rows = bundle["rows"]["test"]
```

The published setup keeps the original precision, unmerged LoRA, non-thinking mode, greedy decoding, and batch size 4. Record any changes in precision, quantization, merging, or decoding as a new inference experiment. Generated text need not match byte-for-byte across hardware and library versions.

## Re-run the original training workflow

```bash
python -m genqa prepare --config configs/experiment.json
python -m genqa run --config configs/experiment.json
python -m genqa report --run-dir runs/<run_id>
```

`run` prepares the data and creates a new `runs/<run_id>/` with predictions, training logs, and a report. It starts a fresh LoRA adapter from the pinned base model. It does not resume the published adapter or reuse saved predictions. A small pipeline check is available with `python -m genqa run --config configs/experiment.json --smoke`; its 16/8/8 train/validation/test samples are not formal results.

For Jupyter, install a kernel with `python -m ipykernel install --user --name genqa --display-name "Python (GenQA)"`, then open [notebooks/experiment.ipynb](notebooks/experiment.ipynb). Re-running that notebook performs the experiment again; opening saved output only displays prior results.

## Published full training run

The published run ID is `20261115T182055_377596Z`. Training seed 17 used 2,048 examples and two epochs. All three methods were evaluated on 2,048 training questions, 256 validation questions, and 900 test questions. LoRA was also evaluated after each epoch.

| Method | Test EM (%) | Test F1 (%) |
|---|---:|---:|
| Baseline | 22.1111 | 44.5315 |
| Three-shot prompting | 24.4444 | 49.1689 |
| LoRA | 77.1111 | 86.1260 |

The run consumed **939,880 non-padding training input tokens**, including **25,392 supervised target tokens**, under a **1,000,000-token** limit, with zero overflow retries. The target tokens are a subset of input tokens, not an additional amount. Inference tokens are outside this training limit.

Reference artifacts: [metrics](published/seed17_epoch2/reference/metrics.csv), [900 test answers](published/seed17_epoch2/reference/test_answers.csv), [result plot](published/seed17_epoch2/reference/results.png), [token budget](published/seed17_epoch2/reference/budget_usage.json), and [run manifest](published/seed17_epoch2/manifest.json). The training implementation derives from [commit `36b3ddc`](https://github.com/sunweiyi2026-ship-it/LLM-project/tree/36b3ddc545d5fafbd09b0eefe1b2557f736a62b9).

![Published training results](published/seed17_epoch2/reference/results.png)

## Data and evaluation

Question, context, and answer text come from the [MRQA 2019 SQuAD subset](https://github.com/mrqa/MRQA-Shared-Task-2019); the [official SQuAD 1.1 release](https://rajpurkar.github.io/SQuAD-explorer/) restores article identity. `release/v2` in the MRQA URL names an MRQA release, not SQuAD 2.0. All included questions are answerable.

Data preparation normalizes duplicates, filters lexical near-duplicates using five-word shingles, maps articles, and splits training/validation articles with data seed 42. The 900 test questions are 20 questions from each of 45 public development articles, kept separate from the training/validation articles. This is a public development split, not a hidden benchmark. It differs from the older project's 900-question set, so older scores should not be compared directly.

Baseline uses a zero-shot short-answer prompt. Three-shot prompting adds fixed examples from three distinct training articles and does not update model weights. LoRA uses the baseline prompt, training seed 17, two epochs, rank 8, alpha 16, and dropout 0.05. Epoch 2 was fixed for final evaluation, not selected by test performance.

EM and token F1 score free-form generated answers after standard SQuAD/MRQA normalization and take the best reference match. Generation uses non-thinking mode, greedy decoding, at most 64 new tokens, and a maximum total input length of 8,192 tokens. Evaluation NLL, perplexity, and teacher-forced accuracy are diagnostic metrics on up to 64 fixed examples per split; EM/F1 use each entire split. Training-set scores are fitting diagnostics, especially because prompting examples come from that set.

The training budget counts every non-padding token in each training forward pass, including prompt, answer, and EOS. Data preparation checks the two-epoch budget before training; FP16 overflow retries also count. Runs over budget stop and are marked incomplete. The historical 935,638-token figure relates to a different data preparation and is not this run's result.

## Verification scope

The original training work passed 25 CPU unit tests, reconstructed the public data twice with identical hashes, and completed a small end-to-end GPU smoke run on Windows/Python 3.12/RTX 5060. The formal two-epoch training run and all three full evaluations completed. A reload of the published adapter on the reference machine reproduced all 900 test answer strings, scores, output-token counts, and reported metrics without retraining; see [validation/checks.json](validation/checks.json). These checks do not guarantee identical output on other hardware or software builds. The [experiment configuration](configs/experiment.json) uses repository-relative paths.
