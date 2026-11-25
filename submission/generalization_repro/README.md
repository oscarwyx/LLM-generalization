# Generalization experiment: reproducible results bundle

This bundle covers the Qwen3-0.6B base model and published seed-17, epoch-2 LoRA adapter on SQuAD 1.1 with irrelevant text **prepended** to the answer-bearing passage. It preserves the initial 90-question study and the later validation on the remaining 810 questions as separate analyses. The postposed-distractor control was not run and is not included as a result.

## What is fixed

- Source repository: <https://github.com/sunweiyi2026-ship-it/LLM-project>, commit `2359fd5d9ea34bb9d1845fe1a6732d657343361f`.
- Published run: `published/seed17_epoch2/manifest.json`, source run `20261115T182055_377596Z`.
- Base: `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca`.
- Adapter SHA-256: `c2e7a43bb695da3903c5406cab4bf6b22cd3d196ab488a2ec3c3588f0f925cac`.
- Test set: the fixed 900-question SQuAD 1.1 published test list. Initial 90 = two deterministically selected questions per each of 45 articles; validation 810 = the exact complement, 18 more per article.
- Distractors: passages from other test articles, with seeds 101 and 202. The main comparison adds approximately 1,024 tokens **before** the original passage. Prompt, checkpoint, greedy decoding, and 64 new-token cap are unchanged.
- Primary validation measure: `(LoRA EM change) − (Baseline EM change)`, paired by question, with the two seeds averaged per question. The interval resamples 45 article means with replacement, 2,000 draws and NumPy RNG seed 42.

The 810-question plan was written after seeing the 90-question results, but before generating answers for the 810. The original prediction that LoRA EM would fall by at least 8 pp on the first 90 was not met by the observed point estimate; it remains reported as such.

## Recompute results without a GPU

Use Python 3.12.7 and the tested analysis dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
sha256sum -c SHA256SUMS
python reproduce.py
python plot.py
```

On macOS, `shasum -a 256 -c SHA256SUMS` can replace `sha256sum -c`.

`reproduce.py` reads the **packaged prediction-level CSV files**, independently recalculates SQuAD EM/F1 from predictions and gold answers, checks the 90/810 split and all published condition summaries, and recomputes both article-bootstrap intervals. It writes `reproduced/verification.json` and new CSV tables. `plot.py` regenerates the figure into `reproduced/`. These scripts do not load the model or generate answers. Both compressed CSV files retain full question, answer, context, prediction, condition, and scoring columns.

Expected direct EM interactions:

| Study | LoRA change − Baseline change | 95% article-cluster interval |
|---|---:|---:|
| Initial 90 | +5.56 pp | −3.89 to +15.56 pp |
| Remaining 810 | +3.40 pp | +0.93 to +5.80 pp |

The second interval supports a **narrow** conclusion: this single LoRA checkpoint has a smaller EM decline than Baseline in this SQuAD prepended-context setting. The 810 questions are new questions from the same 45 articles, not new articles or a new domain. The two distractor seeds and one LoRA training seed limit broader claims.

## Re-run inference with a GPU

The two notebooks in `notebooks/` reproduce the model evaluations. Their setup cells install exact Colab package versions, clone and check out the pinned repository commit, verify the published data/adapter bundle, and load the pinned model revision. Run `initial_90_prepend_only.ipynb` first if you want to regenerate the initial study, then `generalization_810_prepend.ipynb` for the 810-question follow-up. Select a Google Colab GPU runtime. **Change each setup cell's `OUTPUT_DIR` to a new Drive folder** to generate fresh predictions while preserving the original run. Run each notebook top to bottom. Completed conditions are reused after a disconnection. The 810 notebook archives an incomplete condition and reruns that condition. Its final analysis cell can be run by itself after a restart, provided its Drive manifest and all six condition outputs exist. If you changed `OUTPUT_DIR` for a fresh run, set it to that same folder again before running the final cell alone.

The published adapter is already in the pinned GitHub repository; a backup is in [Google Drive](https://drive.google.com/drive/folders/1r6BTWhIgURTq7hfSIJrvzQqR2MSpsimW). The original completed [90-question run](https://drive.google.com/drive/folders/1QIU5KoyjxHKErXKyChWdzL4m9nVHRlSZ) and [810-question run](https://drive.google.com/drive/folders/1rtyW2oJjBtCQrDQV7I0hzVf9ltEFXS5E) remain in Drive. The package includes their analysis-ready prediction records, so Drive is not required to reproduce the reported statistics.

GPU inference may not reproduce every generated string byte-for-byte across different hardware or library builds. The packaged prediction-level data and CPU analysis reproduce the reported statistics exactly in the tested environment.

## Contents

- `data/initial_90/` and `data/validation_810/`: frozen manifests, compressed prediction-level CSVs, saved summary tables and confidence intervals. The validation folder also contains six condition-level run configurations and metrics, plus the 149-row LoRA error-location audit.
- `notebooks/`: inference and Colab analysis code.
- `reproduce.py`, `plot.py`, `requirements-analysis.txt`, `SHA256SUMS`: CPU replication and integrity checks.
- `figures/`: publication and slide versions of the quantitative chart.
- `writing/`: report section, slide content, and representative error review.

No model weights, personal credentials, Colab cache, or checkpoint training files are duplicated into this results bundle.
