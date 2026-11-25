# Inference experiment integration

This directory is a Git subtree of [ZZZeet/qwen3-inference-experiments](https://github.com/ZZZeet/qwen3-inference-experiments) at source commit `6560a4ef24d00c240d69754dc333c8b4f1745ce2`. The merge commit preserves that repository's history and author attribution. All saved CSV, JSON, and PNG result files are copied unchanged.

The benchmark uses the same seed-17, epoch-2 LoRA publication as this repository: base-model revision `c1899de289a04d12100db370d81485cdf75e47ca` and publication manifest SHA-256 `d68443d88ae58e32b8337c0ad52cbb3ec65de2b28e2fc6bfc71a55c90623ab2e`. Its per-question files contain the same 900 published test IDs. Measurements were recorded on an NVIDIA T4 in FP16. The experiment notebook clones the original team repository; verify that its loaded publication manifest matches the recorded hash before rerunning it.

## Reported results

- **KV cache:** throughput 2.06 to 2.51 questions/s (about +22%); peak extra GPU allocation 161.54 to 510.05 MiB (+348.51 MiB). EM was 77.00% with cache off and 76.89% with cache on. The decoded answer matched on 899 of 900 questions across cache settings. Source: `results-KV cache/summary.csv`.
- **Batch size:** throughput 1.27 to 5.70 questions/s from batch 1 to 16 (4.49×), while peak allocated GPU memory rose from 1,315.95 to 2,506.69 MiB. Source: `results-batch size/summary.csv`.
- **Decoding:** greedy achieved 76.89% EM and 86.03% F1. Beam widths 2 and 4 took 2.51× and 6.06× its generation time and yielded lower EM/F1 in this run. Source: `results-decoding strategy/summary.csv`.

The imported notebook's KV-cache conclusion originally contained stale numbers: 2.11 to 2.77 questions/s, 32%, and 159.60 MiB. This combined repository changes only that explanatory paragraph to match the archived CSV; the source commit remains recoverable in Git history.

For a live presentation, display the saved figures and run the short [`generalization_presentation_demo.ipynb`](../generalization/notebooks/generalization_presentation_demo.ipynb) example. Running all inference benchmark cells would regenerate thousands of answers and take substantially longer than the talk.
