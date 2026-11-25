# Generalization demo runbook

Open [the Colab notebook](https://colab.research.google.com/github/oscarwyx/LLM-generalization/blob/main/generalization/notebooks/generalization_presentation_demo.ipynb), choose **Runtime → Change runtime type → GPU**, and run cells 1–3 before the presentation. The first setup downloads the pinned Qwen3-0.6B base model (about 1.5 GB); the LoRA adapter is already in GitHub. Do not disconnect the runtime after loading.

During the presentation, show the initial 90-question and validation 810-question plots from cell 2. Explain that the 810-question paired EM-change difference is +3.40 percentage points (95% article-cluster CI +0.93 to +5.80). Then run cell 4 for one fixed live example. It generates four answers: Base and LoRA on original context and with 1,024 irrelevant tokens prepended. Cell 5 is an optional LoRA failure case and is disabled by default.

The live answers may differ slightly across GPUs. The saved table, figures, and confidence intervals remain available in cell 2 even if GPU inference or network loading fails. Do not present one illustrative question as the source of the aggregate result.

This notebook does not train, mount Drive, or regenerate the full 4,860 validation predictions. Its source commit is fixed at `bff11c7de6450bc1b6bd86682764cf621303c0ec`; the published manifest verifies the adapter and 900 test rows. The pinned base model revision is `c1899de289a04d12100db370d81485cdf75e47ca`.
