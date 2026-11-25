# Published repository

This repository is published at <https://github.com/oscarwyx/LLM-generalization>.

`origin` points to this repository. `upstream` points to the original group training repository at <https://github.com/sunweiyi2026-ship-it/LLM-project>. The original training commit and its history are preserved; the generalization work is a later commit.

To get an independent copy and verify the packaged predictions:

```bash
git clone https://github.com/oscarwyx/LLM-generalization.git
cd LLM-generalization/generalization/repro
shasum -a 256 -c SHA256SUMS
python reproduce.py
```

The 20 MB LoRA adapter is tracked in `training/published/seed17_epoch2/adapter/`. The approximately 1.5 GB Qwen3-0.6B base model is downloaded from the revision pinned in `training/published/seed17_epoch2/manifest.json` when running inference.
