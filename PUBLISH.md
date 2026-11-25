# Publish this local repository

This folder is a Git repository with the original group history and a separate commit for the generalization work. The original group remote is named `upstream`; there is deliberately no `origin` remote, to avoid pushing into the group repository by accident.

1. On your own GitHub account, create an **empty** repository named `LLM-generalization`. Do not initialize it with a README, license, or `.gitignore`.
2. In this folder, run:

```bash
git remote add origin https://github.com/YOUR_USERNAME/LLM-generalization.git
git push -u origin main
```

3. Open `https://github.com/YOUR_USERNAME/LLM-generalization` and confirm `submission/generalization_repro/`, `notebooks/generalization_810_prepend.ipynb`, and `published/seed17_epoch2/adapter/adapter_model.safetensors` are visible. Then send that repository URL to your group.

The adapter is about 20 MB, below GitHub's 100 MB single-file limit. The full base model is downloaded from the pinned public revision and is not stored in this repository.
