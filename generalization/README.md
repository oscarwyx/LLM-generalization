# Generalization

This part evaluates the published seed-17, epoch-2 LoRA checkpoint against its Qwen3-0.6B base model when irrelevant context is placed before the answer-bearing SQuAD 1.1 passage. The 90-question initial study and remaining 810-question validation are separate analyses. The original prediction of an 8-point LoRA EM decline was not met in the initial study. In the 810-question validation, LoRA lost 3.40 fewer EM points than the base model (95% article-cluster bootstrap CI +0.93 to +5.80). The questions come from the same 45 articles, so this is not cross-domain transfer.

- [`notebooks/generalization_length.ipynb`](notebooks/generalization_length.ipynb): initial 90-question experiment.
- [`notebooks/generalization_810_prepend.ipynb`](notebooks/generalization_810_prepend.ipynb): 810-question validation.
- [`notebooks/generalization_presentation_demo.ipynb`](notebooks/generalization_presentation_demo.ipynb): short Colab demo that shows saved results and runs four live answers.
- [`repro/`](repro/README.md): prediction-level data, CPU statistics, figures, checksums, and inference notebooks. From `repro/`, run `python reproduce.py` to recompute the reported statistics.
- [`report/generalization_final_section.md`](report/generalization_final_section.md): final English report section.
- [`presentation/`](presentation/demo_runbook.md): demo runbook and Overleaf source with both result figures.

The adapter and fixed 900-question test set live in [`../training/published/seed17_epoch2/`](../training/published/seed17_epoch2/). The experiment notebooks pin the original training repository commit; they do not retrain the model. The demo notebook pins this repository's archived pre-reorganization commit until its paths are updated in the next commit.
