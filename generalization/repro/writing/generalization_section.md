# Generalization: robustness to irrelevant preceding context

## Research question and prediction

We asked whether extra irrelevant context placed before the answer-bearing passage would degrade short-answer generation on held-out SQuAD 1.1 articles. Before running the stress test, we predicted that adding approximately 1,024 tokens would lower the published epoch-2 LoRA model's exact match (EM) by at least 8 percentage points relative to the same questions without extra context.

## Experimental design

We used the verified `published/seed17_epoch2` release at GitHub commit `2359fd5d9ea34bb9d1845fe1a6732d657343361f`, including its fixed Qwen3-0.6B model revision, baseline prompt, LoRA adapter, and 900-question test set. Two questions per article were selected deterministically, yielding 90 questions from 45 articles. For each question, the original answer-bearing context was retained and unrelated passages from other test articles were prepended to add approximately 512 or 1,024 tokens. Two fixed distractor seeds were used at each length. Baseline and LoRA were compared on identical inputs using the repository's non-thinking, greedy, 64-new-token generation and SQuAD EM/F1 scorer. No further training took place. The 90-question subset was re-evaluated on the Colab runtime for every condition, including the original condition, so the paired comparisons use the same hardware and software path.

All 900 generated records (90 questions × 2 methods × 5 conditions) were checked against the released IDs, answers, and scorer; none was truncated. The original-condition LoRA outputs matched the published reference predictions for all 90 selected questions. Baseline output wording differed on 2/90, although both retained EM=0. These cross-runtime wording differences reinforce the importance of comparing conditions within this Colab run.

## Results

| Extra context | Baseline EM | LoRA EM | Baseline F1 | LoRA F1 |
|---:|---:|---:|---:|---:|
| 0 tokens | 20.00% | 78.89% | 42.81% | 88.20% |
| ~512 tokens | 16.67% | 76.11% | 40.60% | 85.99% |
| ~1,024 tokens | 12.22% | 76.67% | 37.22% | 85.80% |

At the two stressed lengths, the table averages scores across the two distractor seeds. At +1,024 tokens, LoRA's paired EM change was **−2.22 percentage points** (95% article-cluster bootstrap interval **−8.89 to +3.89**). The predeclared 8-point drop was **not met by the point estimate**. Its interval spans both little or no change and an 8-point decline, so the data do not tightly pin down the true effect size. The corresponding baseline change was **−7.78 points** (95% interval **−16.11 to −0.56**). An exploratory difference between the LoRA and baseline changes was **+5.56 points** (95% interval **−3.89 to +15.56**); because this interval crosses zero, we cannot establish that LoRA is more robust than the baseline. LoRA's +1,024-token EM differed across distractor seeds (74.44% versus 78.89%), indicating sensitivity to the added content.

The confidence intervals resample articles while treating the two distractor seeds as fixed. They do not quantify variability over all possible distractor passages or over additional model-training seeds.

## Follow-up validation on the remaining 810 questions

After inspecting the 90-question result, we froze a follow-up plan for the remaining 810 questions in the published test set: 18 further questions from each of the same 45 articles. We retained the published checkpoint, prompt, greedy decoding, distractor construction, and seeds 101 and 202, and evaluated only original and +1,024-token prepended conditions. The primary estimand, fixed before the new inference, was the **LoRA EM change minus the Baseline EM change** on matched questions, averaging the two distractor seeds per question and bootstrapping the 45 articles. All six conditions completed with 810 predictions each and no input truncation.

| Method | Original EM | +1,024 EM, mean of two seeds | Paired change | 95% article-bootstrap CI |
|---|---:|---:|---:|---:|
| Baseline | 22.96% | 14.07% | −8.89 pp | −10.99 to −6.73 pp |
| LoRA | 76.79% | 71.30% | −5.49 pp | −7.72 to −3.39 pp |

The direct **LoRA-minus-Baseline change was +3.40 pp** (95% article-cluster bootstrap interval **+0.93 to +5.80 pp**). This meets the follow-up's stated narrow criterion: the interval's lower bound is above zero. The secondary F1 interaction was +3.18 pp (95% interval +0.78 to +5.50 pp). Thus, this checkpoint lost less EM than the Baseline in this specific SQuAD preceding-context test. This result does not revise the earlier 90-question estimate or turn the original ≥8 pp LoRA-drop prediction into a success. Because the new questions came from the same 45 articles, this is validation across questions within those articles, not transfer to unseen articles or domains. The two distractor seeds are fixed, and there is one LoRA training seed.

The publication-ready figure is archived as `generalization_810_quantitative.svg` (with a PNG copy for slides). The complete 810-question predictions and manifest remain in [Google Drive](https://drive.google.com/drive/folders/1rtyW2oJjBtCQrDQV7I0hzVf9ltEFXS5E). A summary, paired effects, condition scores, and audit are archived in Desktop `outputs/generalization/2026-11-15_seed17_epoch2_810_prepend/`.

## Error analysis

For an Apollo-program question asking how soon NASA created an accident review board, LoRA changed from the correct “immediately” to “winter of 1756” or “January 4, 1996,” depending on the distractor seed. Both wrong answers occur in the prepended passage and not in the original answer-bearing context, illustrating distractor answer substitution. In a Steam-engine question asking for a passenger-ship class, LoRA changed from “Olympic class” to “4-cylinder triple-expansion engine” under both seeds. That wrong span occurs in the original context, showing that added preceding text can alter selection even without copying the distractor. These cases are diagnostic examples, not estimates of failure-type prevalence.

### Error patterns in the 810-question follow-up

Among LoRA responses that were EM-correct without extra text but EM-incorrect under +1,024 tokens, there were 149 seed-specific loss records across 104 distinct questions. We classified the location of each **normalized exact wrong-prediction string** in the original answer-bearing passage and the prepended distractor:

| Location of wrong-prediction string | Records | Share of 149 |
|---|---:|---:|
| Original passage only | 138 | 92.62% |
| Distractor only | 2 | 1.34% |
| Both passages | 2 | 1.34% |
| Neither passage | 7 | 4.70% |
| **Total** | **149** | **100.00%** |

Thus, the **remaining 9 records** beyond the 138 original-only and 2 distractor-only cases comprise **2 found in both passages and 7 found in neither**. “Neither” means no exact normalized string match; it does not by itself prove that the model hallucinated, because wording or inflection can differ from the source. Forty-one wrong predictions still contained the originally correct prediction, illustrating answer-boundary errors under strict EM. For example, the ABC question changed from “The Dating Game” to “The Newlywed Game,” a different title in the original passage. An Apollo answer expanded from “Wernher von Braun” to “Wernher von Braun and his team of rocket engineers,” retaining answer content but failing EM. Two Luther questions produced “Trotsky” or a long phrase about families and churches copied from unrelated distractors. These are checked diagnostic examples; source-string location alone does not establish the model's causal mechanism. The full classification and examples are in `outputs/generalization/2026-11-15_seed17_epoch2_810_prepend/representative_failures.md`.

## Interpretation and limitations

This experiment measures robustness to extra irrelevant text on a fixed subset of the published SQuAD test set. It does not establish cross-domain transfer, and added length is confounded with added content. The original 900-question publication results (LoRA EM 77.11%, F1 86.13%; baseline EM 22.11%, F1 44.53%) describe a different sample size and should not replace the 90-question original-condition scores in this paired analysis. Only one trained LoRA seed was tested. The public SQuAD dev set is not a hidden benchmark.

**Evidence:** Desktop `outputs/generalization/2026-11-15_seed17_epoch2_length/` contains the run manifest, all predictions, score tables, article-bootstrap effects, audit, error cases, and figure. The Colab originals remain in the linked Google Drive run folder.
