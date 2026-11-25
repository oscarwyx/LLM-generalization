# Generalization slides: content ready for final deck

## Slide: question and design

**Title:** Does irrelevant preceding context break QA?

- Published Qwen3-0.6B + epoch-2 LoRA; no retraining.
- 90 fixed questions, two per article, from the published 900-question test set.
- Same question, gold answer, prompt, and greedy decoding; add ~512 or ~1,024 irrelevant tokens before the answer-bearing paragraph.
- Two fixed distractor seeds; article-cluster bootstrap for paired changes.
- Predeclared prediction: LoRA EM falls by at least 8 percentage points at +1,024 tokens.

## Slide: result

Use `outputs/generalization/2026-11-15_seed17_epoch2_length/length_effects.png`.

**Callout:** LoRA EM: 78.89% original → 76.67% at +1,024 (−2.22 pp; 95% article-bootstrap CI −8.89 to +3.89). Point estimate does not meet the predicted 8-point decline. Baseline EM: 20.00% → 12.22% (−7.78 pp; CI −16.11 to −0.56).

**Initial 90-question comparison:** The exploratory difference in declines is +5.56 pp (CI −3.89 to +15.56), so this first sample alone does not establish a difference. Two distractor seeds are shown as x marks; LoRA +1,024 EM varies from 74.44% to 78.89% by seed.

Use `outputs/generalization/2026-11-15_seed17_epoch2_810_prepend/generalization_810_quantitative.png` to show the 810-question EM scores and the 90-versus-810 paired intervals.

**Follow-up on the remaining 810 questions:** Baseline EM fell 8.89 pp (22.96% → 14.07%); LoRA fell 5.49 pp (76.79% → 71.30%). The prespecified direct difference in changes was **+3.40 pp** (95% article-bootstrap CI **+0.93 to +5.80**). This supports a smaller LoRA EM drop for this checkpoint in this SQuAD preceding-context setting. The original ≥8 pp LoRA-drop prediction on the first 90 was still not met.

## Slide: failure cases and boundary

- **Apollo:** correct “immediately” becomes a date from the distractor under either seed.
- **Steam engine:** correct “Olympic class” becomes a different span from the original context.
- Narrow conclusion: robustness to irrelevant preceding SQuAD text on this sample. Added length and content vary together; this is not a cross-domain test.

Follow-up error note: In the 810 new questions, 149 seed-specific LoRA EM losses affected 104 distinct questions. Examples include a wrong title from the original ABC passage, an overlong Apollo answer containing the correct name, and two Luther predictions copied from distractors. Do not imply distractor copying is the dominant failure mode; only 2/149 wrong prediction strings were located solely in distractors.

Speaker note: The published full-900 LoRA EM is 77.11%; the 78.89% original score above is from the selected 90-question stress-test subset.
