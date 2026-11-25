# Representative error review: 810-question follow-up

Source: completed LoRA predictions for both +1,024-token seeds in the [Drive run](https://drive.google.com/drive/folders/1rtyW2oJjBtCQrDQV7I0hzVf9ltEFXS5E), the published SQuAD test contexts, and `lora_failures_1024.csv`. This review uses only records where the same LoRA checkpoint was EM-correct on the original context and EM-incorrect after a prepended distractor.

## Counts

- 149 wrong stressed-condition records across the two fixed seeds, involving 104 distinct questions; 45 questions failed under both seeds. These are transitions, not 149 unique questions.
| Location of normalized exact wrong-prediction string | Records | Share |
|---|---:|---:|
| Original passage only | 138 | 92.62% |
| Distractor only | 2 | 1.34% |
| Both passages | 2 | 1.34% |
| Neither passage | 7 | 4.70% |
| **Total** | **149** | **100.00%** |

The remaining 9 cases are 2 in both passages and 7 in neither. “Neither” denotes lack of an exact normalized substring match, not proof of hallucination. This source-location check does not establish why the model selected an answer.
- In 41/149 records, the wrong exact-match prediction still contains the original correct prediction as a proper substring. These include overlong answers; EM therefore drops even when some answer content is retained.
- Net LoRA EM change is smaller than the gross loss count because some originally incorrect predictions become correct under stress: 60 such seed-specific gains versus 149 losses, a net 89 fewer correct predictions over 1,620 stressed records.

## Checked examples

| Pattern | Question | Original prediction (correct) | +1,024 prediction (wrong) | Evidence |
|---|---|---|---|---|
| Wrong original-context span | Which blind-date concept program debuted by ABC in 1966? | The Dating Game | The Newlywed Game | Seed 101; the wrong title appears later in the original ABC passage, not in that distractor. |
| Overlong original-context span | Who worked on the Saturn series before Apollo began officially? | Wernher von Braun | Wernher von Braun and his team of rocket engineers | Seed 101; this longer phrase appears in the original Apollo passage and contains the correct answer, but EM is 0 (F1 50). |
| Distractor-only answer | Which church money collector said a soul would spring from purgatory through donations? | Johann Tetzel | Trotsky | Seed 101; “Trotsky” appears in the prepended unrelated passage and not the original Luther passage (F1 0). |
| Distractor-only answer | What did the Church do when Luther refused to retract his writings? | excommunication | implore[s] families and churches not to reject or condemn lesbian and gay members and friends | Seed 202; this text appears in an unrelated United Methodist Church distractor, not the original Luther passage (F1 0). |

The two distractor-only examples are diagnostic, not a prevalent failure mode in this run. Most exact wrong answers were found in the original answer-bearing passage. These counts describe the 149 LoRA EM-loss records and should not be presented as all prediction errors or as a causal classification. The two seeds are fixed; each question can contribute up to two records.
