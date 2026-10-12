# Five-way metric summary

Snapshot of the 477-case Highway summary with five variants and eleven metrics. Each row uses the same native-valid, non-null case intersection across all five variants. Cases receive equal weight; inference failures and missing scores are never filled with zero.

| Metric | N | R21-I2 caption | Checkpoint 11k | checkpoint-11k Init. + checkpoint-4k | Checkpoint 6k · 4a4b | Checkpoint 9k · 075a |
|---|---:|---:|---:|---:|---:|---:|
| Target RegionPQ ↑ | 463 | 0.9603 | 0.7392 | 0.8014 | 0.8290 | 0.8361 |
| Target Desc F1 ↑ | 452 | 0.8909 | 0.5060 | 0.6589 | 0.6876 | 0.6970 |
| Target Caption Embedding F1 ↑ | 463 | 0.8980 | 0.6058 | 0.7102 | 0.7352 | 0.7414 |
| Reference GroundedPQ ↑ | 450 | 0.9294 | 0.6653 | 0.7559 | 0.7724 | 0.7866 |
| Local Edit Instruction F1 ↑ | 420 | 0.8155 | 0.3086 | 0.5901 | 0.6116 | 0.6280 |
| Editing Edge Joint F1 ↑ | 450 | 0.8327 | 0.4068 | 0.6102 | 0.6318 | 0.6497 |
| Global HLD F1 ↑ | 452 | 0.8827 | 0.5978 | 0.7029 | 0.7049 | 0.7168 |
| Global Instruction F1 ↑ | 452 | 0.8191 | 0.3653 | 0.6255 | 0.6313 | 0.6492 |
| Global HLD 严重错误率 ↓ | 452 | 3.98% | 19.25% | 13.72% | 13.27% | 12.39% |
| Global Instruction 严重错误率 ↓ | 452 | 4.87% | 35.62% | 19.47% | 17.04% | 15.49% |
| Global–Local Instruction Consistency F1 ↑ | 431 | 0.7361 | 0.6117 | 0.7223 | 0.7326 | 0.7322 |

Quality scores use [0, 1]; critical-error rates are percentages. The CSV and JSON store all values in [0, 1] at their original precision.

Base semantic judge: gpt-6-luna / low. Consistency judge: gpt-6.1-sol / medium. Consistency: 444 eligible cases, 2,203 scored captions, 1 user-excluded caption, and 431 common scored cases. The exclusion does not change the shared cohort. Removal PQ and checkpoint-2k are absent from the active table.

6k and 9k use a different inference prompt from 11k Init. + 4k. 6k inference used H100; 9k used the devbox. Comparisons include these differences. Instruction consistency measures agreement within a caption, not image-grounded correctness.

[CSV](../data/five-way-metrics-20261009.csv) · [JSON](../data/five-way-metrics-20261009.json) · [Metric definitions](metrics.md) · [Live cases](https://grantzhang-alpha-document.highway.canva-internal.dev/#results)
