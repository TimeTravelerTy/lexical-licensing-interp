# Round 4, E13 follow-up. The natural " by" deficit of rare verbs

Descriptive, no model (`analyze_by_deficit.py`, from `pythia14b_scores.csv`). `passive_1` " by" margin good − bad; per band the mean of per-pair means; 95% CIs: pairs within band × contexts (2,000 draws, seed 17). Slope: per-pair margin on the good participle's form Zipf.

| contexts | pairs (Head / Tail / XTail) | Head | Tail | XTail | Head − XTail | Tail − XTail | (Head + Tail)/2 − XTail | slope on Zipf |
|---|---|---|---|---|---|---|---|---|
| curated | all (26 / 50 / 50) | +0.941 [+0.217, +1.663] | +0.963 [+0.595, +1.335] | +0.488 [+0.101, +0.816] | +0.453 [-0.334, +1.271] | +0.475 [-0.050, +1.023] | +0.464 [-0.074, +1.009] | +0.162 [-0.177, +0.497] |
| curated | primary (9 / 25 / 25) | +0.950 [-0.374, +2.166] | +0.830 [+0.353, +1.309] | +0.419 [-0.099, +0.929] | +0.531 [-0.897, +1.875] | +0.412 [-0.291, +1.123] | +0.471 [-0.395, +1.314] | +0.113 [-0.388, +0.674] |
| released | all (26 / 50 / 50) | +1.105 [+0.330, +1.897] | +1.063 [+0.679, +1.444] | +0.474 [+0.103, +0.813] | +0.631 [-0.207, +1.501] | +0.589 [+0.065, +1.116] | +0.610 [+0.072, +1.175] | +0.234 [-0.109, +0.620] |
| released | primary (9 / 25 / 25) | +0.813 [-0.647, +1.957] | +0.722 [+0.189, +1.218] | +0.118 [-0.394, +0.567] | +0.696 [-0.805, +1.960] | +0.604 [-0.117, +1.342] | +0.650 [-0.246, +1.485] | +0.180 [-0.341, +0.744] |

