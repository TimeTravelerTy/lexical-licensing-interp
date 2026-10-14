# Round 5, A5: grammaticality probe vs LP

Spec: `plan.md`, A5. Run: `run_probe.py`; analysis: `analyze_probe.py`. Probe: L2 logistic regression on the last token's residual state (Wang et al. 2026 adaptation); a pair is correct if the good sentence has the higher probe logit (LP: higher log-prob).

Selected layer 13 (dev AUC 0.930); C per split {'0': 0.03125, '1': 0.03125, '2': 0.03125}. Dev AUC by layer (split 0): 1: 0.639, 2: 0.686, 3: 0.751, 4: 0.807, 5: 0.841, 6: 0.856, 7: 0.885, 8: 0.892, 9: 0.900, 10: 0.905, 11: 0.916, 12: 0.930, 13: 0.930, 14: 0.927, 15: 0.925, 16: 0.922, 17: 0.920, 18: 0.918, 19: 0.916, 20: 0.919, 21: 0.917, 22: 0.916, 23: 0.916, 24: 0.915.
Baselines (dev AUC): length/punctuation 0.569; LM log-prob 0.683; per token 0.844.

## Released `passive_1`

| Method | Head | Tail | XTail | D = Head − XTail | Head − Tail |
|---|---|---|---|---|---|
| probe (split 0) | +88.2 [+86.2, +90.1] | +82.0 [+79.6, +84.4] | +71.9 [+69.1, +74.7] | +16.3 [+12.9, +19.8] | +6.2 [+3.0, +9.2] |
| probe (split 1) | +88.6 [+86.7, +90.6] | +82.6 [+80.2, +84.9] | +72.7 [+69.9, +75.3] | +15.9 [+12.7, +19.4] | +6.0 [+3.1, +9.1] |
| probe (split 2) | +88.6 [+86.7, +90.5] | +82.1 [+79.6, +84.5] | +71.8 [+69.0, +74.5] | +16.8 [+13.5, +20.2] | +6.5 [+3.5, +9.5] |
| LP | +79.1 [+76.7, +81.7] | +79.2 [+76.8, +81.6] | +69.4 [+66.7, +72.1] | +9.7 [+5.9, +13.6] | -0.1 [-3.6, +3.6] |

Paired: D_probe − 0.5 D_LP +11.5 [+7.9, +14.9]; D_LP − D_probe -6.6 [-11.0, -2.0]. Verb-cluster sensitivity: D_probe +16.3 [+11.8, +21.1], D_LP +9.7 [+4.7, +14.7], D_LP − D_probe -6.6 [-12.5, -0.8].
Mean probe-logit margin by band: head +3.00, tail +1.61, xtail +0.88.

## Released `passive_2`

| Method | Head | Tail | XTail | D = Head − XTail | Head − Tail |
|---|---|---|---|---|---|
| probe (split 0) | +91.2 [+89.5, +92.9] | +83.9 [+81.6, +86.1] | +71.4 [+68.5, +74.3] | +19.8 [+16.4, +23.2] | +7.3 [+4.4, +10.0] |
| probe (split 1) | +90.8 [+88.9, +92.5] | +83.9 [+81.6, +86.0] | +73.9 [+71.1, +76.4] | +16.9 [+13.7, +20.1] | +6.9 [+3.9, +9.8] |
| probe (split 2) | +90.6 [+88.8, +92.3] | +83.4 [+81.1, +85.6] | +72.8 [+70.0, +75.6] | +17.8 [+14.4, +21.1] | +7.2 [+4.2, +10.1] |
| LP | +81.5 [+79.0, +84.0] | +79.3 [+76.8, +81.7] | +73.2 [+70.4, +75.8] | +8.3 [+4.8, +12.0] | +2.2 [-1.4, +5.8] |

Paired: D_probe − 0.5 D_LP +15.6 [+12.1, +19.3]; D_LP − D_probe -11.5 [-16.2, -6.8]. Verb-cluster sensitivity: D_probe +19.8 [+15.4, +24.5], D_LP +8.3 [+3.6, +13.3], D_LP − D_probe -11.5 [-17.9, -5.0].
Mean probe-logit margin by band: head +3.87, tail +2.40, xtail +1.17.

## All 67 paradigms (descriptive)

Mean Head accuracy: probe 77.6%, LP 79.2%. Mean Head − XTail drop: probe +2.6 pp, LP +1.7 pp; correlation of the drops across paradigms +0.64.

Curated band-cross `passive_1` (descriptive): head: probe 89.2%, LP 81.4%; tail: probe 84.9%, LP 77.7%; xtail: probe 79.1%, LP 72.9%.

## Declared decisions (released passive_1)

- Probe informative: **True**.
- Detectable decline: **True**; declines like LP: **True**; smaller decline than LP: **False**; practically no decline: **False**.

