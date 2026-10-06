# Task 1: natural passive baseline

Pythia-1.4B next-token log-probs (bf16, TSUBAME). Passives: curated `was` prefix + participle, other verbs' contexts only (126 verb pairs x 125 contexts = 15,750 items). Actives: "The AGENT VERB-ed" with the agent of the same curated `passive_1` context. Object start = the, a, an, his, her, their, its, this, that, some, him, them, it. 95% CIs: two-way cluster bootstrap (1000 draws).

## Passive: good (transitive) vs bad (intransitive) verb

| Next token | Verb band | Good | Bad | Good - bad |
|---|---|---:|---:|---:|
| log P(" the") | head | -5.99 | -6.12 | 0.12 [-0.55, 0.83] |
| log P(" the") | tail | -6.15 | -6.56 | 0.41 [0.06, 0.77] |
| log P(" the") | xtail | -6.08 | -6.49 | 0.41 [0.11, 0.71] |
| log P(" the") | XTail - Head |  |  | 0.29 [-0.49, 1.04] |
| log P(" a") | head | -5.92 | -6.00 | 0.08 [-0.65, 0.80] |
| log P(" a") | tail | -5.97 | -5.96 | -0.01 [-0.38, 0.37] |
| log P(" a") | xtail | -5.82 | -5.90 | 0.08 [-0.19, 0.35] |
| log P(" a") | XTail - Head |  |  | 0.00 [-0.76, 0.78] |
| log P(" by") | head | -2.57 | -3.51 | 0.94 [0.21, 1.63] |
| log P(" by") | tail | -2.21 | -3.17 | 0.96 [0.57, 1.34] |
| log P(" by") | xtail | -2.38 | -2.87 | 0.49 [0.11, 0.82] |
| log P(" by") | XTail - Head |  |  | -0.45 [-1.28, 0.40] |
| log P(".") | head | -4.92 | -4.66 | -0.26 [-0.93, 0.39] |
| log P(".") | tail | -3.57 | -4.21 | 0.64 [0.24, 1.03] |
| log P(".") | xtail | -3.63 | -3.75 | 0.12 [-0.23, 0.46] |
| log P(".") | XTail - Head |  |  | 0.38 [-0.36, 1.15] |
| log P(det: the/a/an) | head | -5.12 | -5.22 | 0.10 [-0.58, 0.79] |
| log P(det: the/a/an) | tail | -5.23 | -5.35 | 0.12 [-0.23, 0.47] |
| log P(det: the/a/an) | xtail | -5.13 | -5.30 | 0.17 [-0.08, 0.43] |
| log P(det: the/a/an) | XTail - Head |  |  | 0.07 [-0.68, 0.81] |
| log P(object start) | head | -4.69 | -4.56 | -0.14 [-0.75, 0.53] |
| log P(object start) | tail | -4.83 | -4.91 | 0.08 [-0.26, 0.41] |
| log P(object start) | xtail | -4.67 | -4.81 | 0.13 [-0.12, 0.39] |
| log P(object start) | XTail - Head |  |  | 0.27 [-0.44, 0.93] |

## Active: good (transitive) vs bad (intransitive) verb

| Next token | Verb band | Good | Bad | Good - bad |
|---|---|---:|---:|---:|
| log P(" the") | head | -1.97 | -6.01 | 4.04 [3.56, 4.50] |
| log P(" the") | tail | -1.81 | -5.63 | 3.82 [3.39, 4.20] |
| log P(" the") | xtail | -1.78 | -4.93 | 3.15 [2.80, 3.54] |
| log P(" the") | XTail - Head |  |  | -0.88 [-1.45, -0.28] |
| log P(" a") | head | -3.19 | -5.73 | 2.53 [1.66, 3.35] |
| log P(" a") | tail | -3.53 | -5.18 | 1.65 [1.17, 2.10] |
| log P(" a") | xtail | -3.62 | -4.65 | 1.03 [0.70, 1.40] |
| log P(" a") | XTail - Head |  |  | -1.50 [-2.40, -0.52] |
| log P(".") | head | -6.73 | -4.29 | -2.44 [-3.18, -1.65] |
| log P(".") | tail | -5.71 | -3.62 | -2.10 [-2.66, -1.51] |
| log P(".") | xtail | -5.40 | -3.41 | -2.00 [-2.58, -1.39] |
| log P(".") | XTail - Head |  |  | 0.45 [-0.55, 1.37] |
| log P(det: the/a/an) | head | -1.41 | -4.93 | 3.51 [3.00, 4.04] |
| log P(det: the/a/an) | tail | -1.49 | -4.46 | 2.96 [2.59, 3.32] |
| log P(det: the/a/an) | xtail | -1.51 | -3.86 | 2.35 [2.08, 2.62] |
| log P(det: the/a/an) | XTail - Head |  |  | -1.16 [-1.77, -0.57] |
| log P(object start) | head | -0.84 | -3.90 | 3.05 [2.44, 3.61] |
| log P(object start) | tail | -0.90 | -3.80 | 2.90 [2.55, 3.21] |
| log P(object start) | xtail | -0.82 | -3.11 | 2.28 [1.99, 2.56] |
| log P(object start) | XTail - Head |  |  | -0.77 [-1.35, -0.09] |

## Most probable next tokens

Mean probability over items from each prompt's stored top-10 (a token absent from a prompt's top-10 counts as 0, so values are lower bounds).

| Set | Verb band | Verb | Top tokens (mean p) |
|---|---|---|---|
| passive | head | good | ` by` 0.166, ` to` 0.093, ` in` 0.084, ` for` 0.062, ` with` 0.056, ` on` 0.041, ` at` 0.036, ` as` 0.034 |
| passive | head | bad | ` to` 0.260, ` in` 0.091, ` by` 0.056, ` for` 0.051, ` at` 0.051, ` from` 0.032, ` with` 0.032, `,` 0.030 |
| passive | tail | good | ` by` 0.179, ` in` 0.079, ` with` 0.077, `,` 0.073, ` and` 0.067, `.` 0.048, ` from` 0.044, ` to` 0.038 |
| passive | tail | bad | ` with` 0.121, ` in` 0.095, ` to` 0.069, ` by` 0.067, `,` 0.048, ` at` 0.040, ` and` 0.040, ` from` 0.037 |
| passive | xtail | good | ` by` 0.157, ` with` 0.102, `,` 0.072, ` and` 0.071, ` in` 0.071, `.` 0.045, ` from` 0.039, ` to` 0.035 |
| passive | xtail | bad | ` with` 0.114, ` by` 0.085, ` in` 0.072, `,` 0.062, ` and` 0.057, ` on` 0.041, `.` 0.038, ` over` 0.037 |
| active | head | good | ` the` 0.187, ` a` 0.107, ` his` 0.056, ` up` 0.032, ` her` 0.027, ` me` 0.025, ` him` 0.020, ` himself` 0.020 |
| active | head | bad | ` to` 0.225, ` in` 0.119, `,` 0.073, ` that` 0.063, `.` 0.048, ` from` 0.035, ` for` 0.034, ` and` 0.031 |
| active | tail | good | ` the` 0.235, ` his` 0.060, ` a` 0.048, ` her` 0.033, ` himself` 0.023, ` him` 0.018, ` me` 0.017, ` by` 0.016 |
| active | tail | bad | ` with` 0.144, `,` 0.079, `.` 0.069, ` in` 0.063, ` to` 0.059, ` and` 0.043, ` from` 0.036, ` on` 0.030 |
| active | xtail | good | ` the` 0.214, ` his` 0.080, ` a` 0.049, ` her` 0.048, ` him` 0.035, ` himself` 0.033, ` me` 0.026, `,` 0.016 |
| active | xtail | bad | ` in` 0.087, `,` 0.077, `.` 0.073, ` with` 0.073, ` and` 0.046, ` at` 0.038, ` on` 0.037, ` over` 0.024 |

## Sample prompts (top-10)

- `The dish was exhibited` → ` at` 0.457, ` in` 0.277, ` as` 0.048, ` on` 0.038, ` by` 0.027, ` for` 0.016, ` and` 0.013, ` with` 0.013, ` during` 0.012, ` to` 0.009
- `The dish was proceeded` → ` with` 0.495, ` to` 0.125, ` by` 0.098, ` in` 0.056, ` as` 0.043, ` on` 0.034, `,` 0.018, `\n` 0.012, `.` 0.009, ` and` 0.008
- `The chef exhibited` → ` a` 0.229, ` his` 0.108, ` the` 0.070, ` an` 0.058, ` her` 0.029, ` great` 0.021, ` at` 0.021, ` some` 0.019, ` in` 0.014, ` exceptional` 0.011
- `The chef proceeded` → ` to` 0.862, ` with` 0.033, `,` 0.020, ` in` 0.006, `:` 0.005, ` into` 0.005, ` as` 0.005, `\n` 0.004, ` on` 0.004, `.` 0.003

- `The teacher was insured` → ` by` 0.198, ` and` 0.106, ` for` 0.106, `,` 0.106, `.` 0.083, ` under` 0.073, ` with` 0.050, ` through` 0.030, ` against` 0.029, `.\"` 0.020
- `The teacher was vanished` → `.` 0.217, `,` 0.169, ` and` 0.071, ` from` 0.058, `.\"` 0.055, ` into` 0.048, ` in` 0.033, ` for` 0.022, ` when` 0.018, `;` 0.015
- `The student insured` → ` under` 0.150, `s` 0.103, `,` 0.052, ` is` 0.049, ` by` 0.043, ` shall` 0.043, ` may` 0.031, `\n` 0.026, ` was` 0.024, ` in` 0.020
- `The student vanished` → ` from` 0.192, `.` 0.071, ` into` 0.066, ` after` 0.066, `,` 0.055, ` in` 0.046, ` on` 0.038, `.\"` 0.035, ` and` 0.031, ` without` 0.028

- `The alliance was scrubbed` → ` from` 0.373, ` out` 0.065, ` in` 0.065, ` after` 0.057, ` by` 0.057, ` for` 0.027, `,` 0.025, ` on` 0.024, ` when` 0.021, ` clean` 0.017
- `The alliance was thrived` → ` on` 0.253, ` by` 0.197, ` in` 0.077, ` because` 0.047, ` upon` 0.041, ` and` 0.034, ` through` 0.025, `,` 0.024, ` for` 0.020, ` with` 0.020
- `The war scrubbed` → ` the` 0.280, ` away` 0.075, ` out` 0.067, ` all` 0.030, ` up` 0.028, ` it` 0.023, ` his` 0.020, ` a` 0.018, ` my` 0.017, ` them` 0.015
- `The war thrived` → ` on` 0.387, ` in` 0.118, `,` 0.072, `.` 0.063, ` for` 0.049, ` and` 0.032, ` as` 0.022, ` because` 0.018, ` during` 0.015, ` with` 0.011

- `The paper was recast` → ` as` 0.216, ` from` 0.148, ` and` 0.131, ` by` 0.075, ` in` 0.070, ` to` 0.066, ` for` 0.051, `,` 0.031, ` with` 0.027, ` at` 0.026
- `The paper was reappeared` → ` in` 0.466, ` on` 0.086, ` as` 0.034, ` by` 0.032, `,` 0.030, ` and` 0.028, ` after` 0.026, ` with` 0.023, ` at` 0.020, ` a` 0.018
- `The child recast` → ` the` 0.273, ` his` 0.050, ` herself` 0.035, ` himself` 0.033, ` itself` 0.033, ` as` 0.031, ` her` 0.031, ` a` 0.021, `\n` 0.021, `,` 0.020
- `The child reappeared` → `,` 0.200, ` in` 0.107, ` and` 0.101, `.` 0.089, ` at` 0.057, ` with` 0.054, ` from` 0.042, ` a` 0.033, ` on` 0.029, ` behind` 0.016

- `The garden was debugged` → ` and` 0.164, ` by` 0.106, `,` 0.088, ` with` 0.068, `.` 0.064, ` in` 0.057, ` to` 0.050, ` for` 0.034, ` using` 0.030, ` on` 0.021
- `The garden was tingled` → ` with` 0.718, ` by` 0.097, ` and` 0.036, `,` 0.018, ` in` 0.014, `.` 0.014, ` all` 0.009, ` from` 0.007, ` to` 0.006, ` pink` 0.005
- `The gardener debugged` → ` the` 0.383, ` it` 0.075, ` his` 0.071, ` and` 0.036, ` a` 0.030, `,` 0.026, ` me` 0.026, ` her` 0.023, ` himself` 0.022, `.` 0.020
- `The gardener tingled` → ` with` 0.229, `.` 0.178, ` at` 0.108, ` as` 0.084, ` to` 0.065, `,` 0.065, ` when` 0.033, ` and` 0.029, ` in` 0.020, ` all` 0.011

- `The student was unlatched` → ` from` 0.253, ` and` 0.209, `,` 0.127, ` the` 0.072, `.` 0.064, ` by` 0.041, ` in` 0.016, ` his` 0.011, ` at` 0.011, ` with` 0.011
- `The student was burgeoned` → ` with` 0.247, ` by` 0.103, ` and` 0.075, `,` 0.071, ` in` 0.049, `.` 0.049, ` to` 0.043, ` into` 0.040, ` on` 0.023, ` from` 0.022
- `The teacher unlatched` → ` the` 0.684, ` his` 0.093, ` her` 0.087, ` a` 0.041, ` my` 0.016, ` and` 0.013, ` one` 0.008, ` it` 0.006, ` himself` 0.004, ` from` 0.004
- `The teacher burgeoned` → ` into` 0.253, ` in` 0.112, `.` 0.064, `,` 0.056, ` with` 0.056, ` and` 0.036, ` a` 0.032, ` the` 0.032, ` to` 0.030, ` from` 0.028

