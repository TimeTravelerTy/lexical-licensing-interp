# Task 2: splitting the *by* margin

Released `passive_1` contexts (300 per verb pair; mixed auxiliaries: was/were/is/are and negations). log P(by | prefix) is from the full-sentence scores (`good_by_lp`, `bad_by_lp`). z-scores are against the per-pair means of the reliable-positive pairs (all bands pooled; `_band` columns use the same band only). Reliable = 95% CI over contexts excludes 0.

## Band means of per-pair log P(by)

| Verb band | Class | n | log P(by \| good) | log P(by \| bad) | Margin |
|---|---|---:|---:|---:|---:|
| head | reliable_neg | 5 | -5.25 | -3.04 | -2.21 |
| head | reliable_pos | 21 | -2.48 | -4.37 | 1.89 |
| tail | ns_neg | 1 | -2.31 | -2.23 | -0.08 |
| tail | reliable_neg | 9 | -4.15 | -2.92 | -1.23 |
| tail | reliable_pos | 40 | -2.18 | -3.79 | 1.61 |
| xtail | ns_pos | 3 | -2.80 | -2.88 | 0.07 |
| xtail | reliable_neg | 12 | -3.85 | -2.49 | -1.36 |
| xtail | reliable_pos | 35 | -2.27 | -3.41 | 1.14 |

Reliable-positive reference (n = 96): good -2.28 (sd 0.82), bad -3.78 (sd 0.93), margin 1.50.

## Pairs with a negative *by* margin

`Good dev` and `Bad dev` are the deviations from the reliable mean; the margin shortfall is good dev - bad dev. `Driver` names the larger contributor.

| Band | Pair | Margin | lp by good | lp by bad | Good dev | Bad dev | z good | z bad | Driver |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| head | sum/compete | -3.24 | -5.97 | -2.73 | -3.70 | +1.05 | -4.5 | +1.1 | good side low |
| head | bet/appear | -2.68 | -6.77 | -4.09 | -4.49 | -0.31 | -5.5 | -0.3 | good side low |
| head | include/happen | -2.45 | -5.48 | -3.02 | -3.20 | +0.75 | -3.9 | +0.8 | good side low |
| head | exhibit/proceed | -1.49 | -3.31 | -1.82 | -1.03 | +1.96 | -1.3 | +2.1 | bad side high |
| head | load/respond | -1.17 | -4.71 | -3.54 | -2.43 | +0.24 | -3.0 | +0.3 | good side low |
| tail | blurt/stagnate | -2.16 | -4.85 | -2.69 | -2.57 | +1.08 | -3.1 | +1.2 | good side low |
| tail | stockpile/shudder | -1.95 | -3.47 | -1.52 | -1.19 | +2.25 | -1.5 | +2.4 | bad side high |
| tail | dice/cooperate | -1.77 | -3.71 | -1.94 | -1.43 | +1.84 | -1.7 | +2.0 | bad side high |
| tail | scrub/thrive | -1.50 | -3.93 | -2.42 | -1.65 | +1.35 | -2.0 | +1.5 | good side low |
| tail | heap/grin | -1.12 | -5.54 | -4.42 | -3.26 | -0.64 | -4.0 | -0.7 | good side low |
| tail | stow/abstain | -1.08 | -5.66 | -4.58 | -3.38 | -0.81 | -4.1 | -0.9 | good side low |
| tail | whitewash/wane | -0.82 | -3.23 | -2.41 | -0.95 | +1.37 | -1.2 | +1.5 | bad side high |
| tail | forest/function | -0.54 | -3.07 | -2.53 | -0.79 | +1.25 | -1.0 | +1.3 | bad side high |
| tail | refund/clash | -0.11 | -3.91 | -3.80 | -1.64 | -0.03 | -2.0 | -0.0 | good side low |
| tail | preclude/fluctuate | -0.08 | -2.31 | -2.23 | -0.03 | +1.54 | -0.0 | +1.7 | bad side high |
| xtail | shoo/quiver | -3.60 | -5.62 | -2.02 | -3.34 | +1.76 | -4.1 | +1.9 | good side low |
| xtail | daub/resound | -2.85 | -4.63 | -1.78 | -2.35 | +2.00 | -2.9 | +2.2 | good side low |
| xtail | doff/salivate | -2.09 | -3.98 | -1.89 | -1.70 | +1.89 | -2.1 | +2.0 | bad side high |
| xtail | wad/shimmer | -1.78 | -4.41 | -2.63 | -2.13 | +1.15 | -2.6 | +1.2 | good side low |
| xtail | debug/tingle | -1.30 | -2.88 | -1.59 | -0.60 | +2.19 | -0.7 | +2.4 | bad side high |
| xtail | lard/fizz | -1.06 | -4.09 | -3.02 | -1.81 | +0.76 | -2.2 | +0.8 | good side low |
| xtail | unlatch/burgeon | -0.97 | -3.49 | -2.52 | -1.21 | +1.26 | -1.5 | +1.4 | bad side high |
| xtail | disgorge/exult | -0.89 | -2.87 | -1.98 | -0.59 | +1.80 | -0.7 | +1.9 | bad side high |
| xtail | swaddle/decamp | -0.82 | -3.95 | -3.14 | -1.67 | +0.64 | -2.0 | +0.7 | good side low |
| xtail | shoplift/eventuate | -0.34 | -2.95 | -2.61 | -0.67 | +1.17 | -0.8 | +1.3 | bad side high |
| xtail | prise/jut | -0.32 | -3.91 | -3.59 | -1.63 | +0.19 | -2.0 | +0.2 | good side low |
| xtail | tithe/putter | -0.31 | -3.48 | -3.18 | -1.21 | +0.60 | -1.5 | +0.6 | good side low |

Driver counts: head: bad side high 1, good side low 4; tail: bad side high 5, good side low 5; xtail: bad side high 5, good side low 7

## Top next tokens after each prefix (negative pairs)

Mean probability from the stored top-10, averaged over the 300 released contexts.

| Band | Pair | After good participle | After bad participle |
|---|---|---|---|
| head | sum/compete | ` up` 0.85, ` to` 0.02, ` in` 0.01, ` and` 0.01, `,` 0.01 | ` for` 0.15, ` with` 0.14, ` in` 0.09, ` against` 0.09, ` by` 0.08 |
| head | bet/appear | `rot` 0.35, `tered` 0.10, ` on` 0.04, `t` 0.03, `wi` 0.03 | ` in` 0.19, ` to` 0.17, ` on` 0.07, `,` 0.05, `.` 0.04 |
| head | include/happen | ` in` 0.52, `.` 0.08, `,` 0.08, ` as` 0.03, ` on` 0.03 | ` upon` 0.22, ` to` 0.20, ` by` 0.06, ` in` 0.05, `.` 0.05 |
| head | exhibit/proceed | ` in` 0.22, ` at` 0.10, `,` 0.07, ` by` 0.05, ` as` 0.05 | ` by` 0.23, ` against` 0.15, ` to` 0.13, ` with` 0.12, ` in` 0.05 |
| head | load/respond | ` with` 0.30, ` into` 0.09, `,` 0.05, ` up` 0.05, `.` 0.04 | ` to` 0.71, ` by` 0.05, ` with` 0.04, `,` 0.01, ` well` 0.01 |
| tail | blurt/stagnate | ` out` 0.66, `,` 0.03, ` in` 0.02, ` from` 0.02, ` at` 0.02 | `,` 0.15, `.` 0.14, ` in` 0.12, ` by` 0.08, ` and` 0.07 |
| tail | stockpile/shudder | ` in` 0.21, ` with` 0.12, `,` 0.08, ` for` 0.05, `.` 0.05 | ` by` 0.25, ` at` 0.11, `,` 0.08, `.` 0.07, ` to` 0.07 |
| tail | dice/cooperate | ` up` 0.20, ` into` 0.12, ` and` 0.09, `,` 0.07, ` in` 0.04 | ` with` 0.29, ` by` 0.18, `,` 0.07, ` in` 0.07, `.` 0.06 |
| tail | scrub/thrive | ` from` 0.11, `,` 0.08, ` clean` 0.07, ` and` 0.07, ` out` 0.05 | ` on` 0.26, ` in` 0.15, ` by` 0.10, `,` 0.05, ` upon` 0.04 |
| tail | heap/grin | ` with` 0.28, ` on` 0.13, ` in` 0.08, ` up` 0.08, ` upon` 0.07 | ` at` 0.42, `.` 0.05, `,` 0.05, ` and` 0.04, ` down` 0.03 |
| tail | stow/abstain | ` away` 0.41, ` in` 0.18, `,` 0.05, `.` 0.03, ` on` 0.02 | ` from` 0.66, `,` 0.05, `.` 0.04, ` on` 0.02, ` of` 0.02 |
| tail | whitewash/wane | `,` 0.15, `.` 0.09, ` and` 0.09, ` in` 0.05, ` by` 0.05 | `,` 0.12, ` by` 0.11, `.` 0.09, ` in` 0.06, ` to` 0.06 |
| tail | forest/function | ` with` 0.15, `,` 0.13, ` in` 0.10, `.` 0.08, ` by` 0.08 | ` as` 0.18, ` by` 0.11, ` on` 0.09, ` in` 0.06, ` with` 0.04 |
| tail | refund/clash | `.` 0.10, `,` 0.10, ` their` 0.07, ` the` 0.07, ` for` 0.06 | ` with` 0.40, `,` 0.07, ` in` 0.05, `.` 0.05, ` over` 0.04 |
| tail | preclude/fluctuate | ` from` 0.66, ` by` 0.13, `,` 0.04, `.` 0.03, ` in` 0.01 | ` by` 0.12, ` in` 0.09, `,` 0.07, ` between` 0.06, `.` 0.05 |
| xtail | shoo/quiver | ` away` 0.51, ` off` 0.14, ` out` 0.09, ` from` 0.06, ` into` 0.03 | ` by` 0.17, ` with` 0.12, `,` 0.09, ` in` 0.08, ` at` 0.07 |
| xtail | daub/resound | ` with` 0.36, ` on` 0.13, ` in` 0.10, `,` 0.03, ` up` 0.03 | ` by` 0.19, ` with` 0.15, `,` 0.08, `.` 0.07, ` in` 0.07 |
| xtail | doff/salivate | ` to` 0.13, `,` 0.09, ` their` 0.08, ` and` 0.05, ` in` 0.05 | ` by` 0.17, ` at` 0.07, `,` 0.07, `.` 0.07, ` over` 0.07 |
| xtail | wad/shimmer | ` up` 0.51, ` into` 0.09, ` in` 0.08, ` with` 0.03, ` and` 0.03 | ` with` 0.12, ` in` 0.10, `,` 0.10, ` by` 0.09, ` and` 0.07 |
| xtail | debug/tingle | `,` 0.13, `.` 0.10, ` and` 0.10, ` by` 0.07, ` in` 0.05 | ` by` 0.24, ` with` 0.19, `,` 0.06, `.` 0.06, ` about` 0.03 |
| xtail | lard/fizz | ` with` 0.57, ` up` 0.21, ` by` 0.02, `,` 0.02, ` and` 0.01 | ` with` 0.15, ` up` 0.15, `,` 0.06, ` out` 0.06, ` by` 0.06 |
| xtail | unlatch/burgeon | `,` 0.15, `.` 0.13, ` and` 0.11, ` from` 0.09, `.\"` 0.04 | ` with` 0.12, ` by` 0.10, `,` 0.10, `.` 0.06, ` to` 0.06 |
| xtail | disgorge/exult | ` from` 0.11, ` into` 0.10, `,` 0.09, ` by` 0.06, `.` 0.06 | ` by` 0.15, `,` 0.10, ` in` 0.10, `.` 0.08, ` to` 0.06 |
| xtail | swaddle/decamp | ` in` 0.50, ` up` 0.10, ` and` 0.07, `,` 0.04, ` with` 0.03 | ` to` 0.15, ` from` 0.13, `,` 0.09, `.` 0.07, ` by` 0.05 |
| xtail | shoplift/eventuate | `,` 0.14, ` from` 0.13, `.` 0.10, ` by` 0.06, `.\"` 0.06 | ` by` 0.09, `,` 0.09, ` from` 0.09, ` in` 0.08, ` to` 0.08 |
| xtail | prise/jut | ` from` 0.20, ` out` 0.13, ` open` 0.10, ` off` 0.07, ` away` 0.07 | ` out` 0.13, ` to` 0.05, ` in` 0.05, ` up` 0.05, `,` 0.04 |
| xtail | tithe/putter | ` to` 0.30, `,` 0.12, `.` 0.06, ` by` 0.04, ` and` 0.03 | ` in` 0.15, ` up` 0.10, ` to` 0.07, ` into` 0.06, ` by` 0.05 |

Reliable-positive pairs, pooled: good → ` by` 0.16, `,` 0.09, `.` 0.06, ` in` 0.06, ` with` 0.06, ` to` 0.04; bad → ` to` 0.09, ` in` 0.09, ` with` 0.09, `,` 0.05, ` at` 0.04, `.` 0.04

