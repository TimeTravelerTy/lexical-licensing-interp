# Round 6, A1: Pile exposure and the frame contrast

Spec: `plan.md`, A1. Counts: `fetch_pile_counts_r6.py` (infini-gram `v4_piletrain_llama`); analysis: `analyze_exposure.py`. Outcomes: Pythia-1.4B Δ_pass(V) (24 freq frames) and L. Reproduction of round-5 scores by the D11 run (shared verbs, freq): max |Δ lp| 0.0e+00 (20784 rows). CV: frozen fold map (pairs shuffled within band, seed 17); 95% CIs from pairs within band × contexts (2,000 draws). Decision: b better = 95% CI of Δ = MSE_a − MSE_b above 0; a better = below 0; equivalent = 90% CI of Δ / MSE_a within ±2%; else unresolved. A direction prediction holds only with 'b better' and the predicted coefficient sign (95% CI excluding 0). Primary family: I1, I3, T2, T3 (Bonferroni 98.75% CIs beside).

## Counts

c_total = ▁lower + ▁Cap + after-newline lower + after-newline Cap (boundary-safe). Medians:

| class | band | n | c_total | share ▁lower | share ▁Cap | share after newline | upper / c_total | c_be | c_perf | c_comp (anchored) | c_comp (broad) | share c_be = 0 | share c_comp = 0 |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---|---|
| I | Head | 49 | 1,985,412 | 0.985 | 0.003 | 0.010 | 1.01 | 16,800 | 173,948 | 26,781 | 243,764 | 0.00 | 0.00 |
| I | Tail | 133 | 168,368 | 0.983 | 0.003 | 0.012 | 1.01 | 790 | 11,829 | 1,737 | 16,211 | 0.00 | 0.01 |
| I | XTail | 107 | 18,593 | 0.982 | 0.003 | 0.013 | 1.01 | 169 | 926 | 208 | 2,210 | 0.00 | 0.00 |
| T | Head | 63 | 1,876,246 | 0.971 | 0.016 | 0.013 | 1.07 | 334,947 | 71,418 | 5,254 | 53,278 | 0.00 | 0.00 |
| T | Tail | 150 | 141,747 | 0.972 | 0.010 | 0.015 | 1.03 | 27,928 | 5,666 | 338 | 3,260 | 0.00 | 0.00 |
| T | XTail | 106 | 19,549 | 0.969 | 0.012 | 0.017 | 1.03 | 3,531 | 732 | 60 | 644 | 0.00 | 0.01 |

Forms whose upper-bound count exceeds c_total by > 50% (likely word-internal matches of the unmarked sequence): 33 (ached, awed, buffed, cached, doped, hocked, husked, jigged, lapsed, leased, leashed, lied, loaded, lopped, meditated, numbered, ogled, peed, ported, prised, puttered, racked, raged, ranted, rated, raved, rued, sided, skinned, tagged, tamed, uttered, vied).

## Model comparison: D11 (281 pairs)

- r(Δ_pass(T), ·): Z -0.10, C -0.16, B -0.05, P -0.31, K -0.25, K_broad -0.24, S +0.38, S_fisher +0.38, S_washas +0.48, N +0.10; r(C, K) +0.76; cond(C + K) 44
- r(Δ_pass(I), ·): Z -0.40, C -0.34, B -0.11, P -0.55, K -0.18, K_broad -0.22, S +0.53, S_fisher +0.35, S_washas +0.68, N +0.20; r(C, K) +0.83; cond(C + K) 46

| ID | family | outcome | a → b | MSE a | MSE b | Δ = a − b (95% CI) | Δ / MSE a (90% CI) | decision | added coefficient (95% CI) | in-sample R² a / b | cond(b) | question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I1 | primary | Δ_pass(I) | null → C | 3.061 | 2.722 | +0.339 [+0.102, +0.618]; Bonferroni 98.75% +0.339 [+0.058, +0.696] | +0.111 [+0.043, +0.179] | **b better** | C -0.728 [-0.975, -0.471] | 0.000 / 0.115 | 34 | does total exposure predict intransitive rejection? |
| I2 | secondary | Δ_pass(I) | C → K | 2.722 | 2.974 | -0.252 [-0.431, -0.119] | -0.093 [-0.150, -0.048] | **a better** | K -0.349 [-0.600, -0.101] | 0.115 / 0.034 | 12 | competing-frame count vs total count |
| I3 | primary | Δ_pass(I) | C → C + K | 2.722 | 2.682 | +0.040 [-0.119, +0.203]; Bonferroni 98.75% +0.040 [-0.184, +0.258] | +0.015 [-0.031, +0.057] | **unresolved** | K +0.588 [+0.240, +1.110] | 0.115 / 0.145 | 46 | does the competing frame add to total exposure? |
| I4 | secondary | Δ_pass(I) | C → C + B | 2.722 | 2.636 | +0.086 [-0.032, +0.245] | +0.032 [-0.006, +0.073] | **unresolved** | B +0.451 [+0.173, +0.739] | 0.115 / 0.148 | 41 | does the passive slot add? |
| I5 | secondary | Δ_pass(I) | Z → C | 2.587 | 2.722 | -0.135 [-0.246, -0.037] | -0.052 [-0.088, -0.019] | **a better** | C -0.728 [-0.975, -0.471] | 0.161 / 0.115 | 34 | Pile counts vs wordfreq Zipf |
| T1 | secondary | Δ_pass(T) | null → C | 1.723 | 1.690 | +0.033 [-0.026, +0.122] | +0.019 [-0.012, +0.053] | **unresolved** | C -0.266 [-0.490, -0.062] | 0.000 / 0.026 | 34 | does total exposure predict transitive acceptance? |
| T2 | primary | Δ_pass(T) | C → C + B | 1.690 | 1.596 | +0.094 [-0.005, +0.239]; Bonferroni 98.75% +0.094 [-0.019, +0.295] | +0.056 [+0.002, +0.110] | **unresolved** | B +0.938 [+0.409, +1.475] | 0.026 / 0.086 | 52 | does the passive slot add? |
| T3 | primary | Δ_pass(T) | C → C + S | 1.690 | 1.413 | +0.276 [+0.101, +0.497]; Bonferroni 98.75% +0.276 [+0.067, +0.571] | +0.164 [+0.076, +0.238] | **b better** | S +0.418 [+0.279, +0.573] | 0.026 / 0.192 | 35 | does the BE/HAVE preference add? |
| T4 | secondary | Δ_pass(T) | Z → C | 1.717 | 1.690 | +0.027 [-0.003, +0.072] | +0.016 [+0.001, +0.035] | **unresolved** | C -0.266 [-0.490, -0.062] | 0.011 / 0.026 | 34 | Pile vs Zipf |
| L1 | secondary | L | Z_T + Z_I → C_T + C_I | 4.051 | 4.094 | -0.043 [-0.182, +0.133] | -0.011 [-0.038, +0.024] | **unresolved** | C_T +0.661 [+0.079, +1.251]; C_I -0.045 [-0.615, +0.524] | 0.071 / 0.056 | 49 | Pile vs Zipf at pair level |
| L2a | secondary | L | C_T + C_I → C_T + C_I + S_T | 4.094 | 3.947 | +0.147 [-0.066, +0.416] | +0.036 [-0.009, +0.082] | **unresolved** | S_T +0.324 [+0.127, +0.540] | 0.056 / 0.095 | 50 | does the T verb's preference add? |
| L2b | secondary | L | C_T + C_I → C_T + C_I + K_I | 4.094 | 4.032 | +0.062 [-0.085, +0.217] | +0.015 [-0.015, +0.042] | **unresolved** | K_I -0.615 [-1.120, -0.196] | 0.056 / 0.079 | 62 | does the I verb's competing frame add? |
| L2 | secondary | L | C_T + C_I → C_T + C_I + S_T + K_I | 4.094 | 3.925 | +0.169 [-0.100, +0.415] | +0.041 [-0.015, +0.083] | **unresolved** | S_T +0.304 [+0.103, +0.522]; K_I -0.547 [-1.099, -0.124] | 0.056 / 0.113 | 62 | both |

Sensitivity (D11): token count added (log participle tokens)

| ID | family | outcome | a → b | MSE a | MSE b | Δ = a − b (95% CI) | Δ / MSE a (90% CI) | decision | added coefficient (95% CI) | in-sample R² a / b | cond(b) | question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I1 | primary | Δ_pass(I) | null → C | 2.961 | 2.709 | +0.252 [+0.057, +0.478]; Bonferroni 98.75% +0.252 [+0.022, +0.562] | +0.085 [+0.025, +0.147] | **b better** | C -1.062 [-1.430, -0.694] | 0.039 / 0.129 | 70 | does total exposure predict intransitive rejection? |
| I3 | primary | Δ_pass(I) | C → C + K | 2.709 | 2.662 | +0.047 [-0.093, +0.194]; Bonferroni 98.75% +0.047 [-0.151, +0.246] | +0.017 [-0.023, +0.055] | **unresolved** | K +0.582 [+0.250, +1.092] | 0.129 / 0.159 | 83 | does the competing frame add to total exposure? |
| I5 | secondary | Δ_pass(I) | Z → C | 2.536 | 2.709 | -0.173 [-0.313, -0.043] | -0.068 [-0.115, -0.023] | **a better** | C -1.062 [-1.430, -0.694] | 0.184 / 0.129 | 70 | Pile counts vs wordfreq Zipf |
| T2 | primary | Δ_pass(T) | C → C + B | 1.694 | 1.599 | +0.096 [-0.004, +0.246]; Bonferroni 98.75% +0.096 [-0.018, +0.301] | +0.056 [+0.002, +0.112] | **unresolved** | B +0.939 [+0.417, +1.477] | 0.028 / 0.089 | 92 | does the passive slot add? |
| T3 | primary | Δ_pass(T) | C → C + S | 1.694 | 1.417 | +0.278 [+0.100, +0.501]; Bonferroni 98.75% +0.278 [+0.070, +0.576] | +0.164 [+0.076, +0.238] | **b better** | S +0.418 [+0.281, +0.575] | 0.028 / 0.192 | 70 | does the BE/HAVE preference add? |
| L1 | secondary | L | Z_T + Z_I → C_T + C_I | 4.082 | 4.151 | -0.069 [-0.256, +0.121] | -0.017 [-0.052, +0.019] | **unresolved** | C_T +0.845 [+0.139, +1.561]; C_I +0.066 [-0.593, +0.750] | 0.084 / 0.061 | 116 | Pile vs Zipf at pair level |
| L2a | secondary | L | C_T + C_I → C_T + C_I + S_T | 4.151 | 3.972 | +0.179 [-0.052, +0.478] | +0.043 [-0.007, +0.093] | **unresolved** | S_T +0.345 [+0.142, +0.560] | 0.061 / 0.104 | 118 | does the T verb's preference add? |
| L2b | secondary | L | C_T + C_I → C_T + C_I + K_I | 4.151 | 4.083 | +0.068 [-0.083, +0.223] | +0.016 [-0.014, +0.043] | **unresolved** | K_I -0.601 [-1.106, -0.174] | 0.061 / 0.083 | 128 | does the I verb's competing frame add? |
| L2 | secondary | L | C_T + C_I → C_T + C_I + S_T + K_I | 4.151 | 3.947 | +0.204 [-0.085, +0.473] | +0.049 [-0.010, +0.094] | **unresolved** | S_T +0.323 [+0.115, +0.542]; K_I -0.521 [-1.068, -0.107] | 0.061 / 0.120 | 129 | both |

Sensitivity (D11): c_total marked only

| ID | family | outcome | a → b | MSE a | MSE b | Δ = a − b (95% CI) | Δ / MSE a (90% CI) | decision | added coefficient (95% CI) | in-sample R² a / b | cond(b) | question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I1 | primary | Δ_pass(I) | null → C | 3.061 | 2.722 | +0.339 [+0.102, +0.619]; Bonferroni 98.75% +0.339 [+0.057, +0.698] | +0.111 [+0.043, +0.180] | **b better** | C -0.728 [-0.975, -0.470] | 0.000 / 0.115 | 34 | does total exposure predict intransitive rejection? |
| I3 | primary | Δ_pass(I) | C → C + K | 2.722 | 2.681 | +0.040 [-0.118, +0.203]; Bonferroni 98.75% +0.040 [-0.182, +0.258] | +0.015 [-0.031, +0.057] | **unresolved** | K +0.589 [+0.241, +1.109] | 0.115 / 0.145 | 45 | does the competing frame add to total exposure? |
| I5 | secondary | Δ_pass(I) | Z → C | 2.587 | 2.722 | -0.134 [-0.245, -0.037] | -0.052 [-0.088, -0.019] | **a better** | C -0.728 [-0.975, -0.470] | 0.161 / 0.115 | 34 | Pile counts vs wordfreq Zipf |
| T2 | primary | Δ_pass(T) | C → C + B | 1.689 | 1.594 | +0.096 [-0.005, +0.243]; Bonferroni 98.75% +0.096 [-0.018, +0.298] | +0.057 [+0.002, +0.112] | **unresolved** | B +0.949 [+0.420, +1.485] | 0.026 / 0.088 | 51 | does the passive slot add? |
| T3 | primary | Δ_pass(T) | C → C + S | 1.689 | 1.412 | +0.277 [+0.101, +0.497]; Bonferroni 98.75% +0.277 [+0.067, +0.572] | +0.164 [+0.076, +0.238] | **b better** | S +0.418 [+0.280, +0.574] | 0.026 / 0.193 | 35 | does the BE/HAVE preference add? |
| L1 | secondary | L | Z_T + Z_I → C_T + C_I | 4.051 | 4.096 | -0.045 [-0.183, +0.128] | -0.011 [-0.039, +0.023] | **unresolved** | C_T +0.653 [+0.068, +1.241]; C_I -0.040 [-0.605, +0.533] | 0.071 / 0.056 | 49 | Pile vs Zipf at pair level |
| L2a | secondary | L | C_T + C_I → C_T + C_I + S_T | 4.096 | 3.949 | +0.147 [-0.065, +0.416] | +0.036 [-0.009, +0.082] | **unresolved** | S_T +0.325 [+0.128, +0.541] | 0.056 / 0.094 | 49 | does the T verb's preference add? |
| L2b | secondary | L | C_T + C_I → C_T + C_I + K_I | 4.096 | 4.035 | +0.061 [-0.086, +0.216] | +0.015 [-0.015, +0.042] | **unresolved** | K_I -0.614 [-1.118, -0.194] | 0.056 / 0.079 | 62 | does the I verb's competing frame add? |
| L2 | secondary | L | C_T + C_I → C_T + C_I + S_T + K_I | 4.096 | 3.928 | +0.168 [-0.100, +0.414] | +0.041 [-0.016, +0.082] | **unresolved** | S_T +0.305 [+0.103, +0.523]; K_I -0.545 [-1.099, -0.121] | 0.056 / 0.112 | 62 | both |

Sensitivity (D11): c_total upper bound

| ID | family | outcome | a → b | MSE a | MSE b | Δ = a − b (95% CI) | Δ / MSE a (90% CI) | decision | added coefficient (95% CI) | in-sample R² a / b | cond(b) | question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I1 | primary | Δ_pass(I) | null → C | 3.061 | 2.756 | +0.305 [+0.087, +0.563]; Bonferroni 98.75% +0.305 [+0.040, +0.650] | +0.100 [+0.036, +0.168] | **b better** | C -0.670 [-0.914, -0.423] | 0.000 / 0.103 | 34 | does total exposure predict intransitive rejection? |
| I3 | primary | Δ_pass(I) | C → C + K | 2.756 | 2.734 | +0.022 [-0.066, +0.124]; Bonferroni 98.75% +0.022 [-0.108, +0.164] | +0.008 [-0.018, +0.033] | **unresolved** | K +0.410 [+0.074, +0.864] | 0.103 / 0.119 | 44 | does the competing frame add to total exposure? |
| I5 | secondary | Δ_pass(I) | Z → C | 2.587 | 2.756 | -0.168 [-0.284, -0.067] | -0.065 [-0.106, -0.030] | **a better** | C -0.670 [-0.914, -0.423] | 0.161 / 0.103 | 34 | Pile counts vs wordfreq Zipf |
| T2 | primary | Δ_pass(T) | C → C + B | 1.692 | 1.645 | +0.048 [-0.036, +0.155]; Bonferroni 98.75% +0.048 [-0.053, +0.200] | +0.028 [-0.014, +0.068] | **unresolved** | B +0.601 [+0.151, +1.101] | 0.023 / 0.057 | 49 | does the passive slot add? |
| T3 | primary | Δ_pass(T) | C → C + S | 1.692 | 1.418 | +0.275 [+0.100, +0.497]; Bonferroni 98.75% +0.275 [+0.065, +0.575] | +0.162 [+0.075, +0.236] | **b better** | S +0.417 [+0.277, +0.576] | 0.023 / 0.189 | 35 | does the BE/HAVE preference add? |
| L1 | secondary | L | Z_T + Z_I → C_T + C_I | 4.051 | 4.092 | -0.040 [-0.213, +0.178] | -0.010 [-0.045, +0.032] | **unresolved** | C_T +0.779 [+0.267, +1.325]; C_I -0.182 [-0.697, +0.327] | 0.071 / 0.060 | 50 | Pile vs Zipf at pair level |
| L2a | secondary | L | C_T + C_I → C_T + C_I + S_T | 4.092 | 3.949 | +0.143 [-0.074, +0.414] | +0.035 [-0.011, +0.083] | **unresolved** | S_T +0.312 [+0.110, +0.528] | 0.060 / 0.095 | 50 | does the T verb's preference add? |
| L2b | secondary | L | C_T + C_I → C_T + C_I + K_I | 4.092 | 4.043 | +0.049 [-0.063, +0.169] | +0.012 [-0.011, +0.033] | **unresolved** | K_I -0.498 [-0.933, -0.099] | 0.060 / 0.076 | 61 | does the I verb's competing frame add? |
| L2 | secondary | L | C_T + C_I → C_T + C_I + S_T + K_I | 4.092 | 3.927 | +0.165 [-0.091, +0.417] | +0.040 [-0.013, +0.081] | **unresolved** | S_T +0.295 [+0.088, +0.517]; K_I -0.436 [-0.899, -0.050] | 0.060 / 0.108 | 62 | both |

Sensitivity (D11): K = broad pronoun adjacency

| ID | family | outcome | a → b | MSE a | MSE b | Δ = a − b (95% CI) | Δ / MSE a (90% CI) | decision | added coefficient (95% CI) | in-sample R² a / b | cond(b) | question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I1 | primary | Δ_pass(I) | null → C | 3.061 | 2.722 | +0.339 [+0.102, +0.618]; Bonferroni 98.75% +0.339 [+0.058, +0.696] | +0.111 [+0.043, +0.179] | **b better** | C -0.728 [-0.975, -0.471] | 0.000 / 0.115 | 34 | does total exposure predict intransitive rejection? |
| I3 | primary | Δ_pass(I) | C → C + K | 2.722 | 2.615 | +0.107 [-0.034, +0.301]; Bonferroni 98.75% +0.107 [-0.092, +0.369] | +0.039 [-0.006, +0.086] | **unresolved** | K +1.012 [+0.474, +1.806] | 0.115 / 0.160 | 47 | does the competing frame add to total exposure? |
| I5 | secondary | Δ_pass(I) | Z → C | 2.587 | 2.722 | -0.135 [-0.246, -0.037] | -0.052 [-0.088, -0.019] | **a better** | C -0.728 [-0.975, -0.471] | 0.161 / 0.115 | 34 | Pile counts vs wordfreq Zipf |
| T2 | primary | Δ_pass(T) | C → C + B | 1.690 | 1.596 | +0.094 [-0.005, +0.239]; Bonferroni 98.75% +0.094 [-0.019, +0.295] | +0.056 [+0.002, +0.110] | **unresolved** | B +0.938 [+0.409, +1.475] | 0.026 / 0.086 | 52 | does the passive slot add? |
| T3 | primary | Δ_pass(T) | C → C + S | 1.690 | 1.413 | +0.276 [+0.101, +0.497]; Bonferroni 98.75% +0.276 [+0.067, +0.571] | +0.164 [+0.076, +0.238] | **b better** | S +0.418 [+0.279, +0.573] | 0.026 / 0.192 | 35 | does the BE/HAVE preference add? |
| L1 | secondary | L | Z_T + Z_I → C_T + C_I | 4.051 | 4.094 | -0.043 [-0.182, +0.133] | -0.011 [-0.038, +0.024] | **unresolved** | C_T +0.661 [+0.079, +1.251]; C_I -0.045 [-0.615, +0.524] | 0.071 / 0.056 | 49 | Pile vs Zipf at pair level |
| L2a | secondary | L | C_T + C_I → C_T + C_I + S_T | 4.094 | 3.947 | +0.147 [-0.066, +0.416] | +0.036 [-0.009, +0.082] | **unresolved** | S_T +0.324 [+0.127, +0.540] | 0.056 / 0.095 | 50 | does the T verb's preference add? |
| L2b | secondary | L | C_T + C_I → C_T + C_I + K_I | 4.094 | 4.024 | +0.070 [-0.133, +0.267] | +0.017 [-0.024, +0.051] | **unresolved** | K_I -0.931 [-1.702, -0.258] | 0.056 / 0.083 | 61 | does the I verb's competing frame add? |
| L2 | secondary | L | C_T + C_I → C_T + C_I + S_T + K_I | 4.094 | 3.905 | +0.189 [-0.091, +0.442] | +0.046 [-0.013, +0.090] | **unresolved** | S_T +0.310 [+0.118, +0.527]; K_I -0.870 [-1.693, -0.234] | 0.056 / 0.118 | 61 | both |

Sensitivity (D11): S = Fisher-based

| ID | family | outcome | a → b | MSE a | MSE b | Δ = a − b (95% CI) | Δ / MSE a (90% CI) | decision | added coefficient (95% CI) | in-sample R² a / b | cond(b) | question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I1 | primary | Δ_pass(I) | null → C | 3.061 | 2.722 | +0.339 [+0.102, +0.618]; Bonferroni 98.75% +0.339 [+0.058, +0.696] | +0.111 [+0.043, +0.179] | **b better** | C -0.728 [-0.975, -0.471] | 0.000 / 0.115 | 34 | does total exposure predict intransitive rejection? |
| I3 | primary | Δ_pass(I) | C → C + K | 2.722 | 2.682 | +0.040 [-0.119, +0.203]; Bonferroni 98.75% +0.040 [-0.184, +0.258] | +0.015 [-0.031, +0.057] | **unresolved** | K +0.588 [+0.240, +1.110] | 0.115 / 0.145 | 46 | does the competing frame add to total exposure? |
| I5 | secondary | Δ_pass(I) | Z → C | 2.587 | 2.722 | -0.135 [-0.246, -0.037] | -0.052 [-0.088, -0.019] | **a better** | C -0.728 [-0.975, -0.471] | 0.161 / 0.115 | 34 | Pile counts vs wordfreq Zipf |
| T2 | primary | Δ_pass(T) | C → C + B | 1.690 | 1.596 | +0.094 [-0.005, +0.239]; Bonferroni 98.75% +0.094 [-0.019, +0.295] | +0.056 [+0.002, +0.110] | **unresolved** | B +0.938 [+0.409, +1.475] | 0.026 / 0.086 | 52 | does the passive slot add? |
| T3 | primary | Δ_pass(T) | C → C + S | 1.690 | 1.384 | +0.306 [+0.119, +0.528]; Bonferroni 98.75% +0.306 [+0.087, +0.618] | +0.181 [+0.089, +0.255] | **b better** | S +0.003 [+0.002, +0.004] | 0.026 / 0.209 | 1590 | does the BE/HAVE preference add? |
| L1 | secondary | L | Z_T + Z_I → C_T + C_I | 4.051 | 4.094 | -0.043 [-0.182, +0.133] | -0.011 [-0.038, +0.024] | **unresolved** | C_T +0.661 [+0.079, +1.251]; C_I -0.045 [-0.615, +0.524] | 0.071 / 0.056 | 49 | Pile vs Zipf at pair level |
| L2a | secondary | L | C_T + C_I → C_T + C_I + S_T | 4.094 | 3.890 | +0.204 [-0.011, +0.471] | +0.050 [+0.003, +0.094] | **unresolved** | S_T +0.003 [+0.001, +0.004] | 0.056 / 0.109 | 1617 | does the T verb's preference add? |
| L2b | secondary | L | C_T + C_I → C_T + C_I + K_I | 4.094 | 4.032 | +0.062 [-0.085, +0.217] | +0.015 [-0.015, +0.042] | **unresolved** | K_I -0.615 [-1.120, -0.196] | 0.056 / 0.079 | 62 | does the I verb's competing frame add? |
| L2 | secondary | L | C_T + C_I → C_T + C_I + S_T + K_I | 4.094 | 3.841 | +0.253 [-0.008, +0.516] | +0.062 [+0.005, +0.103] | **unresolved** | S_T +0.002 [+0.001, +0.004]; K_I -0.602 [-1.118, -0.195] | 0.056 / 0.131 | 1876 | both |

Sensitivity (D11): S = BE vs pronoun frame

| ID | family | outcome | a → b | MSE a | MSE b | Δ = a − b (95% CI) | Δ / MSE a (90% CI) | decision | added coefficient (95% CI) | in-sample R² a / b | cond(b) | question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I1 | primary | Δ_pass(I) | null → C | 3.061 | 2.722 | +0.339 [+0.102, +0.618]; Bonferroni 98.75% +0.339 [+0.058, +0.696] | +0.111 [+0.043, +0.179] | **b better** | C -0.728 [-0.975, -0.471] | 0.000 / 0.115 | 34 | does total exposure predict intransitive rejection? |
| I3 | primary | Δ_pass(I) | C → C + K | 2.722 | 2.682 | +0.040 [-0.119, +0.203]; Bonferroni 98.75% +0.040 [-0.184, +0.258] | +0.015 [-0.031, +0.057] | **unresolved** | K +0.588 [+0.240, +1.110] | 0.115 / 0.145 | 46 | does the competing frame add to total exposure? |
| I5 | secondary | Δ_pass(I) | Z → C | 2.587 | 2.722 | -0.135 [-0.246, -0.037] | -0.052 [-0.088, -0.019] | **a better** | C -0.728 [-0.975, -0.471] | 0.161 / 0.115 | 34 | Pile counts vs wordfreq Zipf |
| T2 | primary | Δ_pass(T) | C → C + B | 1.690 | 1.596 | +0.094 [-0.005, +0.239]; Bonferroni 98.75% +0.094 [-0.019, +0.295] | +0.056 [+0.002, +0.110] | **unresolved** | B +0.938 [+0.409, +1.475] | 0.026 / 0.086 | 52 | does the passive slot add? |
| T3 | primary | Δ_pass(T) | C → C + S | 1.690 | 1.591 | +0.099 [-0.000, +0.232]; Bonferroni 98.75% +0.099 [-0.019, +0.280] | +0.058 [+0.008, +0.110] | **unresolved** | S +0.181 [+0.098, +0.272] | 0.026 / 0.090 | 36 | does the BE/HAVE preference add? |
| L1 | secondary | L | Z_T + Z_I → C_T + C_I | 4.051 | 4.094 | -0.043 [-0.182, +0.133] | -0.011 [-0.038, +0.024] | **unresolved** | C_T +0.661 [+0.079, +1.251]; C_I -0.045 [-0.615, +0.524] | 0.071 / 0.056 | 49 | Pile vs Zipf at pair level |
| L2a | secondary | L | C_T + C_I → C_T + C_I + S_T | 4.094 | 4.038 | +0.056 [-0.059, +0.204] | +0.014 [-0.011, +0.040] | **unresolved** | S_T +0.143 [+0.021, +0.274] | 0.056 / 0.072 | 50 | does the T verb's preference add? |
| L2b | secondary | L | C_T + C_I → C_T + C_I + K_I | 4.094 | 4.032 | +0.062 [-0.085, +0.217] | +0.015 [-0.015, +0.042] | **unresolved** | K_I -0.615 [-1.120, -0.196] | 0.056 / 0.079 | 62 | does the I verb's competing frame add? |
| L2 | secondary | L | C_T + C_I → C_T + C_I + S_T + K_I | 4.094 | 3.991 | +0.103 [-0.088, +0.271] | +0.025 [-0.016, +0.054] | **unresolved** | S_T +0.132 [+0.006, +0.262]; K_I -0.584 [-1.102, -0.164] | 0.056 / 0.092 | 63 | both |

Sensitivity (D11): S = was vs has

| ID | family | outcome | a → b | MSE a | MSE b | Δ = a − b (95% CI) | Δ / MSE a (90% CI) | decision | added coefficient (95% CI) | in-sample R² a / b | cond(b) | question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I1 | primary | Δ_pass(I) | null → C | 3.061 | 2.722 | +0.339 [+0.102, +0.618]; Bonferroni 98.75% +0.339 [+0.058, +0.696] | +0.111 [+0.043, +0.179] | **b better** | C -0.728 [-0.975, -0.471] | 0.000 / 0.115 | 34 | does total exposure predict intransitive rejection? |
| I3 | primary | Δ_pass(I) | C → C + K | 2.722 | 2.682 | +0.040 [-0.119, +0.203]; Bonferroni 98.75% +0.040 [-0.184, +0.258] | +0.015 [-0.031, +0.057] | **unresolved** | K +0.588 [+0.240, +1.110] | 0.115 / 0.145 | 46 | does the competing frame add to total exposure? |
| I5 | secondary | Δ_pass(I) | Z → C | 2.587 | 2.722 | -0.135 [-0.246, -0.037] | -0.052 [-0.088, -0.019] | **a better** | C -0.728 [-0.975, -0.471] | 0.161 / 0.115 | 34 | Pile counts vs wordfreq Zipf |
| T2 | primary | Δ_pass(T) | C → C + B | 1.690 | 1.596 | +0.094 [-0.005, +0.239]; Bonferroni 98.75% +0.094 [-0.019, +0.295] | +0.056 [+0.002, +0.110] | **unresolved** | B +0.938 [+0.409, +1.475] | 0.026 / 0.086 | 52 | does the passive slot add? |
| T3 | primary | Δ_pass(T) | C → C + S | 1.690 | 1.301 | +0.388 [+0.158, +0.644]; Bonferroni 98.75% +0.388 [+0.099, +0.767] | +0.230 [+0.117, +0.320] | **b better** | S +0.405 [+0.279, +0.551] | 0.026 / 0.262 | 35 | does the BE/HAVE preference add? |
| L1 | secondary | L | Z_T + Z_I → C_T + C_I | 4.051 | 4.094 | -0.043 [-0.182, +0.133] | -0.011 [-0.038, +0.024] | **unresolved** | C_T +0.661 [+0.079, +1.251]; C_I -0.045 [-0.615, +0.524] | 0.071 / 0.056 | 49 | Pile vs Zipf at pair level |
| L2a | secondary | L | C_T + C_I → C_T + C_I + S_T | 4.094 | 3.892 | +0.201 [-0.069, +0.493] | +0.049 [-0.007, +0.103] | **unresolved** | S_T +0.306 [+0.143, +0.502] | 0.056 / 0.108 | 50 | does the T verb's preference add? |
| L2b | secondary | L | C_T + C_I → C_T + C_I + K_I | 4.094 | 4.032 | +0.062 [-0.085, +0.217] | +0.015 [-0.015, +0.042] | **unresolved** | K_I -0.615 [-1.120, -0.196] | 0.056 / 0.079 | 62 | does the I verb's competing frame add? |
| L2 | secondary | L | C_T + C_I → C_T + C_I + S_T + K_I | 4.094 | 3.880 | +0.214 [-0.111, +0.509] | +0.052 [-0.015, +0.102] | **unresolved** | S_T +0.289 [+0.126, +0.488]; K_I -0.530 [-1.089, -0.122] | 0.056 / 0.125 | 63 | both |

Sensitivity (D11): smoothing 0.1

| ID | family | outcome | a → b | MSE a | MSE b | Δ = a − b (95% CI) | Δ / MSE a (90% CI) | decision | added coefficient (95% CI) | in-sample R² a / b | cond(b) | question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I1 | primary | Δ_pass(I) | null → C | 3.061 | 2.722 | +0.339 [+0.102, +0.618]; Bonferroni 98.75% +0.339 [+0.058, +0.696] | +0.111 [+0.043, +0.179] | **b better** | C -0.728 [-0.975, -0.471] | 0.000 / 0.115 | 34 | does total exposure predict intransitive rejection? |
| I3 | primary | Δ_pass(I) | C → C + K | 2.722 | 2.702 | +0.020 [-0.179, +0.201]; Bonferroni 98.75% +0.020 [-0.293, +0.256] | +0.008 [-0.051, +0.056] | **unresolved** | K +0.543 [+0.213, +1.107] | 0.115 / 0.143 | 45 | does the competing frame add to total exposure? |
| I5 | secondary | Δ_pass(I) | Z → C | 2.587 | 2.722 | -0.135 [-0.246, -0.037] | -0.052 [-0.088, -0.019] | **a better** | C -0.728 [-0.975, -0.471] | 0.161 / 0.115 | 34 | Pile counts vs wordfreq Zipf |
| T2 | primary | Δ_pass(T) | C → C + B | 1.690 | 1.596 | +0.094 [-0.005, +0.239]; Bonferroni 98.75% +0.094 [-0.019, +0.295] | +0.056 [+0.002, +0.110] | **unresolved** | B +0.938 [+0.409, +1.475] | 0.026 / 0.086 | 52 | does the passive slot add? |
| T3 | primary | Δ_pass(T) | C → C + S | 1.690 | 1.413 | +0.276 [+0.101, +0.497]; Bonferroni 98.75% +0.276 [+0.067, +0.571] | +0.163 [+0.076, +0.238] | **b better** | S +0.418 [+0.279, +0.573] | 0.026 / 0.192 | 35 | does the BE/HAVE preference add? |
| L1 | secondary | L | Z_T + Z_I → C_T + C_I | 4.051 | 4.094 | -0.043 [-0.182, +0.133] | -0.011 [-0.038, +0.024] | **unresolved** | C_T +0.661 [+0.079, +1.251]; C_I -0.045 [-0.615, +0.524] | 0.071 / 0.056 | 49 | Pile vs Zipf at pair level |
| L2a | secondary | L | C_T + C_I → C_T + C_I + S_T | 4.094 | 3.947 | +0.147 [-0.066, +0.416] | +0.036 [-0.009, +0.082] | **unresolved** | S_T +0.324 [+0.127, +0.540] | 0.056 / 0.095 | 50 | does the T verb's preference add? |
| L2b | secondary | L | C_T + C_I → C_T + C_I + K_I | 4.094 | 4.038 | +0.056 [-0.091, +0.201] | +0.014 [-0.015, +0.039] | **unresolved** | K_I -0.583 [-1.100, -0.196] | 0.056 / 0.079 | 61 | does the I verb's competing frame add? |
| L2 | secondary | L | C_T + C_I → C_T + C_I + S_T + K_I | 4.094 | 3.938 | +0.156 [-0.137, +0.404] | +0.038 [-0.021, +0.080] | **unresolved** | S_T +0.302 [+0.099, +0.523]; K_I -0.510 [-1.087, -0.112] | 0.056 / 0.112 | 62 | both |

Sensitivity (D11): smoothing 1.0

| ID | family | outcome | a → b | MSE a | MSE b | Δ = a − b (95% CI) | Δ / MSE a (90% CI) | decision | added coefficient (95% CI) | in-sample R² a / b | cond(b) | question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I1 | primary | Δ_pass(I) | null → C | 3.061 | 2.722 | +0.339 [+0.102, +0.618]; Bonferroni 98.75% +0.339 [+0.058, +0.696] | +0.111 [+0.043, +0.179] | **b better** | C -0.728 [-0.975, -0.471] | 0.000 / 0.115 | 34 | does total exposure predict intransitive rejection? |
| I3 | primary | Δ_pass(I) | C → C + K | 2.722 | 2.675 | +0.048 [-0.096, +0.203]; Bonferroni 98.75% +0.048 [-0.151, +0.263] | +0.017 [-0.025, +0.058] | **unresolved** | K +0.608 [+0.252, +1.112] | 0.115 / 0.146 | 46 | does the competing frame add to total exposure? |
| I5 | secondary | Δ_pass(I) | Z → C | 2.587 | 2.722 | -0.135 [-0.246, -0.037] | -0.052 [-0.088, -0.019] | **a better** | C -0.728 [-0.975, -0.471] | 0.161 / 0.115 | 34 | Pile counts vs wordfreq Zipf |
| T2 | primary | Δ_pass(T) | C → C + B | 1.690 | 1.596 | +0.094 [-0.005, +0.239]; Bonferroni 98.75% +0.094 [-0.019, +0.295] | +0.056 [+0.002, +0.110] | **unresolved** | B +0.938 [+0.409, +1.475] | 0.026 / 0.086 | 52 | does the passive slot add? |
| T3 | primary | Δ_pass(T) | C → C + S | 1.690 | 1.413 | +0.276 [+0.101, +0.497]; Bonferroni 98.75% +0.276 [+0.067, +0.571] | +0.164 [+0.076, +0.237] | **b better** | S +0.418 [+0.280, +0.574] | 0.026 / 0.192 | 35 | does the BE/HAVE preference add? |
| L1 | secondary | L | Z_T + Z_I → C_T + C_I | 4.051 | 4.094 | -0.043 [-0.182, +0.133] | -0.011 [-0.038, +0.024] | **unresolved** | C_T +0.661 [+0.079, +1.251]; C_I -0.045 [-0.615, +0.524] | 0.071 / 0.056 | 49 | Pile vs Zipf at pair level |
| L2a | secondary | L | C_T + C_I → C_T + C_I + S_T | 4.094 | 3.947 | +0.147 [-0.066, +0.416] | +0.036 [-0.009, +0.082] | **unresolved** | S_T +0.325 [+0.128, +0.541] | 0.056 / 0.095 | 50 | does the T verb's preference add? |
| L2b | secondary | L | C_T + C_I → C_T + C_I + K_I | 4.094 | 4.030 | +0.064 [-0.082, +0.224] | +0.016 [-0.015, +0.044] | **unresolved** | K_I -0.628 [-1.143, -0.196] | 0.056 / 0.080 | 62 | does the I verb's competing frame add? |
| L2 | secondary | L | C_T + C_I → C_T + C_I + S_T + K_I | 4.094 | 3.921 | +0.173 [-0.093, +0.419] | +0.042 [-0.014, +0.084] | **unresolved** | S_T +0.305 [+0.104, +0.522]; K_I -0.563 [-1.110, -0.132] | 0.056 / 0.113 | 63 | both |

- T3 decomposition (D11): C → C + B + P: Δ +0.267 [+0.081, +0.485] → b better; B coefficient +0.861 [+0.399, +1.298], P coefficient -1.033 [-1.577, -0.618] (a preference effect from more BE has B > 0; from less HAVE, P < 0).

### Single-token subset (a_V = Δ_pass + a verb-independent constant; descriptive)

- I1 (65 single-token I verbs): MSE 2.739 → 2.479, Δ +0.261 [-0.251, +0.848] → unresolved; added coefficient -1.477 [-2.277, -0.549].
- I3 (65 single-token I verbs): MSE 2.479 → 2.500, Δ -0.021 [-0.860, +0.358] → unresolved; added coefficient +0.429 [+0.094, +1.500].
- T2 (69 single-token T verbs): MSE 1.564 → 1.605, Δ -0.041 [-0.539, +0.416] → unresolved; added coefficient +0.805 [-0.604, +1.919].
- T3 (69 single-token T verbs): MSE 1.564 → 1.483, Δ +0.081 [-0.240, +0.482] → unresolved; added coefficient +0.200 [-0.007, +0.455].

## Model comparison: curated (126 pairs)

- r(Δ_pass(T), ·): Z -0.14, C -0.14, B -0.06, P -0.32, K -0.24, K_broad -0.21, S +0.44, S_fisher +0.39, S_washas +0.66, N +0.08; r(C, K) +0.80; cond(C + K) 41
- r(Δ_pass(I), ·): Z -0.58, C -0.50, B -0.24, P -0.69, K -0.30, K_broad -0.35, S +0.63, S_fisher +0.45, S_washas +0.76, N +0.30; r(C, K) +0.86; cond(C + K) 48

| ID | family | outcome | a → b | MSE a | MSE b | Δ = a − b (95% CI) | Δ / MSE a (90% CI) | decision | added coefficient (95% CI) | in-sample R² a / b | cond(b) | question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I1 | primary | Δ_pass(I) | null → C | 3.483 | 2.642 | +0.841 [+0.365, +1.389]; Bonferroni 98.75% +0.841 [+0.269, +1.513] | +0.241 [+0.127, +0.350] | **b better** | C -1.080 [-1.359, -0.788] | -0.000 / 0.250 | 33 | does total exposure predict intransitive rejection? |
| I2 | secondary | Δ_pass(I) | C → K | 2.642 | 3.222 | -0.580 [-0.961, -0.312] | -0.220 [-0.345, -0.123] | **a better** | K -0.562 [-0.847, -0.275] | 0.250 / 0.088 | 11 | competing-frame count vs total count |
| I3 | primary | Δ_pass(I) | C → C + K | 2.642 | 2.433 | +0.209 [-0.072, +0.517]; Bonferroni 98.75% +0.209 [-0.145, +0.607] | +0.079 [-0.012, +0.154] | **unresolved** | K +1.018 [+0.511, +1.642] | 0.250 / 0.323 | 48 | does the competing frame add to total exposure? |
| I4 | secondary | Δ_pass(I) | C → C + B | 2.642 | 2.568 | +0.074 [-0.203, +0.391] | +0.028 [-0.054, +0.118] | **unresolved** | B +0.567 [+0.081, +1.051] | 0.250 / 0.288 | 40 | does the passive slot add? |
| I5 | secondary | Δ_pass(I) | Z → C | 2.346 | 2.642 | -0.296 [-0.487, -0.142] | -0.126 [-0.199, -0.064] | **a better** | C -1.080 [-1.359, -0.788] | 0.336 / 0.250 | 33 | Pile counts vs wordfreq Zipf |
| T1 | secondary | Δ_pass(T) | null → C | 2.026 | 2.028 | -0.001 [-0.118, +0.117] | -0.001 [-0.046, +0.037] | **unresolved** | C -0.225 [-0.546, +0.041] | 0.000 / 0.020 | 31 | does total exposure predict transitive acceptance? |
| T2 | primary | Δ_pass(T) | C → C + B | 2.028 | 2.006 | +0.021 [-0.109, +0.145]; Bonferroni 98.75% +0.021 [-0.148, +0.222] | +0.011 [-0.042, +0.052] | **unresolved** | B +0.563 [+0.015, +1.198] | 0.020 / 0.045 | 45 | does the passive slot add? |
| T3 | primary | Δ_pass(T) | C → C + S | 2.028 | 1.665 | +0.362 [+0.013, +0.811]; Bonferroni 98.75% +0.362 [-0.087, +0.966] | +0.179 [+0.028, +0.302] | **b better** | S +0.547 [+0.314, +0.776] | 0.020 / 0.233 | 32 | does the BE/HAVE preference add? |
| T4 | secondary | Δ_pass(T) | Z → C | 2.022 | 2.028 | -0.006 [-0.049, +0.044] | -0.003 [-0.019, +0.014] | **equivalent** | C -0.225 [-0.546, +0.041] | 0.019 / 0.020 | 31 | Pile vs Zipf |
| L1 | secondary | L | Z_T + Z_I → C_T + C_I | 4.321 | 4.615 | -0.293 [-0.872, +0.203] | -0.068 [-0.157, +0.024] | **unresolved** | C_T +1.147 [+0.067, +2.278]; C_I -0.235 [-1.346, +0.826] | 0.167 / 0.139 | 46 | Pile vs Zipf at pair level |
| L2a | secondary | L | C_T + C_I → C_T + C_I + S_T | 4.615 | 4.516 | +0.099 [-0.477, +0.648] | +0.021 [-0.077, +0.116] | **unresolved** | S_T +0.428 [+0.130, +0.727] | 0.139 / 0.191 | 46 | does the T verb's preference add? |
| L2b | secondary | L | C_T + C_I → C_T + C_I + K_I | 4.615 | 4.400 | +0.215 [-0.326, +0.790] | +0.047 [-0.053, +0.134] | **unresolved** | K_I -1.094 [-1.840, -0.367] | 0.139 / 0.197 | 63 | does the I verb's competing frame add? |
| L2 | secondary | L | C_T + C_I → C_T + C_I + S_T + K_I | 4.615 | 4.123 | +0.492 [-0.278, +1.235] | +0.107 [-0.037, +0.212] | **unresolved** | S_T +0.528 [+0.226, +0.834]; K_I -1.329 [-1.988, -0.705] | 0.139 / 0.274 | 63 | both |

Sensitivity (curated): token count added (log participle tokens)

| ID | family | outcome | a → b | MSE a | MSE b | Δ = a − b (95% CI) | Δ / MSE a (90% CI) | decision | added coefficient (95% CI) | in-sample R² a / b | cond(b) | question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I1 | primary | Δ_pass(I) | null → C | 3.202 | 2.520 | +0.682 [+0.315, +1.115]; Bonferroni 98.75% +0.682 [+0.238, +1.238] | +0.213 [+0.115, +0.305] | **b better** | C -1.723 [-2.177, -1.307] | 0.088 / 0.292 | 74 | does total exposure predict intransitive rejection? |
| I3 | primary | Δ_pass(I) | C → C + K | 2.520 | 2.365 | +0.155 [-0.101, +0.442]; Bonferroni 98.75% +0.155 [-0.191, +0.541] | +0.062 [-0.025, +0.135] | **unresolved** | K +0.903 [+0.371, +1.540] | 0.292 / 0.348 | 86 | does the competing frame add to total exposure? |
| I5 | secondary | Δ_pass(I) | Z → C | 2.167 | 2.520 | -0.353 [-0.622, -0.139] | -0.163 [-0.274, -0.072] | **a better** | C -1.723 [-2.177, -1.307] | 0.397 / 0.292 | 74 | Pile counts vs wordfreq Zipf |
| T2 | primary | Δ_pass(T) | C → C + B | 2.056 | 2.040 | +0.016 [-0.136, +0.154]; Bonferroni 98.75% +0.016 [-0.194, +0.226] | +0.008 [-0.048, +0.053] | **unresolved** | B +0.556 [+0.007, +1.213] | 0.022 / 0.047 | 87 | does the passive slot add? |
| T3 | primary | Δ_pass(T) | C → C + S | 2.056 | 1.686 | +0.370 [+0.005, +0.830]; Bonferroni 98.75% +0.370 [-0.117, +0.985] | +0.180 [+0.026, +0.301] | **b better** | S +0.548 [+0.311, +0.781] | 0.022 / 0.233 | 67 | does the BE/HAVE preference add? |
| L1 | secondary | L | Z_T + Z_I → C_T + C_I | 4.289 | 4.594 | -0.305 [-0.860, +0.097] | -0.071 [-0.167, +0.007] | **unresolved** | C_T +1.148 [+0.154, +2.216]; C_I +0.345 [-0.921, +1.508] | 0.212 / 0.160 | 117 | Pile vs Zipf at pair level |
| L2a | secondary | L | C_T + C_I → C_T + C_I + S_T | 4.594 | 4.481 | +0.113 [-0.512, +0.713] | +0.025 [-0.081, +0.118] | **unresolved** | S_T +0.431 [+0.118, +0.738] | 0.160 / 0.212 | 118 | does the T verb's preference add? |
| L2b | secondary | L | C_T + C_I → C_T + C_I + K_I | 4.594 | 4.416 | +0.178 [-0.343, +0.804] | +0.039 [-0.058, +0.128] | **unresolved** | K_I -1.016 [-1.789, -0.289] | 0.160 / 0.207 | 127 | does the I verb's competing frame add? |
| L2 | secondary | L | C_T + C_I → C_T + C_I + S_T + K_I | 4.594 | 4.144 | +0.450 [-0.386, +1.273] | +0.098 [-0.053, +0.208] | **unresolved** | S_T +0.520 [+0.212, +0.830]; K_I -1.244 [-1.936, -0.593] | 0.160 / 0.280 | 128 | both |

Sensitivity (curated): c_total marked only

| ID | family | outcome | a → b | MSE a | MSE b | Δ = a − b (95% CI) | Δ / MSE a (90% CI) | decision | added coefficient (95% CI) | in-sample R² a / b | cond(b) | question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I1 | primary | Δ_pass(I) | null → C | 3.483 | 2.640 | +0.844 [+0.366, +1.393]; Bonferroni 98.75% +0.844 [+0.269, +1.516] | +0.242 [+0.127, +0.351] | **b better** | C -1.081 [-1.360, -0.791] | -0.000 / 0.251 | 33 | does total exposure predict intransitive rejection? |
| I3 | primary | Δ_pass(I) | C → C + K | 2.640 | 2.429 | +0.210 [-0.070, +0.518]; Bonferroni 98.75% +0.210 [-0.143, +0.611] | +0.080 [-0.012, +0.155] | **unresolved** | K +1.018 [+0.511, +1.641] | 0.251 / 0.324 | 48 | does the competing frame add to total exposure? |
| I5 | secondary | Δ_pass(I) | Z → C | 2.346 | 2.640 | -0.293 [-0.484, -0.139] | -0.125 [-0.199, -0.063] | **a better** | C -1.081 [-1.360, -0.791] | 0.336 / 0.251 | 33 | Pile counts vs wordfreq Zipf |
| T2 | primary | Δ_pass(T) | C → C + B | 2.027 | 2.002 | +0.025 [-0.108, +0.153]; Bonferroni 98.75% +0.025 [-0.143, +0.232] | +0.012 [-0.041, +0.055] | **unresolved** | B +0.580 [+0.031, +1.230] | 0.020 / 0.047 | 44 | does the passive slot add? |
| T3 | primary | Δ_pass(T) | C → C + S | 2.027 | 1.663 | +0.363 [+0.013, +0.812]; Bonferroni 98.75% +0.363 [-0.086, +0.967] | +0.179 [+0.028, +0.302] | **b better** | S +0.547 [+0.315, +0.776] | 0.020 / 0.234 | 32 | does the BE/HAVE preference add? |
| L1 | secondary | L | Z_T + Z_I → C_T + C_I | 4.321 | 4.623 | -0.302 [-0.880, +0.194] | -0.070 [-0.160, +0.022] | **unresolved** | C_T +1.128 [+0.048, +2.265]; C_I -0.216 [-1.328, +0.844] | 0.167 / 0.137 | 46 | Pile vs Zipf at pair level |
| L2a | secondary | L | C_T + C_I → C_T + C_I + S_T | 4.623 | 4.523 | +0.101 [-0.476, +0.655] | +0.022 [-0.077, +0.116] | **unresolved** | S_T +0.429 [+0.130, +0.728] | 0.137 / 0.190 | 46 | does the T verb's preference add? |
| L2b | secondary | L | C_T + C_I → C_T + C_I + K_I | 4.623 | 4.407 | +0.217 [-0.325, +0.794] | +0.047 [-0.053, +0.135] | **unresolved** | K_I -1.094 [-1.840, -0.368] | 0.137 / 0.196 | 62 | does the I verb's competing frame add? |
| L2 | secondary | L | C_T + C_I → C_T + C_I + S_T + K_I | 4.623 | 4.127 | +0.496 [-0.279, +1.245] | +0.107 [-0.037, +0.214] | **unresolved** | S_T +0.529 [+0.226, +0.835]; K_I -1.329 [-1.984, -0.710] | 0.137 / 0.273 | 63 | both |

Sensitivity (curated): c_total upper bound

| ID | family | outcome | a → b | MSE a | MSE b | Δ = a − b (95% CI) | Δ / MSE a (90% CI) | decision | added coefficient (95% CI) | in-sample R² a / b | cond(b) | question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I1 | primary | Δ_pass(I) | null → C | 3.483 | 2.703 | +0.780 [+0.312, +1.309]; Bonferroni 98.75% +0.780 [+0.229, +1.433] | +0.224 [+0.112, +0.333] | **b better** | C -1.040 [-1.319, -0.739] | -0.000 / 0.236 | 33 | does total exposure predict intransitive rejection? |
| I3 | primary | Δ_pass(I) | C → C + K | 2.703 | 2.561 | +0.143 [-0.086, +0.426]; Bonferroni 98.75% +0.143 [-0.146, +0.510] | +0.053 [-0.021, +0.119] | **unresolved** | K +0.851 [+0.351, +1.435] | 0.236 / 0.290 | 48 | does the competing frame add to total exposure? |
| I5 | secondary | Δ_pass(I) | Z → C | 2.346 | 2.703 | -0.357 [-0.560, -0.186] | -0.152 [-0.234, -0.084] | **a better** | C -1.040 [-1.319, -0.739] | 0.336 / 0.236 | 33 | Pile counts vs wordfreq Zipf |
| T2 | primary | Δ_pass(T) | C → C + B | 2.038 | 2.046 | -0.007 [-0.114, +0.075]; Bonferroni 98.75% -0.007 [-0.158, +0.129] | -0.004 [-0.044, +0.024] | **unresolved** | B +0.281 [-0.224, +0.847] | 0.012 / 0.019 | 44 | does the passive slot add? |
| T3 | primary | Δ_pass(T) | C → C + S | 2.038 | 1.679 | +0.360 [+0.010, +0.808]; Bonferroni 98.75% +0.360 [-0.099, +0.966] | +0.176 [+0.027, +0.298] | **b better** | S +0.547 [+0.312, +0.781] | 0.012 / 0.225 | 32 | does the BE/HAVE preference add? |
| L1 | secondary | L | Z_T + Z_I → C_T + C_I | 4.321 | 4.409 | -0.087 [-0.630, +0.475] | -0.020 [-0.113, +0.083] | **unresolved** | C_T +1.399 [+0.544, +2.278]; C_I -0.480 [-1.384, +0.337] | 0.167 / 0.164 | 47 | Pile vs Zipf at pair level |
| L2a | secondary | L | C_T + C_I → C_T + C_I + S_T | 4.409 | 4.323 | +0.086 [-0.474, +0.611] | +0.019 [-0.078, +0.114] | **unresolved** | S_T +0.408 [+0.106, +0.698] | 0.164 / 0.211 | 47 | does the T verb's preference add? |
| L2b | secondary | L | C_T + C_I → C_T + C_I + K_I | 4.409 | 4.242 | +0.167 [-0.293, +0.670] | +0.038 [-0.054, +0.115] | **unresolved** | K_I -1.009 [-1.700, -0.323] | 0.164 / 0.215 | 62 | does the I verb's competing frame add? |
| L2 | secondary | L | C_T + C_I → C_T + C_I + S_T + K_I | 4.409 | 4.041 | +0.368 [-0.377, +1.017] | +0.083 [-0.060, +0.188] | **unresolved** | S_T +0.480 [+0.189, +0.772]; K_I -1.173 [-1.774, -0.555] | 0.164 / 0.279 | 63 | both |

Sensitivity (curated): K = broad pronoun adjacency

| ID | family | outcome | a → b | MSE a | MSE b | Δ = a − b (95% CI) | Δ / MSE a (90% CI) | decision | added coefficient (95% CI) | in-sample R² a / b | cond(b) | question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I1 | primary | Δ_pass(I) | null → C | 3.483 | 2.642 | +0.841 [+0.365, +1.389]; Bonferroni 98.75% +0.841 [+0.269, +1.513] | +0.241 [+0.127, +0.350] | **b better** | C -1.080 [-1.359, -0.788] | -0.000 / 0.250 | 33 | does total exposure predict intransitive rejection? |
| I3 | primary | Δ_pass(I) | C → C + K | 2.642 | 2.270 | +0.372 [+0.062, +0.758]; Bonferroni 98.75% +0.372 [-0.006, +0.940] | +0.141 [+0.038, +0.231] | **b better** | K +2.025 [+1.204, +3.041] | 0.250 / 0.370 | 51 | does the competing frame add to total exposure? |
| I5 | secondary | Δ_pass(I) | Z → C | 2.346 | 2.642 | -0.296 [-0.487, -0.142] | -0.126 [-0.199, -0.064] | **a better** | C -1.080 [-1.359, -0.788] | 0.336 / 0.250 | 33 | Pile counts vs wordfreq Zipf |
| T2 | primary | Δ_pass(T) | C → C + B | 2.028 | 2.006 | +0.021 [-0.109, +0.145]; Bonferroni 98.75% +0.021 [-0.148, +0.222] | +0.011 [-0.042, +0.052] | **unresolved** | B +0.563 [+0.015, +1.198] | 0.020 / 0.045 | 45 | does the passive slot add? |
| T3 | primary | Δ_pass(T) | C → C + S | 2.028 | 1.665 | +0.362 [+0.013, +0.811]; Bonferroni 98.75% +0.362 [-0.087, +0.966] | +0.179 [+0.028, +0.302] | **b better** | S +0.547 [+0.314, +0.776] | 0.020 / 0.233 | 32 | does the BE/HAVE preference add? |
| L1 | secondary | L | Z_T + Z_I → C_T + C_I | 4.321 | 4.615 | -0.293 [-0.872, +0.203] | -0.068 [-0.157, +0.024] | **unresolved** | C_T +1.147 [+0.067, +2.278]; C_I -0.235 [-1.346, +0.826] | 0.167 / 0.139 | 46 | Pile vs Zipf at pair level |
| L2a | secondary | L | C_T + C_I → C_T + C_I + S_T | 4.615 | 4.516 | +0.099 [-0.477, +0.648] | +0.021 [-0.077, +0.116] | **unresolved** | S_T +0.428 [+0.130, +0.727] | 0.139 / 0.191 | 46 | does the T verb's preference add? |
| L2b | secondary | L | C_T + C_I → C_T + C_I + K_I | 4.615 | 4.201 | +0.413 [-0.196, +1.014] | +0.090 [-0.020, +0.184] | **unresolved** | K_I -2.054 [-3.116, -1.018] | 0.139 / 0.223 | 65 | does the I verb's competing frame add? |
| L2 | secondary | L | C_T + C_I → C_T + C_I + S_T + K_I | 4.615 | 4.019 | +0.596 [-0.117, +1.215] | +0.129 [+0.003, +0.217] | **unresolved** | S_T +0.481 [+0.216, +0.779]; K_I -2.216 [-3.344, -1.216] | 0.139 / 0.289 | 65 | both |

Sensitivity (curated): S = Fisher-based

| ID | family | outcome | a → b | MSE a | MSE b | Δ = a − b (95% CI) | Δ / MSE a (90% CI) | decision | added coefficient (95% CI) | in-sample R² a / b | cond(b) | question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I1 | primary | Δ_pass(I) | null → C | 3.483 | 2.642 | +0.841 [+0.365, +1.389]; Bonferroni 98.75% +0.841 [+0.269, +1.513] | +0.241 [+0.127, +0.350] | **b better** | C -1.080 [-1.359, -0.788] | -0.000 / 0.250 | 33 | does total exposure predict intransitive rejection? |
| I3 | primary | Δ_pass(I) | C → C + K | 2.642 | 2.433 | +0.209 [-0.072, +0.517]; Bonferroni 98.75% +0.209 [-0.145, +0.607] | +0.079 [-0.012, +0.154] | **unresolved** | K +1.018 [+0.511, +1.642] | 0.250 / 0.323 | 48 | does the competing frame add to total exposure? |
| I5 | secondary | Δ_pass(I) | Z → C | 2.346 | 2.642 | -0.296 [-0.487, -0.142] | -0.126 [-0.199, -0.064] | **a better** | C -1.080 [-1.359, -0.788] | 0.336 / 0.250 | 33 | Pile counts vs wordfreq Zipf |
| T2 | primary | Δ_pass(T) | C → C + B | 2.028 | 2.006 | +0.021 [-0.109, +0.145]; Bonferroni 98.75% +0.021 [-0.148, +0.222] | +0.011 [-0.042, +0.052] | **unresolved** | B +0.563 [+0.015, +1.198] | 0.020 / 0.045 | 45 | does the passive slot add? |
| T3 | primary | Δ_pass(T) | C → C + S | 2.028 | 1.715 | +0.313 [+0.016, +0.749]; Bonferroni 98.75% +0.313 [-0.043, +0.916] | +0.154 [+0.023, +0.276] | **b better** | S +0.003 [+0.002, +0.004] | 0.020 / 0.201 | 1448 | does the BE/HAVE preference add? |
| L1 | secondary | L | Z_T + Z_I → C_T + C_I | 4.321 | 4.615 | -0.293 [-0.872, +0.203] | -0.068 [-0.157, +0.024] | **unresolved** | C_T +1.147 [+0.067, +2.278]; C_I -0.235 [-1.346, +0.826] | 0.167 / 0.139 | 46 | Pile vs Zipf at pair level |
| L2a | secondary | L | C_T + C_I → C_T + C_I + S_T | 4.615 | 4.329 | +0.285 [-0.164, +0.839] | +0.062 [-0.022, +0.154] | **unresolved** | S_T +0.003 [+0.001, +0.005] | 0.139 / 0.216 | 1505 | does the T verb's preference add? |
| L2b | secondary | L | C_T + C_I → C_T + C_I + K_I | 4.615 | 4.400 | +0.215 [-0.326, +0.790] | +0.047 [-0.053, +0.134] | **unresolved** | K_I -1.094 [-1.840, -0.367] | 0.139 / 0.197 | 63 | does the I verb's competing frame add? |
| L2 | secondary | L | C_T + C_I → C_T + C_I + S_T + K_I | 4.615 | 3.956 | +0.658 [+0.002, +1.346] | +0.143 [+0.018, +0.237] | **b better** | S_T +0.003 [+0.002, +0.005]; K_I -1.286 [-1.924, -0.666] | 0.139 / 0.295 | 1878 | both |

Sensitivity (curated): S = BE vs pronoun frame

| ID | family | outcome | a → b | MSE a | MSE b | Δ = a − b (95% CI) | Δ / MSE a (90% CI) | decision | added coefficient (95% CI) | in-sample R² a / b | cond(b) | question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I1 | primary | Δ_pass(I) | null → C | 3.483 | 2.642 | +0.841 [+0.365, +1.389]; Bonferroni 98.75% +0.841 [+0.269, +1.513] | +0.241 [+0.127, +0.350] | **b better** | C -1.080 [-1.359, -0.788] | -0.000 / 0.250 | 33 | does total exposure predict intransitive rejection? |
| I3 | primary | Δ_pass(I) | C → C + K | 2.642 | 2.433 | +0.209 [-0.072, +0.517]; Bonferroni 98.75% +0.209 [-0.145, +0.607] | +0.079 [-0.012, +0.154] | **unresolved** | K +1.018 [+0.511, +1.642] | 0.250 / 0.323 | 48 | does the competing frame add to total exposure? |
| I5 | secondary | Δ_pass(I) | Z → C | 2.346 | 2.642 | -0.296 [-0.487, -0.142] | -0.126 [-0.199, -0.064] | **a better** | C -1.080 [-1.359, -0.788] | 0.336 / 0.250 | 33 | Pile counts vs wordfreq Zipf |
| T2 | primary | Δ_pass(T) | C → C + B | 2.028 | 2.006 | +0.021 [-0.109, +0.145]; Bonferroni 98.75% +0.021 [-0.148, +0.222] | +0.011 [-0.042, +0.052] | **unresolved** | B +0.563 [+0.015, +1.198] | 0.020 / 0.045 | 45 | does the passive slot add? |
| T3 | primary | Δ_pass(T) | C → C + S | 2.028 | 1.951 | +0.076 [-0.082, +0.269]; Bonferroni 98.75% +0.076 [-0.141, +0.329] | +0.038 [-0.029, +0.098] | **unresolved** | S +0.178 [+0.057, +0.314] | 0.020 / 0.074 | 32 | does the BE/HAVE preference add? |
| L1 | secondary | L | Z_T + Z_I → C_T + C_I | 4.321 | 4.615 | -0.293 [-0.872, +0.203] | -0.068 [-0.157, +0.024] | **unresolved** | C_T +1.147 [+0.067, +2.278]; C_I -0.235 [-1.346, +0.826] | 0.167 / 0.139 | 46 | Pile vs Zipf at pair level |
| L2a | secondary | L | C_T + C_I → C_T + C_I + S_T | 4.615 | 4.596 | +0.019 [-0.296, +0.338] | +0.004 [-0.048, +0.052] | **unresolved** | S_T +0.158 [-0.024, +0.347] | 0.139 / 0.156 | 47 | does the T verb's preference add? |
| L2b | secondary | L | C_T + C_I → C_T + C_I + K_I | 4.615 | 4.400 | +0.215 [-0.326, +0.790] | +0.047 [-0.053, +0.134] | **unresolved** | K_I -1.094 [-1.840, -0.367] | 0.139 / 0.197 | 63 | does the I verb's competing frame add? |
| L2 | secondary | L | C_T + C_I → C_T + C_I + S_T + K_I | 4.615 | 4.313 | +0.302 [-0.292, +0.867] | +0.065 [-0.047, +0.151] | **unresolved** | S_T +0.198 [-0.001, +0.390]; K_I -1.187 [-1.890, -0.538] | 0.139 / 0.223 | 63 | both |

Sensitivity (curated): S = was vs has

| ID | family | outcome | a → b | MSE a | MSE b | Δ = a − b (95% CI) | Δ / MSE a (90% CI) | decision | added coefficient (95% CI) | in-sample R² a / b | cond(b) | question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I1 | primary | Δ_pass(I) | null → C | 3.483 | 2.642 | +0.841 [+0.365, +1.389]; Bonferroni 98.75% +0.841 [+0.269, +1.513] | +0.241 [+0.127, +0.350] | **b better** | C -1.080 [-1.359, -0.788] | -0.000 / 0.250 | 33 | does total exposure predict intransitive rejection? |
| I3 | primary | Δ_pass(I) | C → C + K | 2.642 | 2.433 | +0.209 [-0.072, +0.517]; Bonferroni 98.75% +0.209 [-0.145, +0.607] | +0.079 [-0.012, +0.154] | **unresolved** | K +1.018 [+0.511, +1.642] | 0.250 / 0.323 | 48 | does the competing frame add to total exposure? |
| I5 | secondary | Δ_pass(I) | Z → C | 2.346 | 2.642 | -0.296 [-0.487, -0.142] | -0.126 [-0.199, -0.064] | **a better** | C -1.080 [-1.359, -0.788] | 0.336 / 0.250 | 33 | Pile counts vs wordfreq Zipf |
| T2 | primary | Δ_pass(T) | C → C + B | 2.028 | 2.006 | +0.021 [-0.109, +0.145]; Bonferroni 98.75% +0.021 [-0.148, +0.222] | +0.011 [-0.042, +0.052] | **unresolved** | B +0.563 [+0.015, +1.198] | 0.020 / 0.045 | 45 | does the passive slot add? |
| T3 | primary | Δ_pass(T) | C → C + S | 2.028 | 1.177 | +0.851 [+0.381, +1.492]; Bonferroni 98.75% +0.851 [+0.292, +1.689] | +0.420 [+0.244, +0.542] | **b better** | S +0.653 [+0.497, +0.808] | 0.020 / 0.459 | 32 | does the BE/HAVE preference add? |
| L1 | secondary | L | Z_T + Z_I → C_T + C_I | 4.321 | 4.615 | -0.293 [-0.872, +0.203] | -0.068 [-0.157, +0.024] | **unresolved** | C_T +1.147 [+0.067, +2.278]; C_I -0.235 [-1.346, +0.826] | 0.167 / 0.139 | 46 | Pile vs Zipf at pair level |
| L2a | secondary | L | C_T + C_I → C_T + C_I + S_T | 4.615 | 4.293 | +0.322 [-0.779, +1.380] | +0.070 [-0.121, +0.228] | **unresolved** | S_T +0.552 [+0.260, +0.847] | 0.139 / 0.259 | 47 | does the T verb's preference add? |
| L2b | secondary | L | C_T + C_I → C_T + C_I + K_I | 4.615 | 4.400 | +0.215 [-0.326, +0.790] | +0.047 [-0.053, +0.134] | **unresolved** | K_I -1.094 [-1.840, -0.367] | 0.139 / 0.197 | 63 | does the I verb's competing frame add? |
| L2 | secondary | L | C_T + C_I → C_T + C_I + S_T + K_I | 4.615 | 3.927 | +0.688 [-0.479, +1.749] | +0.149 [-0.055, +0.301] | **unresolved** | S_T +0.579 [+0.298, +0.872]; K_I -1.199 [-1.842, -0.578] | 0.139 / 0.329 | 64 | both |

Sensitivity (curated): smoothing 0.1

| ID | family | outcome | a → b | MSE a | MSE b | Δ = a − b (95% CI) | Δ / MSE a (90% CI) | decision | added coefficient (95% CI) | in-sample R² a / b | cond(b) | question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I1 | primary | Δ_pass(I) | null → C | 3.483 | 2.642 | +0.841 [+0.365, +1.389]; Bonferroni 98.75% +0.841 [+0.269, +1.513] | +0.241 [+0.127, +0.350] | **b better** | C -1.080 [-1.359, -0.788] | -0.000 / 0.250 | 33 | does total exposure predict intransitive rejection? |
| I3 | primary | Δ_pass(I) | C → C + K | 2.642 | 2.434 | +0.208 [-0.072, +0.516]; Bonferroni 98.75% +0.208 [-0.145, +0.605] | +0.079 [-0.012, +0.154] | **unresolved** | K +1.014 [+0.508, +1.638] | 0.250 / 0.323 | 48 | does the competing frame add to total exposure? |
| I5 | secondary | Δ_pass(I) | Z → C | 2.346 | 2.642 | -0.296 [-0.487, -0.142] | -0.126 [-0.199, -0.064] | **a better** | C -1.080 [-1.359, -0.788] | 0.336 / 0.250 | 33 | Pile counts vs wordfreq Zipf |
| T2 | primary | Δ_pass(T) | C → C + B | 2.028 | 2.006 | +0.021 [-0.109, +0.145]; Bonferroni 98.75% +0.021 [-0.148, +0.222] | +0.011 [-0.042, +0.052] | **unresolved** | B +0.564 [+0.015, +1.198] | 0.020 / 0.045 | 45 | does the passive slot add? |
| T3 | primary | Δ_pass(T) | C → C + S | 2.028 | 1.665 | +0.362 [+0.013, +0.811]; Bonferroni 98.75% +0.362 [-0.087, +0.966] | +0.179 [+0.028, +0.302] | **b better** | S +0.547 [+0.314, +0.776] | 0.020 / 0.233 | 32 | does the BE/HAVE preference add? |
| L1 | secondary | L | Z_T + Z_I → C_T + C_I | 4.321 | 4.615 | -0.293 [-0.872, +0.203] | -0.068 [-0.157, +0.024] | **unresolved** | C_T +1.147 [+0.067, +2.278]; C_I -0.235 [-1.346, +0.826] | 0.167 / 0.139 | 46 | Pile vs Zipf at pair level |
| L2a | secondary | L | C_T + C_I → C_T + C_I + S_T | 4.615 | 4.516 | +0.099 [-0.477, +0.648] | +0.021 [-0.077, +0.116] | **unresolved** | S_T +0.428 [+0.130, +0.727] | 0.139 / 0.191 | 46 | does the T verb's preference add? |
| L2b | secondary | L | C_T + C_I → C_T + C_I + K_I | 4.615 | 4.401 | +0.214 [-0.325, +0.788] | +0.046 [-0.053, +0.134] | **unresolved** | K_I -1.089 [-1.832, -0.366] | 0.139 / 0.197 | 63 | does the I verb's competing frame add? |
| L2 | secondary | L | C_T + C_I → C_T + C_I + S_T + K_I | 4.615 | 4.124 | +0.491 [-0.277, +1.232] | +0.106 [-0.037, +0.212] | **unresolved** | S_T +0.528 [+0.226, +0.834]; K_I -1.324 [-1.982, -0.704] | 0.139 / 0.274 | 63 | both |

Sensitivity (curated): smoothing 1.0

| ID | family | outcome | a → b | MSE a | MSE b | Δ = a − b (95% CI) | Δ / MSE a (90% CI) | decision | added coefficient (95% CI) | in-sample R² a / b | cond(b) | question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I1 | primary | Δ_pass(I) | null → C | 3.483 | 2.642 | +0.841 [+0.365, +1.389]; Bonferroni 98.75% +0.841 [+0.269, +1.513] | +0.241 [+0.127, +0.350] | **b better** | C -1.080 [-1.359, -0.788] | -0.000 / 0.250 | 33 | does total exposure predict intransitive rejection? |
| I3 | primary | Δ_pass(I) | C → C + K | 2.642 | 2.433 | +0.209 [-0.072, +0.518]; Bonferroni 98.75% +0.209 [-0.145, +0.608] | +0.079 [-0.012, +0.154] | **unresolved** | K +1.022 [+0.514, +1.648] | 0.250 / 0.324 | 48 | does the competing frame add to total exposure? |
| I5 | secondary | Δ_pass(I) | Z → C | 2.346 | 2.642 | -0.296 [-0.487, -0.142] | -0.126 [-0.199, -0.064] | **a better** | C -1.080 [-1.359, -0.788] | 0.336 / 0.250 | 33 | Pile counts vs wordfreq Zipf |
| T2 | primary | Δ_pass(T) | C → C + B | 2.028 | 2.006 | +0.021 [-0.109, +0.145]; Bonferroni 98.75% +0.021 [-0.148, +0.222] | +0.011 [-0.042, +0.052] | **unresolved** | B +0.563 [+0.015, +1.198] | 0.020 / 0.045 | 45 | does the passive slot add? |
| T3 | primary | Δ_pass(T) | C → C + S | 2.028 | 1.665 | +0.362 [+0.013, +0.811]; Bonferroni 98.75% +0.362 [-0.087, +0.966] | +0.179 [+0.028, +0.301] | **b better** | S +0.547 [+0.315, +0.777] | 0.020 / 0.233 | 32 | does the BE/HAVE preference add? |
| L1 | secondary | L | Z_T + Z_I → C_T + C_I | 4.321 | 4.615 | -0.293 [-0.872, +0.203] | -0.068 [-0.157, +0.024] | **unresolved** | C_T +1.147 [+0.067, +2.278]; C_I -0.235 [-1.346, +0.826] | 0.167 / 0.139 | 46 | Pile vs Zipf at pair level |
| L2a | secondary | L | C_T + C_I → C_T + C_I + S_T | 4.615 | 4.516 | +0.099 [-0.477, +0.648] | +0.021 [-0.077, +0.116] | **unresolved** | S_T +0.428 [+0.130, +0.728] | 0.139 / 0.191 | 46 | does the T verb's preference add? |
| L2b | secondary | L | C_T + C_I → C_T + C_I + K_I | 4.615 | 4.399 | +0.216 [-0.326, +0.792] | +0.047 [-0.053, +0.134] | **unresolved** | K_I -1.100 [-1.848, -0.369] | 0.139 / 0.197 | 63 | does the I verb's competing frame add? |
| L2 | secondary | L | C_T + C_I → C_T + C_I + S_T + K_I | 4.615 | 4.122 | +0.492 [-0.279, +1.236] | +0.107 [-0.037, +0.213] | **unresolved** | S_T +0.528 [+0.226, +0.834]; K_I -1.334 [-1.996, -0.708] | 0.139 / 0.274 | 63 | both |

- T3 decomposition (curated): C → C + B + P: Δ +0.391 [-0.212, +0.935] → unresolved; B coefficient +0.851 [+0.359, +1.427], P coefficient -1.650 [-2.541, -0.832] (a preference effect from more BE has B > 0; from less HAVE, P < 0).

## Re-binning by Pile counts (D11)

Within each class, verbs ranked by c_total (descending; ties alphabetical) and cut into bins of the band sizes (top 47 = bin 1, next 132 = bin 2, lowest 102 = bin 3); pairs by mean log c_total the same way. Means with 95% CIs (pairs within band × contexts). Rejection gradient in −Δ_pass(I) units.

| outcome | grouping | 1 (Head / top) | 2 (Tail / middle) | 3 (XTail / bottom) | 1 − 3 |
|---|---|---|---|---|---|
| Δ_pass(T) | Zipf band | +0.48 [-0.07, +1.00] | +0.33 [-0.06, +0.78] | +0.84 [+0.44, +1.28] | -0.36 [-0.84, +0.12] |
| Δ_pass(T) | count bin | +0.24 [-0.29, +0.73] | +0.43 [+0.03, +0.86] | +0.82 [+0.41, +1.28] | -0.59 [-1.07, -0.13] |
| Δ_pass(I) | Zipf band | -2.89 [-3.45, -2.32] | -2.23 [-2.66, -1.79] | -1.12 [-1.60, -0.66] | -1.77 [-2.37, -1.17] |
| Δ_pass(I) | count bin | -2.77 [-3.33, -2.24] | -2.16 [-2.57, -1.74] | -1.28 [-1.74, -0.76] | -1.49 [-2.10, -0.90] |
| L | Zipf band | +3.37 [+2.75, +4.00] | +2.56 [+2.21, +2.92] | +1.96 [+1.55, +2.40] | +1.41 [+0.67, +2.12] |
| L | count bin | +3.21 [+2.59, +3.85] | +2.56 [+2.20, +2.93] | +2.04 [+1.63, +2.46] | +1.18 [+0.42, +1.89] |

Intransitive rejection gradient (top − bottom, −Δ_pass(I)): count bins +1.49 [+0.90, +2.10], Zipf bands +1.77 [+1.17, +2.37]; difference -0.27 [-0.56, +0.05].

Pairs, Zipf band × count bin: Head → bin1 43, bin2 4, bin3 0; Tail → bin1 4, bin2 120, bin3 8; XTail → bin1 0, bin2 8, bin3 94

B1 (CV, secondary): Zipf-band dummies → count-bin dummies:

- Δ_pass(T): MSE 1.694 → 1.698, Δ -0.005 [-0.070, +0.058] → **unresolved**.
- Δ_pass(I): MSE 2.669 → 2.803, Δ -0.134 [-0.319, +0.013] → **unresolved**.
- L: MSE 4.123 → 4.196, Δ -0.073 [-0.193, +0.016] → **unresolved**.

## Diagnosis: curated intransitives (round 5: Zipf beat Pile log c)

- D1, c_total: Pile MSE 2.642 vs Zipf 2.346; Zipf advantage MSE_C − MSE_Z +0.296 [+0.142, +0.487] → a better (a = Zipf, b = Pile).
- D1, marked only: Pile MSE 2.640 vs Zipf 2.346; Zipf advantage MSE_C − MSE_Z +0.293 [+0.139, +0.484] → a better (a = Zipf, b = Pile).
- D1, upper bound: Pile MSE 2.703 vs Zipf 2.346; Zipf advantage MSE_C − MSE_Z +0.357 [+0.186, +0.560] → a better (a = Zipf, b = Pile).
- D1, round-5 lowercase only: Pile MSE 2.639 vs Zipf 2.346; Zipf advantage MSE_C − MSE_Z +0.293 [+0.138, +0.484] → a better (a = Zipf, b = Pile).
- Zipf advantage with c_total relative to round 5's count: 1.01 (prediction: ≤ 0.5 and ≥ 0).

- D2: residual of log10 c_total on Zipf over the 608 forms (slope 0.92); the 10 curated I verbs with the largest positive and negative residuals:

| direction | verb | band | Zipf | log10 c_total | residual | Δ_pass(I) | CV pred. Zipf | CV pred. Pile | closer |
|---|---|---|---|---|---|---|---|---|---|
| Pile > Zipf | testified | Head | 3.82 | 6.90 | +0.75 | -1.72 | -3.50 | -4.13 | Zipf |
| Pile > Zipf | grinned | Tail | 2.87 | 6.01 | +0.73 | -0.75 | -2.33 | -3.02 | Zipf |
| Pile > Zipf | glared | Tail | 2.59 | 5.67 | +0.65 | -0.57 | -1.96 | -2.62 | Zipf |
| Pile > Zipf | quivered | XTail | 1.96 | 5.07 | +0.62 | -1.08 | -1.21 | -2.02 | Zipf |
| Pile > Zipf | shuddered | Tail | 2.47 | 5.54 | +0.62 | -1.46 | -1.79 | -2.47 | Zipf |
| Pile > Zipf | panted | XTail | 1.93 | 5.01 | +0.59 | +0.96 | -1.30 | -1.99 | Zipf |
| Pile > Zipf | gaped | XTail | 2.03 | 5.06 | +0.55 | +0.34 | -1.27 | -1.95 | Zipf |
| Pile > Zipf | shimmered | XTail | 1.89 | 4.89 | +0.51 | -2.09 | -1.10 | -1.78 | Pile |
| Pile > Zipf | tottered | XTail | 1.59 | 4.58 | +0.48 | -1.57 | -0.77 | -1.47 | Pile |
| Pile > Zipf | jutted | XTail | 1.80 | 4.77 | +0.47 | -3.11 | -1.13 | -1.75 | Pile |
| Pile < Zipf | peed | Tail | 3.10 | 4.80 | -0.70 | -1.63 | -2.65 | -1.78 | Pile |
| Pile < Zipf | scrammed | XTail | 1.23 | 3.24 | -0.53 | +0.07 | -0.33 | +0.06 | Pile |
| Pile < Zipf | lied | Head | 4.21 | 6.09 | -0.42 | -2.31 | -3.99 | -3.20 | Pile |
| Pile < Zipf | eventuated | XTail | 1.93 | 4.03 | -0.39 | -1.63 | -1.17 | -0.82 | Zipf |
| Pile < Zipf | quacked | XTail | 1.39 | 3.55 | -0.37 | +0.56 | -0.64 | -0.47 | Pile |
| Pile < Zipf | hitchhiked | XTail | 2.12 | 4.23 | -0.36 | -2.34 | -1.40 | -1.05 | Zipf |
| Pile < Zipf | drooled | XTail | 2.15 | 4.27 | -0.35 | -2.36 | -1.42 | -1.09 | Zipf |
| Pile < Zipf | colluded | Tail | 2.74 | 4.85 | -0.32 | -2.69 | -2.14 | -1.75 | Zipf |
| Pile < Zipf | stagnated | Tail | 2.65 | 4.78 | -0.30 | -2.91 | -2.00 | -1.64 | Zipf |
| Pile < Zipf | flirted | Tail | 2.84 | 4.96 | -0.30 | -4.42 | -2.30 | -1.90 | Zipf |

Mean squared error on these 20 verbs: Zipf 1.70, Pile 2.38; on the other 106: Zipf 2.47, Pile 2.69.

- D3: Pile subsets and hand-labelled uses of sampled occurrences (variants in proportion to their counts):

| group | verbs | occurrences | top Pile subsets (share) | verbal | adjectival | nominal / name | other |
|---|---:|---:|---|---|---|---|---|
| Pile < Zipf | 10 | 200 | Books3 0.29, Pile-CC 0.28, OpenWebText2 0.20, FreeLaw 0.06 | 0.95 | 0.03 | 0.01 | 0.02 |
| Pile > Zipf | 10 | 200 | Books3 0.64, Pile-CC 0.12, Gutenberg (PG-19) 0.09, FreeLaw 0.07 | 0.97 | 0.01 | 0.00 | 0.01 |
| control | 10 | 200 | Books3 0.40, Pile-CC 0.20, OpenWebText2 0.13, PubMed Central 0.07 | 0.93 | 0.07 | 0.00 | 0.01 |

- Precision of the broad pronoun proxy (share labelled finite active): competed (I) 1.00 of 20, debugged (T) 0.40 of 15, glared (I) 1.00 of 20, puttered (I) 1.00 of 20, robbed (T) 1.00 of 20, stowed (T) 0.50 of 20.

