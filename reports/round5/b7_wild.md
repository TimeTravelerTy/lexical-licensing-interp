# Round 5, B7: switch neurons on natural text (Pile-10k)

Spec: `plan.md`, B7. Run: `run_neurons_wild.py` (seed 17; 2000 documents × ≤ 512 tokens = 722,625 tokens; fp32 post-activations). Analysis: `analyze_neurons_wild.py describe` (and `decide` for the annotation). Proxies are surface patterns, not grammatical classes: (a) the token ends a word in -ed/-en or one of A1's 962 participles; (b) a be/get-form among the 3 preceding words; (c) a wh-word or relativizer among the 8 preceding words and no be/get-form among the 3; (d) the next token is " by" (id 407). (a)–(c) are defined on word-final tokens (the run saved preceding words there only); other tokens fail them. CIs: 1,000 document resamples with the top 100 recomputed in each.

## Activation distributions

| rank | neuron | s_n | mean | SD | share > 0 | p99 | p99.9 | max | min |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | MLP11.n3462 | +1 | -0.037 | 0.126 | 0.012 | 0.08 | 2.02 | 4.59 | -0.17 |
| 2 | MLP14.n2114 | +1 | -0.080 | 0.135 | 0.032 | 0.38 | 1.79 | 4.37 | -0.17 |
| 3 | MLP14.n5407 | +1 | -0.078 | 0.142 | 0.034 | 0.39 | 1.84 | 5.59 | -0.17 |
| 4 | MLP11.n4925 | +1 | -0.042 | 0.083 | 0.009 | -0.00 | 0.98 | 4.34 | -0.17 |
| 5 | MLP14.n4241 | +1 | -0.074 | 0.217 | 0.085 | 1.00 | 2.37 | 3.96 | -0.17 |
| 6 | MLP14.n496 | +1 | -0.084 | 0.159 | 0.066 | 0.61 | 1.78 | 4.40 | -0.17 |
| 7 | MLP11.n1686 | +1 | -0.084 | 0.146 | 0.048 | 0.58 | 1.67 | 3.81 | -0.17 |
| 8 | MLP14.n3517 | +1 | -0.067 | 0.132 | 0.030 | 0.38 | 1.75 | 3.90 | -0.17 |
| 9 | MLP11.n5408 | -1 | -0.051 | 0.098 | 0.008 | -0.00 | 1.21 | 5.74 | -0.17 |
| 10 | MLP14.n503 | -1 | -0.049 | 0.189 | 0.171 | 0.79 | 1.38 | 2.54 | -0.17 |

**Baselines** (share of tokens meeting each proxy):

| baseline | n | (a) participle | (b) be/get in 3 | (c) wh/rel in 8 | (d) " by" next |
|---|---:|---:|---:|---:|---:|
| all tokens | 722,625 | 0.031 | 0.063 | 0.073 | 0.003 |
| word-final tokens | 465,649 | 0.048 | 0.097 | 0.113 | 0.004 |
| participle candidates (a) | 22,119 | 1.000 | 0.263 | 0.109 | 0.048 |

## Top 100 contexts by raw activation

Each cell: share of the top 100 [95% CI]. Docs / words = unique documents / unique word forms in the top 100.

| rank | neuron | docs | words | (a) participle | (b) be/get in 3 | (c) wh/rel in 8 | (d) " by" next | most frequent words |
|---:|---|---:|---:|---|---|---|---|---|
| 1 | MLP11.n3462 | 94 | 81 | 0.24 [0.15, 0.32] | 0.10 [0.04, 0.16] | 0.18 [0.10, 0.26] | 0.09 [0.03, 0.14] | use (6), used (3), to (3), see (3), want (2), had (2) |
| 2 | MLP14.n2114 | 88 | 73 | 0.76 [0.68, 0.85] | 0.45 [0.34, 0.55] | 0.07 [0.02, 0.12] | 0.53 [0.42, 0.62] | used (10), provided (6), told (4), instructed (3), approved (2), use (2) |
| 3 | MLP14.n5407 | 95 | 44 | 0.89 [0.82, 0.95] | 0.53 [0.43, 0.62] | 0.01 [0.00, 0.04] | 0.78 [0.68, 0.86] | affected (12), followed (12), inspired (7), accompanied (6), hit (5), influenced (4) |
| 4 | MLP11.n4925 | 85 | 68 | 0.04 [0.01, 0.08] | 0.20 [0.12, 0.28] | 0.21 [0.13, 0.29] | 0.01 [0.00, 0.03] | in (9), on (8), with (5), into (5), to (3), about (3) |
| 5 | MLP14.n4241 | 82 | 73 | 0.82 [0.75, 0.89] | 0.82 [0.73, 0.89] | 0.04 [0.01, 0.08] | 0.11 [0.05, 0.17] | reviewed (7), described (4), executed (4), accessed (3), evaluated (3), characterized (3) |
| 6 | MLP14.n496 | 71 | 65 | 0.47 [0.37, 0.58] | 0.27 [0.16, 0.39] | 0.05 [0.01, 0.10] | 0.52 [0.41, 0.61] | the (13), generated (4), opened (4), manufactured (3), designed (3), produced (3) |
| 7 | MLP11.n1686 | 91 | 79 | 0.51 [0.40, 0.61] | 0.26 [0.18, 0.36] | 0.16 [0.09, 0.25] | 0.21 [0.14, 0.30] | ended (5), led (4), filled (3), run (3), applied (3), read (2) |
| 8 | MLP14.n3517 | 72 | 74 | 0.31 [0.20, 0.43] | 0.23 [0.13, 0.35] | 0.10 [0.03, 0.18] | 0.00 [0.00, 0.00] | charged (6), asked (6), them (4), tte (3), die (3), me (2) |
| 9 | MLP11.n5408 | 84 | 83 | 0.24 [0.15, 0.34] | 0.34 [0.25, 0.44] | 0.22 [0.15, 0.30] | 0.00 [0.00, 0.02] | work (9), made (2), deal (2), out (2), them (2), working (2) |
| 10 | MLP14.n503 | 84 | 61 | 0.16 [0.09, 0.23] | 0.28 [0.19, 0.40] | 0.16 [0.08, 0.23] | 0.01 [0.00, 0.03] | to (10), be (4), you (4), are (3), is (3), in (3) |

**Sensitivities.** Signed ranking s_n · a (identical to raw where s_n = +1; for ranks 9–10 it is the *lowest*-activation contexts) and one context per document (each document's maximum, top 100):

| rank | version | docs | (a) participle | (b) be/get in 3 | (c) wh/rel in 8 | (d) " by" next | most frequent words |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | one per document | 100 | 0.26 | 0.09 | 0.17 | 0.09 | use (6), used (4), see (3), to (3), had (2) |
| 2 | one per document | 100 | 0.74 | 0.45 | 0.07 | 0.47 | used (10), provided (6), told (4), instructed (3), accused (2) |
| 3 | one per document | 100 | 0.90 | 0.51 | 0.02 | 0.78 | affected (13), followed (13), accompanied (6), inspired (5), dominated (4) |
| 4 | one per document | 100 | 0.03 | 0.18 | 0.22 | 0.01 | on (10), in (9), with (6), to (3), about (3) |
| 5 | one per document | 100 | 0.81 | 0.81 | 0.05 | 0.13 | reviewed (7), read (4), investigated (4), executed (4), evaluated (3) |
| 6 | one per document | 100 | 0.47 | 0.24 | 0.05 | 0.47 | the (14), or (8), created (4), generated (4), written (3) |
| 7 | one per document | 100 | 0.50 | 0.24 | 0.18 | 0.21 | ended (5), led (4), application (3), run (3), applied (3) |
| 8 | one per document | 100 | 0.37 | 0.31 | 0.11 | 0.02 | asked (8), charged (6), them (5), me (3), provided (2) |
| 9 | signed (lowest activation) | 97 | 0.10 | 0.13 | 0.04 | 0.00 | «» (6), «.» (4), the (4), «-» (3), a (2) |
| 9 | one per document | 100 | 0.22 | 0.32 | 0.22 | 0.01 | work (10), out (2), them (2), working (2), made (2) |
| 10 | signed (lowest activation) | 99 | 0.03 | 0.04 | 0.05 | 0.00 | the (9), «,» (5), a (4), «.» (3), to (3) |
| 10 | one per document | 100 | 0.17 | 0.28 | 0.14 | 0.01 | to (10), be (5), as (3), you (3), in (3) |

## Example contexts (top 8 by raw activation, ranks 1–3)

**Rank 1, MLP11.n3462**

- 4.59: …law, “because they have shown respect for the law by not immigrating illegally.” If there’s no law to[[ respect]] for immigration, how…
- 4.27: …the mobile performance gap that’s most widely lamented is cart abandonment. While the 70.6% cart abandonment rate[[ achieved]] via computer browsers isn…
- 4.09: …symlink.) ⏎ delete_bin_copy = off ⏎  ⏎ # Specify the name of the machinery module to[[ use]], this module will…
- 4.09: …immigrants as a group increase the nation’s fiscal deficit. In other words, the government benefits they receive exceed the taxes[[ paid]]. ⏎  ⏎ The…
- 3.96: …GetCapabilities data ⏎             var capability = response.capability; ⏎  ⏎             // I want a list of names to[[ use]] in my queries ⏎…
- 3.95: …mom let her wear a new pink dress and boots to school. It was the last outfit the redheaded girl would ever[[ pick]] out. "She…
- 3.89: …natural environment, which has a great effect on cell signaling and behavior. I have focused on the physical behaviors NCCs[[ use]] to carry out EMT…
- 3.85: …range of advice and experience about building your startup that is anything but a rehash of all the other advice you’ve[[ heard]]. ⏎  ⏎ Also…

**Rank 2, MLP14.n2114**

- 4.37: …measures and electromyographic, kinematic and oxygen expenditure analysis during gait and cycling. The treatment was safe, feasible and well[[ accepted]] by the 2 children…
- 3.81: …ized in the judgment of the synthesis tools) is generated for a designer in response to each set of behavioral specification and constraints[[ provided]]. (The behavioral…
- 3.78: …oversight and additional levels of protection”. Training, he said, is now focused on best practice, and advanced technology, when[[ approved]], is used at…
- 3.76: …right-wing media universe, this kind of talk probably puzzled you. But make no mistake: The Ukraine fantasies [[peddled]] by House Republicans are…
- 3.72: ….S. and South Korean sweatshop zone being a side project and distraction. If this mining plan is given a green[[ light]] while Haiti is under…
- 3.72: …Palestine Chronicle Staff ⏎  ⏎ The politically-symbolic act of two young American athletes has been dubbed ‘antisemitic[[’]] by Israel’s…
- 3.68: …evening was an experience in itself, however before the night was over Mosslands was to reach the top step. Generously[[ sponsored]] by All About STEM…
- 3.61: …information to access the third-party service on the user device. Once the authenticating information is complete, submitted, and[[ processed]], the customer can…

**Rank 3, MLP14.n5407**

- 5.59: …Design Development department’s focus was to lend the Pur Sport a confident appearance. As a result, the front end is[[ dominated]] by an intentionally dynamic…
- 5.41: …oscopy showing a violaceous, homogeneous, structureless and patternless lesion ⏎  ⏎ Histopathology revealed a proliferation of blood vessels[[ surrounded]] by small, uniform…
- 5.27: …the coming months. ⏎  ⏎ Still, any information about Intel's graphics plans is welcome. Right now the graphics market is[[ dominated]] by AMD and N…
- 4.97: …possible ways to explain it. Because we sometimes don’t articulate a clear strategy because we’re outnumbered and [[overrun]] with crises or immediate…
- 4.77: …A review of the subject is presented, emphasizing aspects that must be present at the time of a patient with this condition,[[ accompanied]] by one of our…
- 4.69: …to the Victorian era, when there was a boom in new developments, along with the Georgian period. The area is[[ dominated]] by striking period buildings…
- 4.59: …ing to his own disturbed world. He started having daydreams about raping and stabbing women. In 1972,[[ inspired]] by the Alfred Hitch…
- 4.54: …which it had acquired two years earlier, in 2005, in the form of Android, Inc. a Palo Alto startup[[ headed]] by Andy Rubin…

Annotation items: 120 (top 40 raw contexts of ranks 1–3, pooled and shuffled; the word holding the token is marked [[…]]; `b7_annotation_items.csv`).

