# C1 trained-direction controls and C4 phase 1b (where the effect goes, natural mediation, the core neurons): rules (declared Nov 26, before any run)

**Why.**
- **C1 controls.** C1 compares v2 only with random directions, and only in training-time units (pc_frac ≈ 0). v2's own
  finals hold a shuffled-label direction per fold. It recovers 0.45 / 0.55 / 0.47 of held-out pc_frac at the first
  rule site of Pythia / Gemma / OLMo, against v2's 0.80 / 0.99 / 0.81. The paper needs both controls in its own unit
  (D / G on the 89 eval plain pairs), on the same plan and draws as v2.
- **C4 phase 1b.** Phase 1 (`reports/pythia/mechanism_v2_rules.md`, read in `reports/pythia/v2_mech/readings.md`)
  leaves three gaps:
  - about three quarters of the edit's effect on *by* is not located (P 0.19, all of MLPs 11 + 14 0.26);
  - every mediated share is a share of the edit's effect, not of the model's own T − I gap;
  - the neurons have natural-text descriptions (R5-B7), but no test ties a description to the neuron's causal role
    or its response in other constructions.

Phase 1's sets, keys, halves and instruments are reused unchanged. No new selection is made anywhere in this file.

## Part A. Trained-direction controls for C1 (Pythia, Gemma, OLMo)

**Instruments**, each a rank-1 interchange with its own unit direction per (split, fold), at the first two rule sites
of each model's C1 table (Pythia 8, 12; Gemma 5, 8; OLMo 9, 11):
- **Shuffled-label direction** (`shuffled_bases.pt` of the same v2 final). It was trained with the class labels
  permuted across v2's training verbs (`run_das_v2.py`; generator seed 100 · split + fold), with the same objective,
  pool and epochs as v2.
- **Random directions:** 3 draws per (split, fold) key, standard normal then normalized, at the first rule site only.
  The generator is seeded with 910001 + 100 · draw + 10 · split + fold, a stream no stored run uses.

**Run.** `run_passive_test.py` with the same prompts, plan, folds and settings as v2's `passive_rule` run. Only
`--site-dirs` changes: each points to a directory holding the control's `bases.pt` (keys as v2's) and v2's
`pairs_folds.csv`. Checks:
- natural readouts equal v's stored expanded run (`natcheck`, ≤ 1e-6);
- the runner's self-patch check passes;
- the fold assignment check passes;
- the control's keys and shapes equal v2's.

**Analysis.** `analyze_c1_controls.py`, on the same populations and bootstrap draws (seed and n_boot as
`analyze_v2_expanded.py`) and the same `d_bad_all` / `summarize` code. Per site and readout (*by*, O), on eval plain
(89): D, D / G, and the paired difference v2 − control on the same draws, as nats and as control / v2. Eval new
plain and lemma-fresh pairs are descriptive.

**Declared (per model, at each first rule site, readout *by*):**
- **A1: v2 exceeds its shuffled-label direction.** lo95(D_v2 − D_shuf) > 0.
- **A2: random directions do not move *by*.** For each draw, the 95% CI of D lies within ±0.1 · G. If a draw's CI
  falls outside, that draw is reported.

The second rule site and the O readout are reported, not declared.

**Similarity to v2 (descriptive):** per (split, fold), |cos| between the shuffled-label direction and v2's direction
(random rank-1 directions in 2,048 dimensions give about 0.02). The text reports the mean and the range. A round-2
note on v read the shuffled direction as "a weaker copy of d", because a permutation leaves many labels correct. This
file reports the cosine and does not test that account.

**Reading for the paper.** The shuffled-label direction is a trained-on-noise floor. The objective still rewards a
direction along which an interchange moves the output's T / I preference, so the floor is not zero.
- A1 holds: "a direction trained with shuffled labels moves *by* by X of the gap; v2 moves it by Y, Z more on the same
  draws".
- A1 fails at a site: that site's v2 transfer is not distinguished from the trained-on-noise floor, and C1 says so
  there.
- In either case, "not an artefact of DAS training" rests on u_md (training-free) and on A1 together.

## Part B. Pythia C4 phase 1b

Rows: phase 1's mediation keys (half B, 46 pairs, 8,414 keys). The runners rebuild them and assert equality with
`results/pythia/das_v2/mech/med_v2/mediation_keys.parquet`. Readout: log P(" by") at the participle's last token,
in fp32. Every share is pair-averaged, then a ratio of averages. Pair bootstrap: 2,000 draws, seed 17. Controls are
the mean of their sets within each draw, as in phase 1.

### B1. Where the edit's effect goes: component sweep (v2)

For each component at the participle's last token, clamp its output to its value in the other arm's run of the same
key. This uses phase 1's T ← I and I ← T definitions, applied to component outputs instead of neuron
post-activations. Components:
- attention output (`attention.dense` output) for each of layers 8–23;
- MLP output for each of layers 8–23;
- groups: all attention 8–23; all MLP 8–23; all components 8–23; MLPs 11 + 14.

The site-8 edit enters the residual stream before block 8 (output of block 7). "All components" therefore leaves only
the direct residual path from the edit to the readout.

**Check:** the MLPs 11 + 14 output clamp must reproduce phase 1's ALL-neuron clamp, max |Δ log P| ≤ 1e-4 per key, in
both directions. Otherwise B1 is invalid.

**Reported (descriptive, no declared test):**
- m and r for every component and group, with CIs;
- the direct-path share, 1 − m(all components), next to the sum of single-component shares (they need not add up:
  clamps interact);
- the rank of MLP 11, MLP 14 and P among the single components.

The figure shows where the effect goes. P is drawn as a labelled slice of MLPs 11 + 14.

### B2. Natural mediation: do the same neurons carry the model's own T − I gap?

Items: the half-B pairs' natural passives "The N was V", with the same 32 contexts per pair as phase 1 (same rng draw).
The good (T verb) and bad (I verb) prompts of each item share the context. G = Y[good] − Y[bad].
- **B_nat(S) = Y[good] − Y[good ← bad(S)]:** the gap removed by clamping S, at the participle's last token, to the bad
  prompt's values. m_nat = B_nat / G.
- **R_nat(S) = Y[bad ← good(S)] − Y[bad]:** the gap restored by the converse. r_nat = R_nat / G.

Sets: phase 1's frozen `mediation_sets.npz` (P, ALL, P_mlp11 / 14, top-k, rand / mag / de × 20), plus the core sets of
B3.

**Caveat.** The clamp moves the other verb's values, so it carries lexical identity along with class. The matched
controls take the generic part, and the text states this.

**Declared:**
- **Gate:** G lo95 > 0.
- **NM1 (P carries part of the natural gap):** lo95 m_nat(P) > 0 and lo95(m_nat(P) − mean m_nat(mag)) > 0.
- **NM2:** the same for r_nat.
- **Beyond de:** stated separately, as in phase 1.
- Reported beside them: m_nat(ALL), P / ALL where lo95 m_nat(ALL) > 0, and the descriptive new / lemma-fresh subsets.

**Outcomes:**
- **NM1 holds:** "the same neurons carry X of the model's own T − I gap on held-out verbs (Y for activation-matched
  neurons)", next to the edit's 0.19.
- **NM1 fails:** "the edit is read through these neurons; the model's own contrast is not detectably carried by them
  under this clamp", stated as a limit in §5. B5's ablation result (0.09 of the natural gap, on v's set) is consistent
  with either outcome, and is cited in the appendix only.

### B3. The core neurons: what they respond to and what each carries

**Core (fixed now):** the top 5 of P by half-A direct effect on the switch, from
`results/pythia/das_v2/mech/conj_v2/conjunction_switch_neurons.csv`:
MLP11.n3462, MLP14.n5407, MLP14.n2114, MLP11.n4925, MLP14.n4241. These are v's top 5 in a different order (v's ranks
1, 3, 2, 4, 5), so the stored instrument-free Pile-10k activations of R5-B7 cover them.

**B3a. Labels from natural text.**
- Ranks 1–3 keep R5-B7's labels (two independent annotators, κ = 1.00): MLP11.n3462 *gap-sensitive participle
  reader*; MLP14.n2114 and MLP14.n5407 *passive-participle detectors*.
- MLP11.n4925 and MLP14.n4241 were not annotated. Their top 40 raw contexts from the stored `wild_tokens.parquet`
  (same extraction as B7) are pooled, shuffled and labelled independently by Claude and Codex with B7's frozen guide
  (`results/round5/wild/b7_annotation_guide.md`).
- Labels follow B7's rule exactly (`analyze_neurons_wild.py decide`), for both annotators:
  - verbal passive ≥ 0.5 of the top 40 → *passive-participle detector*;
  - otherwise, verbal passive + object gap ≥ 0.5 with object gap ≥ 0.15 → *gap-sensitive participle reader*;
  - otherwise *other*.
- κ and the confusion table are reported. The items come from the stored `b7_top_contexts.csv` (raw mode, k ≤ 40),
  pooled across the two neurons and shuffled (seed 17). The key stays hidden from both annotators.
- **Seen before this file:** B7's surface-proxy summary for these two neurons. MLP11.n4925's top words are
  prepositions; MLP14.n4241 has 0.82 participles and 0.82 be / get within 3 words. The proxies suggest *other* and
  *passive-participle detector*. The labels come from the annotation, not from the proxies.

**B3b. Causal role of each core neuron.** Sets, as masks on phase 1's layout:
- each core neuron alone; the top 3; the top 5;
- single-neuron controls: per layer, the 20 non-P neurons with the largest half-A |direct effect| (ties by index).
  This was checked while building the sets, before any run: no non-P neuron reaches the core's coupling. Core: MLP 11
  0.055–0.197, MLP 14 0.068–0.157. Strongest non-P neuron: MLP 11 0.058, MLP 14 0.013. So these are the strongest
  available non-conjunction comparators, not matched ones, and the text says so.

They enter the edit mediation (phase 1's runner with this sets file) and B2. Descriptive: m, r, m_nat and r_nat per
set, each core neuron against its layer's 20 controls (mean and rank).

**B3c. Declared double dissociation (natural activations, no instrument).** For each core neuron, the T − I difference
of its post-activation at the verb's last token:
- **passive:** good − bad on the half-B items of B2 ("The N was V"), recorded in B2's unclamped runs;
- **object-gap frame:** R4-C8's embedded items, "I know what John V" (fill) and "I know that John V" (nofill). The
  values are the stored natural post-activations (`results/round5/shared/sr_natural_fg.parquet`), T items minus I
  items, all 156 C8 lemmas;
- **matrix frames (descriptive):** "What did John V" and "Did John V".

Index: ρ = (fill T − I) / (passive T − I). Bootstrap: lemmas resampled for C8 and pairs for the passives,
independently, 2,000 draws, seed 17.

Declared:
- **D1 (the gap reader depends on the filler):** for MLP11.n3462, lo95(fill T − I) > 0 and lo95((fill − nofill)
  T − I) > 0.
- **D2 (dissociation):** for each passive detector (MLP14.n2114, MLP14.n5407, and any rank-4 / 5 neuron labelled
  passive detector in B3a), lo95(ρ(n3462) − ρ(detector)) > 0.
- Neurons labelled *other* are reported without a test.

**Outcomes:**
- **D1 and D2 hold:** §5 gets an interpretable core. One neuron reads a transitive verb whose object is missing,
  whether the gap comes from passive voice or from a filler. Others detect the passive participle itself. Each is
  shown with its share of the edit's effect and of the natural gap.
- **D2 fails:** the core neurons are described by their natural-text labels only, and the text does not claim they
  divide the work.

**Disclosed:** C9's set-level results on these items were seen (Oct 14), including that the top 10 respond more in the
"what" frame than in passives (ρ_nat 2.2). Per-neuron frame profiles were not tabulated then and are first computed
here. The B7 labels of ranks 1–3 were also seen.

## Checks (a stage whose check fails is reported as invalid, not read)

- **A:** natural readouts; self-patch; folds; control keys and shapes equal v2's.
- **B:** keys equal phase 1's; empty-set clamps equal their references (≤ 1e-5); hook checks as in phase 1; B1's
  MLPs 11 + 14 vs ALL (≤ 1e-4); B2's G recomputed from the unclamped runs matches the stored natural passive readouts
  for the same prompts (`results/pythia/das_v2/expanded_eval/passive_rule/natural.parquet`, joined by prompt; median
  |Δ| ≤ 0.05 for the bf16 reference).
- **B3:** the core neuron indices equal the switch table's; the C8 columns are mapped by name through `sr_meta.json`'s
  `neuron_columns`.

## Forking paths

- Part A's sites and readouts are C1's, fixed before this file.
- B1 is descriptive by design. Its component list is the full set of blocks after the edit.
- B2 reuses phase 1's sets and rules unchanged.
- The core is the exploratory top 5 from phase 1, fixed here before any B3 run. The B3c predictions follow from the B7
  labels, which were set in October on v's ranking.
- Not run: heads (phase 2), a second model, a serial head → neuron → *by* test.
