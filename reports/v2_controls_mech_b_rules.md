# C1 trained-direction controls and C4 phase 1b (where the effect goes, natural mediation, the core neurons): rules (declared Nov 26, before any run; revised the same day after Codex review, still before any full run)

**Why.**
- **C1 controls.** C1 compares v2 only with random directions, and only in training-time units (pc_frac ≈ 0). v2's own
  finals hold a shuffled-label direction per fold. It recovers 0.45 / 0.55 / 0.47 of held-out pc_frac at the first
  rule site of Pythia / Gemma / OLMo, against v2's 0.80 / 0.99 / 0.81. The paper needs both controls in its own unit
  (D / G on the 89 eval plain pairs), on the same plan as v2.
- **C4 phase 1b.** Phase 1 (`reports/pythia/mechanism_v2_rules.md`, read in `reports/pythia/v2_mech/readings.md`)
  leaves three gaps:
  - about three quarters of the edit's effect on *by* is not located (P 0.19, all of MLPs 11 + 14 0.26);
  - every mediated share is a share of the edit's effect, not of the model's own T − I gap;
  - the neurons have natural-text descriptions (R5-B7), but nothing ties a description to the neuron's causal role or
    its response in other constructions.

Phase 1's P, keys, halves and instruments are reused unchanged. The only new sets are controls (B2, B3b), built from
half-A data before any half-B outcome of this file is computed.

**Revision after Codex (gpt-6.2-sol, xhigh), before any full run:**
- every check now fails closed: an analysis whose check fails writes INVALID and no decision;
- B2's primary controls are now matched on the natural swap, not the edit;
- B2 adds a same-class different-verb swap, which decides the "class-specific" wording;
- B2 adds a v2-coordinate swap between the paired natural sentences, and asks whether P mediates it;
- B3c's ratio dissociation (old D2) is descriptive, gated on a positive passive contrast, and B3c is labelled a
  follow-up on reused activations;
- A1's wording is narrowed, and (D_v2 − D_shuf) / G is added;
- B1 reports both references, with no additive "slice" wording;
- the "no non-P neuron reaches the core" sentence is corrected (B3b).

## Part A. Trained-direction controls for C1 (Pythia, Gemma, OLMo)

**Instruments.** Each is a rank-1 interchange with its own unit direction per (split, fold), at the first two rule
sites of each model's C1 table (Pythia 8, 12; Gemma 5, 8; OLMo 9, 11).
- **Shuffled-label trained comparator** (`shuffled_bases.pt` of the same v2 final):
  - one direction per (split, fold), so 15 independent label permutations across v2's training verbs (`run_das_v2.py`,
    generator 100 · split + fold);
  - the same objective, pool and epochs as v2, a different training seed.
  - The permutation changes the BCE targets and which swaps count as same- or cross-class, so this is a label-corrupted
    trained direction, not a null.
- **Random directions:** 3 draws per (split, fold) key, standard normal then normalized, first rule site only.
  Generator 910001 + 100 · draw + 10 · split + fold, a stream no stored run uses.

**Run.** `run_passive_test.py` with v2's `passive_rule` prompts, plan, folds and settings. Only `--site-dirs` changes,
pointing to a directory with the control's `bases.pt` and the final's `pairs_folds.csv`.
- **Jobs:** one per model and instrument family. `heldout_swaps.csv.gz` is absent from control directories, so that
  diagnostic is skipped (v2's swaps refer to v2's basis).

**Checks (fail closed; any failure writes INVALID and no decision):**
- natural readouts equal v's stored expanded run: same pid set, all finite, max abs difference ≤ 1e-6;
- self-patch ≤ 0.1 (amended Nov 26 from 0.05, see Amendments; v2's own runs: 0 to 0.042);
- the runner's fold check;
- the control's keys, shapes and unit norms equal v2's, and the sha256 in `run_meta.json` equals the control's
  `control_meta.json`;
- every expected control × site file present;
- the plan's `row` column equals its position;
- every patched row finite.

**Analysis.** `analyze_c1_controls.py`. One set of joint draws per site is shared by v2, every control and G:
contexts, DAS donor pairs and pairs within band, seed 17, 2,000 draws. The point estimates equal `transfer.csv`'s. CI
endpoints can differ slightly from it, because the stored analysis consumes the generator in a different order.

Per site and readout (*by*, O), on eval plain (89):
- D and D / G;
- v2 − control in nats and as a share of G;
- control / v2.

Eval new plain and lemma-fresh are descriptive.

**Declared (per model, first rule site, *by*):**
- **A1:** lo95(D_v2 − D_shuf) > 0.
- **A2:** for each of the three random draws, the 95% CI of D lies within ±0.1 · G (lo95(D + 0.1 G) > 0 and
  hi95(D − 0.1 G) < 0). This is a statement about these three draws.

**Descriptive:** |cos| with v2 per (split, fold) (random 2,048-dimensional directions: about 0.02); the second rule
site; the O readout.

**Wording.**
- A1 holds: "v2 transfers more strongly than its shuffled-label trained comparator, by Δ nats (Δ / G of the natural gap)
  on the same items and donor draws; the comparator itself moves *by* by X of the gap."
- A1 fails at a site: v2 is not distinguished from its label-corrupted comparator there, and C1 says so.
- "Not an artefact of DAS optimization" rests on u_md, the direction estimated without DAS, stated separately. A1 does
  not carry that claim.

## Part B. Pythia C4 phase 1b

**Common to all of Part B:**
- **Edit rows:** phase 1's mediation keys (half B, 46 pairs, 8,414 keys), rebuilt and asserted equal to
  `results/pythia/das_v2/mech/med_v2/mediation_keys.parquet`.
- **Natural rows:** the same sampled items (half B, 46 pairs, 32 contexts each, 1,472 items).
- **Readout:** log P(" by") at the participle's last token, fp32.
- **Shares:** pair-averaged, then ratios of averages. Pair bootstrap, 2,000 draws, seed 17. Controls are the mean of
  their sets within each draw.

### B1. Where the edit's effect goes: component sweep (v2, descriptive)

Each component's output at the participle's last token is set to its value in the other arm's run of the same key,
as phase 1's T ← I and I ← T. Components:
- attention output (`attention.dense`) for each of layers 8–23;
- MLP output for each of layers 8–23;
- groups: all attention 8–23, all MLP 8–23, all components 8–23, MLPs 11 + 14.

Under "all components", only the residual path from the edit (output of block 7) to the final layer norm differs.

**Checks (fail closed):**
- MLPs 11 + 14 reproduce phase 1's ALL-neuron clamp, max |Δ log P| ≤ 1e-4 per key, both directions;
- TT and II equal phase 1's (≤ 1e-5);
- empty-set clamps equal their references;
- no non-finite cells;
- under "all" (T ← I), the final-norm input minus II's equals the site-8 state difference, max abs difference
  ≤ 1e-4 × the largest abs value of that difference.

**Reported:**
- m (I-referenced) and r (T-referenced) for every component and group, with CIs;
- the residual-only shares 1 − m(all) and 1 − r(all);
- the sum of single-component m beside the joint clamps (clamps interact; no additive partition is implied);
- P and the MLP 11 + 14 joint clamp side by side;
- the ranks of MLP 11 and MLP 14 among the single components.

The paper keeps one grouped comparison; the layer sweep goes to the appendix.

### B2. Natural mediation: do the same neurons carry the model's own T − I gap?

Each half-B item has a good (T verb) and a bad (I verb) prompt with the same context. G = Y[good] − Y[bad].

**(a) Neuron swap.**
- B_nat(S) = Y[good] − Y[good ← bad(S)]: S's post-activations at the last token set to the bad prompt's;
  m_nat = B_nat / G.
- R_nat(S) = Y[bad ← good(S)] − Y[bad]; r_nat = R_nat / G.
- Both are reported in nats and as shares of G.

**Natural-matched controls (primary).** Built from a half-A natural run (the half-A pairs' sampled items, good and
bad, unclamped) before half B is opened:
- **magnat_k (k < 20):** layer × decile matching on |pair-averaged half-A good − bad| of each neuron's post-activation,
  i.e. the size of the natural swap. Decile counts are matched to P's, drawn from the complement of P, with the lower
  decile shortfall rule as in phase 1. Seed 900101 + 3000 + k.
- **denat_k:** the same on |pair-averaged half-A good − bad × the neuron's " by" output weight| (unscaled coupling).
  Seed 900101 + 4000 + k.
- The P vs control means of both matching variables, and the shortfall, are reported. Phase 1's edit-matched mag / de
  / rand are reported as secondary.

**(b) Same-class swap control.**
- Each half-B item's good prompt gets a donor: the good prompt of another eval plain pair with the same context and a
  different good lemma. Same token count where available, otherwise nearest; ties at random, seed 17.
- B_same(S) = Y[good] − Y[good ← good′(S)].
- If the neuron swap only disrupted generic lexical content, B_same(P) would match B_nat(P).

**(c) v2-coordinate swap between the paired sentences.** At site 8, the good prompt's v2 coordinate is set to the
bad prompt's own (h ← h + ((h_bad − h) · d) d), with d the split-0 basis of the item's fold from the plan, and the
converse.
- C_g = Y[good] − Y[good ~ bad]; C_b = Y[bad ~ good] − Y[bad]; shares of G.
- **Mediation of the swap by S** (S held at the unswapped run's values during the swap):
  - BC_g(S) = Y[good ~ bad | S ← good] − Y[good ~ bad], as a share of C_g;
  - BC_b(S) = Y[bad ~ good] − Y[bad ~ good | S ← bad], as a share of C_b.
- Sets: P, ALL, and phase 1's mag (edit-matched; the swap is an interchange along v2).

**Declared:**
- **Gate:** G lo95 > 0.
- **NM1:** lo95 m_nat(P) > 0 and lo95(m_nat(P) − mean m_nat(magnat)) > 0.
- **NM2:** the same for r_nat.
- Beyond denat and beyond the edit-matched mag are stated separately.
- **NM-class:** lo95((B_nat(P) − B_same(P)) / G) > 0. Without it the text may not call the swap effect class-specific.
- **CS1 (the coordinate carries part of the model's own gap):** lo95 C_g / G > 0.
- **CS2 (P mediates part of the coordinate's natural effect):** lo95 BC_g(P) / C_g > 0 and lo95((BC_g(P) − mean
  BC_g(mag)) / C_g) > 0. The b-direction counterparts are reported, not declared.
- **Descriptive:** m_nat(ALL), P / ALL where lo95 > 0, new / lemma-fresh subsets.

**Wording:**
- **NM1:** "Replacing P's activations in the good prompts with those of the paired bad prompts lowers log P(*by*) by
  X nats, Y of the observed gap (natural-matched neurons: Z)."
- **"Class-specific"** only with NM-class.
- **CS1 + CS2:** "Swapping only the v2 coordinate between the paired sentences removes C of the gap, and holding P fixed
  blocks B of that."
- **NM1 fails:** a limit in §5: "the edit is read through these neurons; the model's own contrast is not detectably
  carried by them under this swap".

### B3. The core neurons: what they respond to and what each carries

**Core (fixed before any B3 run):** the top 5 of P by half-A direct effect on the switch
(`results/pythia/das_v2/mech/conj_v2/conjunction_switch_neurons.csv`): MLP11.n3462, MLP14.n5407, MLP14.n2114,
MLP11.n4925, MLP14.n4241. These are v's top 5 in another order, so R5-B7's instrument-free Pile-10k activations cover
them.

**B3a. Labels from natural text** (done, `reports/pythia/v2_mech/b3a_annotation.md`, κ 0.91).
- Ranks 1–3 keep R5-B7's labels: MLP11.n3462 *gap-sensitive participle reader*; MLP14.n2114 and MLP14.n5407
  *passive-participle detectors*.
- Ranks 4–5 were labelled blind by Claude and Codex with B7's frozen guide and rule: MLP14.n4241 *passive-participle
  detector*; MLP11.n4925 *other* (mostly stranded prepositions, object gaps 0.30–0.35).
- The analysis requires exactly these five labels and does not let the B3a file overwrite B7's.

**B3b. Causal role of each core neuron (descriptive).**
- **Sets:** each core neuron alone, the top 3, the top 5.
- **Per-layer controls:** the 20 non-P neurons with the largest half-A |direct effect| on the was metric.
- **Coupling, measured while building the sets:** the core's half-A |direct effect| is 0.055–0.197 (MLP 11) and
  0.068–0.157 (MLP 14). The strongest non-P neuron is 0.058 in MLP 11, above MLP11.n4925's 0.055, and 0.013 in MLP 14.
  So the controls are the strongest available non-P comparators. They do not isolate the conjunction property at fixed
  coupling.
- **Reported:** edit m, r and natural m_nat, r_nat per set, against the layer's controls (mean, max, rank). The edit
  shares are also given with donors averaged within item first, for comparison with the natural shares.

**B3c. Frame profiles (a targeted follow-up on reused activations).** For each core neuron, the T − I difference of its
post-activation at the verb's last token:
- **passive:** good − bad on B2's half-B items;
- **C8 frames:** the stored natural post-activations (`results/round5/shared/sr_natural_fg.parquet`, a column per
  neuron mapped through `sr_meta.json`), T lemmas minus I lemmas, on a complete (class, lemma) grid:
  - embedded "I know what John V" (fill) and "I know that John V" (nofill);
  - matrix "What did John V" and "Did John V".

Bootstrap: C8 lemmas resampled within class and passive pairs, on the same draw index (the two populations share 28
lemmas; the intervals condition on both item pools).

**Declared:**
- **D1 (the gap reader's class response depends on a filler):** for MLP11.n3462, lo95(fill T − I) > 0 and
  lo95((fill − nofill) T − I) > 0.

**Descriptive:**
- every neuron's raw contrasts with CIs;
- ρ = fill / passive, shown only for neurons whose passive contrast has lo95 > 0;
- the same on C8's gated population (non-DAS T lemmas and the kept non-DAS I lemmas of `kept_I_emb.csv`).

**Wording:** "the core neurons have different frame profiles". A division of labour is not claimed. D1 supports "the
gap reader's class response appears with a filler", as an activation profile, not a causal role in the filler-gap
construction.

**Disclosed:**
- C9's set-level results on these items were seen on Oct 14 (top 10 respond more in the "what" frame; per-neuron sign
  agreement and cosine).
- Per-neuron profiles are first computed here.
- The B7 labels of ranks 1–3 were seen.

## Amendments

- **Nov 26, self-patch tolerance 0.05 → 0.1 (Part A), before any C1 number was produced.** The Gemma analysis failed
  closed on one check: random draw 0 at site 5 had a self-patch max |ΔM| of 0.052. Every other check of every control
  passed. The self-patch writes the item's own state back, so it does not depend on the direction: it measures batching
  noise in the patched forward. That noise is exactly 0 in Pythia and OLMo and 0.03–0.05 in Gemma (v2's own Gemma
  runs 0.033–0.042, the other Gemma controls 0.030–0.045; the stored held-out recompute reaches 0.091, median 0). A
  pipeline fault would show as an error of order 1. The analysis exited at the check, so no D, G or decision was
  written for any Gemma instrument. The random draw is not rerun.
- **Nov 26, smoke fix (B2), before any full B run.** `run_mech_b.py --mode natural_halfA` overwrote
  `natural_halfA.npz` with its readout file at exit; the readouts now go to `natural_halfA_by.npz`. No rule changes.

## Forking paths

- Part A's sites and readouts are C1's.
- B1 is descriptive over the full block list.
- B2's natural-matched controls and same-class donors are fixed by rules written before half B is opened. The
  coordinate swap follows Codex's suggestion, made before any B run.
- The core is phase 1's exploratory top 5. B3c reuses an inspected dataset.
- **Not run:** heads (phase 2), a second model, a serial head → neuron → *by* test, fresh-verb frame replication.
