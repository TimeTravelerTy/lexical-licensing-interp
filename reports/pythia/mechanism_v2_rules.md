# Pythia reading mechanism (C4) with v2 and u_md on the paper's evaluation set: rules (declared Nov 25, before any run)

**Why.** Every C4 result (step-1 MLP decomposition R2-mech, heads B7, voice switch D10, conjunction neurons D11,
natural necessity B5) was measured with the stored site-8 v on the 64 original primary pairs. The paper's instrument is
now v2 (`reports/v2_retrain_rules.md`), and its evaluation set is the 224 held-out pairs
(`data/das_v2/expanded_eval_pairs.csv`, `eval_v2`). This port does three things:
- reselects the conjunction neurons with v2;
- tests the frozen set with the training-free u_md and a shuffled-label direction;
- adds the missing test: whether the selected neurons mediate the verb-site edit's effect on *by*. This is the bridge
  from C1 to C4.

v appears only in the analysis-code reproduction check. Reviewed by Codex (gpt-6.2-sol, xhigh) before any run; changes
from that review are folded in below.

## Fixed inputs

- **Model:** Pythia-1.4B, fp32 (as every stored C4 run).
- **Instruments at site 8 (the participle's last token), split 0, fold from the stored plan:**
  - **v2:** `results/pythia/das_v2/final_strict_site8/bases.pt` (sha256 a4994801…).
  - **u_md:** `results/pythia/expanded_eval/umd_bases/umd_site8/bases.pt`.
  - **Shuffled-label control:** v2's own `shuffled_bases.pt` from the same final (same keys).
  - Each uses its own rank-1 interchange h ← h + ((h_src − h)·d) d with the donor's state. No dose matching: every
    share below is relative to the same instrument's own effect. The raw numerator and denominator are reported with
    every share.
- **Items:** the stored expanded passive test (`data/verb_expansion/eval/passive_test/`).
  - The items file is filtered to the `eval_v2` pairs **before** context sampling (`data/das_v2/mechanism/items.csv`,
    written by the job). Prompts, plan, plan row ids, splits and folds are unchanged. Folds are never reassigned, which
    keeps the protection of v's eight fold-protected pairs.
  - The runners keep `bad_class == "plain"`, giving the **89 plain pairs**.
  - Rows are as B7: split 0, bad side, T and I donors from the plan, and 32 contexts per pair (seed 17; all 89 pairs
    have 32).
- **Pair halves:** `pair_halves(bands, seed 17)` over the 89 pairs, stratified by `band`. This gives 43 A / 46 B pairs.
  - The halves share no target lemma, participle or base prompt.
  - They do share all 126 contexts and the whole donor pool (29 donor pairs, 406 donor prompts). So half B tests
    transfer across target verbs with the donor and context pools fixed. Pair-bootstrap intervals are conditional on
    those pools.
  - B holds 19 new pairs, 15 of them lemma-fresh.
  - Half A selects; every declared test is on half B.
- **Layers and readouts:** MLPs 11 and 14 (post-activation, the input of `dense_4h_to_h`, at the participle's last
  token). " by" is read as in the stored runs: centered r_by for the direct-effect switch, and log P(" by") for
  mediation.

## Phase 1 (this run)

### Stage 4. Conjunction neurons reselected with v2 (D11 procedure)

Runner: `run_conjunction.py` with v2, then `analyze_conjunction.py` (unchanged selection code).

**Selection on half A:**
- I_n = (was T − I) − (has T − I);
- one-sample t across half-A pairs;
- BH-FDR q < 0.05 over 16,384 neurons;
- passive-conjunction: the was effect has the direction of I_n, and |has| < 0.5 |was|.

The selection (pairs, halves, mask, signs) is saved to `conjunction_selection.npz` and frozen for every later stage.

**Declared, half B:**
- **N1 (exists):** ≥ 10 passive-conjunction neurons; sign agreement ≥ 0.8; aligned I_n and was-effect CIs above 0;
  |has| < 0.5 × was. As D11.
- **N2 (carries part of the switch):** the MLP 11 + 14 " by" switch (direct effect of I_n) has lo95 > 0, and the set's
  share of it has lo95 > 0.
- **"At least a quarter":** only if the share's lo95 ≥ 0.25. D11's stored rule (point ≥ 0.25 and lo95 > 0) is reported
  beside it.

### Stage 5. Instrument checks on the frozen v2 set

Runner: `run_conjunction.py` with u_md bases, and again with the v2 shuffled-label bases, on the same rows. Analysis:
`analyze_conjunction.py --frozen <v2 conjunction_selection.npz>`. Nothing is selected with these instruments.

**Declared, half B, u_md:**
- **U1:** sign agreement with the v2 half-A signs ≥ 0.8, and aligned I_n CI above 0.
- **U2:** u_md's own MLP 11 + 14 switch has lo95 > 0, and the set's share of it has lo95 > 0 ("at least a quarter" as in
  N2).
- **Reading:** "the frozen v2-selected set also responds to a training-free class contrast". This is convergence of two
  class-derived directions. It is not proof that no trained direction is involved, since u_md is also estimated from
  class-labelled items.

**Shuffled-label direction (descriptive specificity check):** its aligned I_n and switch on the set, beside v2's and
u_md's. The expectation is a much smaller response; there is no pass / fail.

### Stage 6. Mediation

Runner: `run_conjunction_mediation.py`; analysis: `analyze_conjunction_mediation.py`.

**Rows:**
- Half-B pairs.
- Each plan key (base item × donor pair) has exactly one T row and one I row with the same base, fold and donor
  subject. The runner asserts this.
- The frame is "The N was V" (stage 4's ids); the readout is log P(" by") at the participle's last token.

**Runs per key and instrument X:**

Y[a, b] is the edit from arm a with set S clamped to its values in arm b's run of the same key.
- TT and II are the full edits; nat is no edit.
- **D = Y[TT] − Y[II]** (the C1 contrast).
- **Primary (I-anchored):**
  - **B(S) = Y[TT] − Y[T ← I(S)]:** the effect removed by clamping S to the I run inside the T run. **m(S) = B / D.**
  - **R(S) = Y[I ← T(S)] − Y[II]:** the effect restored by clamping S to the T run inside the I run. **r(S) = R / D.**
- **Secondary:** Bnat(S) = D − (Y[T ← nat(S)] − Y[I ← nat(S)]). Both arms are clamped to the unedited base, a different
  operating point.
- Contrasts are averaged within target pair, then over pairs; shares are ratios of those averages.

**Sets S** (`build_mediation_sets.py`, from the v2 stage-4 outputs, frozen before stage 6 runs; the same sets are reused
for u_md):
- **P:** the passive-conjunction set.
- **ALL:** every neuron of MLPs 11 and 14, as a joint intervention.
- **P_mlp11 and P_mlp14:** the layer-only parts of P. The joint clamp counts the serial route once; separate layer
  effects are not summed.
- **mag_k (k < 20), the declared comparator:**
  - layer-matched counts, matched by decile of the half-A |mean was T − I| (pairs equally weighted);
  - deciles within each layer over all 8,192 neurons, ties broken by index;
  - drawn without replacement from the complement of P;
  - a decile with too few non-P neurons takes all of them, and the shortfall comes from the nearest lower decile, then
    the next. The shortfall count and the P vs control mean magnitudes are reported.
- **de_k (k < 20), the stricter comparator:** the same matching on the half-A |direct effect of was T − I on " by"|,
  i.e. activation × output coupling.
- **rand_k (k < 20):** layer-matched, uniform over the complement of P.
- **Exploratory:** top 10 / 50 / 100 of P by half-A direct effect.
- Seeds: 900101 + k (rand), + 1000 + k (mag), + 2000 + k (de).

**Declared (per instrument):**
- **Gate:** D has lo95 > 0. Without it, no share is read.
- **M1 (P mediates part of the edit's effect):** lo95 m(P) > 0 and lo95(m(P) − mean m(mag)) > 0.
- **M2 (P restores part of it in the I run):** the same for r.
- **Beyond output coupling:** the M1 / M2 difference against de, stated separately.
- **u_md:** M1 and M2 are read again with u_md. Only the frozen v2 sets are used.
- **No size threshold.** The sentence states m(P) in nats and as a share, against the matched controls.

**Reported with every reading:**
- the excess over controls, in nats and as a share;
- the control distribution and P's rank among each family (descriptive);
- m(ALL), and P / ALL only where lo95 m(ALL) > 0;
- the 11-only, 14-only and joint numbers.

**Bootstrap:** pairs, 2,000 draws, seed 17. Controls enter as the mean over their 20 sets within each draw, so the
uncertainty is over pairs, conditional on the sets.

**Descriptive only:** stage-4 to stage-6 numbers on the new (19) and lemma-fresh (15) pairs of half B; the overlap of the
v2 set with v's stored 2,782.

## Phase 2 (deferred; declared separately before it runs, if time allows)

Heads (B7) and the voice switch (D10) with v2. They stay out of phase 1 because the stored analyzers do not meet this
file's standard:
- they select heads on context halves and test on pooled halves, not on pair half B;
- B7's H1 / alternative pass vacuously with zero eligible MLPs (`analyze_path_patch.py:85`, 0 ≥ ceil(0/2));
- D10's fixed five heads are v's B7 heads;
- `analyze_voice.py` subtracts T / I arrays by position, with no keyed alignment check.

Each needs a fix first. Until then the paper's §5 cites the v-based routing as prior work on v in one labelled appendix
sentence, or drops it. **B5 is cut:** stage 6 gives the causal number that sits beside the switch share in the figure.

## Checks (a stage whose check fails is reported as invalid, not read)

- **Analysis-code reproduction:** `analyze_conjunction.py` (edited: frozen mode, an empty-cell assertion, new CIs on a
  separate random stream) is run on v's stored `conjunction.npz`. Every stored decision and estimate in
  `conjunction_decisions.json` must reproduce exactly. The runner `run_conjunction.py` is not edited.
- **Inventory:** 89 pairs, 43 / 46 halves, 32 contexts per pair, no empty pair × frame × donor-class cell. The mediation
  keys are recorded with both plan row ids.
- **Mediation hooks:**
  - every clamp with the empty set equals its unclamped run (max |Δ| ≤ 1e-5);
  - entries outside S are unchanged at the hook (max 0);
  - entries in S equal their targets (max ≤ 1e-6).
- **C1 row check (v2):** Y[TT] and Y[II] are joined to the stored per-row v2 expanded run
  (`results/pythia/das_v2/expanded_eval/passive_rule/patches_site8.parquet`, bf16) by plan row:
  - no missing rows;
  - median |Δ| ≤ 0.05 per arm;
  - |pooled ΔD| ≤ 0.02 nats.

## What each outcome means for the paper

- **N1–N2 hold:** C4's neuron claim is restated with v2's numbers; v's appear nowhere.
- **A v2 stage fails:** the failure is reported in C4 with the v2 number. v's result is not substituted.
- **U1–U2 hold:** one sentence: the selected readers also respond to the training-free class contrast.
- **M1 holds:** the bridge sentence: "on held-out verbs, clamping the selected neurons to their I-run values removes X
  nats (Y%) of the edit's T − I effect on *by*, against Z for activation-matched neurons." M2 is reported independently.
  The "beyond de" result decides whether the text may say the conjunction property matters beyond output coupling.
- **M1 fails:** mediation is unresolved under this clamp. C4 is then presented as where the edit's signal is read, not
  as the route of its effect, and §5 shrinks to 0.6–0.8 page.
- **Even if every stage holds:** no serial head → neuron → *by* claim is made. That needs a serial intervention, which is
  not run.

## Forking paths

- The layers (11, 14), the frames and the D11 selection rule were set on v and on pairs that included original half-B
  pairs, and are reused unchanged. The lemma-fresh half-B rows are the cleanest check on that earlier choice.
- 59 of the 89 pairs are originals whose v-based C4 readings were seen.
- The mediation design, the comparators and the N2 / U2 wording are fixed here, before any v2 C4 output exists.
