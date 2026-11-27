# C1 trained-direction controls and C4 phase 1b: readings (Nov 27)

Rules: `reports/v2_controls_mech_b_rules.md` (declared Nov 26, revised after Codex before any run; two amendments
before any number, listed there). Tables: `reports/v2_controls/c1_controls_{pythia,gemma,olmo}.md` and
`reports/pythia/v2_mech/mech_b.md`. Decisions: `results/{m}/das_v2/c1_controls/c1_controls_decisions.json` and
`results/pythia/das_v2/mech_b/mech_b_decisions.json`. Jobs: C1 9574034–50 and 9578071 (Gemma analysis after the
amendment); B 9586154 / 9586155 at c1f2a06. Every check passed; no part was INVALID.

## Part A. C1 controls (first rule site, *by*, eval plain, 89 pairs)

| model, site | v2 D / G | shuffled D / G | v2 − shuffled, nats [CI] (/ G) | shuffled / v2 | \|cos\| shuffled–v2, mean (min–max) | random D (3 draws) |
|---|---|---|---|---|---|---|
| Pythia 8 | 1.05 | 0.71 | +0.26 [0.21, 0.31] (0.34) | 0.67 | 0.75 (0.69–0.83) | 0.00 |
| Gemma 5 | 1.27 | 0.80 | +0.18 [0.09, 0.27] (0.46) | 0.63 | 0.56 (0.12–0.67) | 0.00–0.01 |
| OLMo 9 | 0.89 | 0.66 | +0.12 [0.10, 0.15] (0.23) | 0.74 | 0.76 (0.71–0.80) | 0.00 |

Second rule sites (shuffled only): Pythia 12 v2 1.11 vs 0.47 (+0.48 nats); Gemma 8 1.77 vs 1.31 (+0.18); OLMo 11 0.46
vs 0.34 (+0.07). On O the comparator is also weaker everywhere (shuffled / v2 0.28–0.65).

- **A1 holds in all three models:** v2 exceeds its shuffled-label DAS comparator.
- **A2 holds for all 9 random draws:** each CI lies within ±0.1 G.
- **Reading (outline C1, after Codex):** the comparator is not a null. Its overlap with v2 is about what the objective
  gives when labels carry no information (c = 1/√2 under margin averaging), so v2 − comparator is not "what the
  labels add". Gemma 5 overlaps less (0.56; one fold 0.12) and still transfers at 0.63 of v2.

## Part B. Pythia C4 phase 1b (46 half-B pairs)

### B1. Where the edit's *by* effect goes (descriptive)

D = +0.78 [0.68, 0.88] nats. The clamp set every component's output at the participle's last token to the other arm's
value.
- **Most of the effect is not recomputed at the readout position.** With every attention and MLP output of layers
  8–23 clamped, 0.58 [0.51, 0.66] of the effect remains (T-referenced 0.62). That remainder is the edited site-8
  state carried by the residual stream to the final norm.
- **Components that carry it forward:** MLP14 +0.17, attention 23 +0.17, MLP15 +0.12, MLP11 +0.10 (all CIs above 0).
  MLP14 ranks first of the 32 single components and MLP11 fourth. MLP 11 + 14 together: 0.26, which reproduces phase 1.
- **Components that oppose it:** MLP23 −0.50 [−0.54, −0.46], MLP12 −0.15, MLP20 −0.09. Clamped to the I value, each
  of these makes the edit's effect larger.
- **The clamps interact:** single-component shares sum to −0.22, against 0.42 for the joint clamp. No additive slices.

### B2. Natural mediation: the model's own T − I gap

G = +0.80 [0.41, 1.19] nats (gate holds).

| declared | rule | result | holds |
|---|---|---|---|
| NM1 | lo95 m_nat(P) > 0 and beyond magnat | m_nat(P) 0.06 [−0.02, 0.13]; P − magnat 0.06 [−0.03, 0.15] | **no** |
| NM2 | the same for r_nat | r_nat(P) 0.23 [0.13, 0.36]; P − magnat 0.10 [−0.01, 0.23] | **no** |
| NM-class | lo95((B_nat(P) − B_same(P)) / G) > 0 | +0.15 [0.09, 0.28] | yes |
| CS1 | lo95 C_g / G > 0 | 0.14 [0.09, 0.27] (0.11 nats) | yes |
| CS2 | lo95 BC_g(P) / C_g > 0 and beyond mag | 0.27 [0.18, 0.34]; P − mag 0.29 [0.20, 0.38] | yes |

- **NM1 fails, so the declared limit applies:** "the edit is read through these neurons; the model's own contrast is
  not detectably carried by them under this swap." The natural-matched controls match P on the matching variable
  (|half-A good − bad| 0.0443 vs 0.0438; no shortfall).
  - All of MLPs 11 + 14: m_nat 0.03 [−0.17, 0.13], so the null is not specific to P.
  - The restoring direction is larger (P 0.23, ALL 0.61), but P's r_nat does not clear its controls.
- **NM-class holds, but it does not rescue NM1.** It is carried partly by B_same(P) being negative (−0.09 [−0.26,
  −0.01]): a different transitive verb's P values raise log P(*by*) slightly. There is no class-specific mediation
  claim without NM1.
- **CS1 + CS2 hold. This is the link between v2, the neurons and natural behavior.** Wording as declared: "Swapping
  only the v2 coordinate between the paired sentences removes 0.14 of the gap, and holding P fixed blocks 0.27 of
  that."
  - The good → bad direction is 0.14 of G. The bad → good direction is larger, 0.46 [0.34, 0.78] of G, and P blocks
    0.21 of it.
  - The chain through P is therefore small in natural units: about 0.14 × 0.27 ≈ 0.04 of G in the declared
    direction.
- **Exploratory, labelled:** ranked subsets do carry some of the natural gap where the full set does not (top 50
  0.09 [0.04, 0.15], top 100 0.11 [0.05, 0.17]), as phase 1 found for the edit. This ranking is from the edit, half A.
- **New / lemma-fresh half-B pairs (19 / 15):** G CIs are wide (the lemma-fresh one includes 0); nothing to read.

### B3b. The core neurons' causal shares (descriptive)

| neuron (label) | edit m | natural m_nat | edit rank among 21 (self + 20 layer controls) |
|---|---|---|---|
| MLP14.n5407 (passive detector) | **0.093** [0.081, 0.106] | −0.01 [−0.08, 0.03] | 1 |
| MLP14.n2114 (passive detector) | 0.048 [0.036, 0.062] | **0.058** [0.017, 0.119] | 1 (natural 1) |
| MLP11.n3462 (gap reader) | 0.009 [0.003, 0.014] | 0.00 | 2 |
| MLP14.n4241 (passive detector) | 0.008 [0.006, 0.011] | 0.00 | 1 |
| MLP11.n4925 (other) | −0.005 | 0.00 | 21 |
| top 5 together | **0.167** [0.148, 0.186] | 0.05 [−0.02, 0.09] | |

- **Five neurons carry 0.89 of P's mediated share** of the edit (0.167 of 0.188). Two MLP14 passive detectors do most
  of it: n5407 0.093 and n2114 0.048.
- **The largest edit mediator, n5407, carries none of the natural gap.** Its natural share is −0.01, and its natural
  T − I activation is 0 in every frame (B3c). The only core neuron with a detectable natural share is n2114 (0.058).
- **The gap reader, n3462, has the largest direct effect on the switch but mediates little of the edit (0.009).** The
  direct-effect ranking and the mediation ranking differ.

### B3c. Frame profiles (natural T − I of the post-activation; reused C8 activations)

| neuron | passive | *what* (fill) | *that* (no fill) | matrix *what* | matrix *did* |
|---|---|---|---|---|---|
| MLP11.n3462 (gap reader) | 0.00 [−0.11, 0.11] | **1.19** | 0.04 | 0.83 | 0.07 |
| MLP11.n4925 (other) | −0.03 | **0.73** | −0.08 | 0.62 | 0.09 |
| MLP14.n2114 (passive) | **0.30** [0.11, 0.49] | −0.04 | −0.01 | −0.04 | −0.06 |
| MLP14.n4241 (passive) | **0.47** [0.28, 0.66] | **0.48** | −0.05 | 0.67 | −0.04 |
| MLP14.n5407 (passive) | −0.01 [−0.27, 0.26] | 0.01 | 0.02 | −0.01 | 0.01 |

- **D1 holds** (all lemmas, and on the C8-gated population): n3462's class response is present with a filler and
  stronger than without one (fill − nofill +1.15 [1.06, 1.23]). Wording: "stronger with a filler".
- **The profiles differ** (wording as declared; no division of labour). n2114 responds to verb class only in passives.
  n3462 and n4925 respond only with a filler. n4241 responds in both, at equal size (ρ = fill / passive 1.03
  [0.70, 1.79]). n5407 shows no class response in any natural frame.
- These are follow-up readings on activations that were inspected earlier (rules, disclosed).

## What this does to C4

1. **The edit and the model's own contrast are only partly the same mechanism.**
   - The edit's mediated share runs mostly through n5407, which has no natural class response.
   - Under neuron swaps, the natural gap is not detectably carried by P or by all of MLPs 11 + 14 (NM1 / NM2 fail).
   - What does connect them is the coordinate swap: the natural v2 coordinate carries 0.14–0.46 of the model's own
     gap, and P mediates about a quarter of that piece.
2. **§5 has three parts now:**
   - the edit's path (mostly direct; MLP11 / 14 / 15 amplify; MLP12 / 23 oppose; P 0.19, five neurons 0.17);
   - the natural link through the v2 coordinate (CS1, CS2);
   - the core neurons' distinct frame profiles. Only n2114 carries the natural gap.
   The NM1 limit sentence goes in the text.
3. **The interpretability point the section can make:** the neurons that read the coordinate split by construction
   (passive only, filler only, both). The edit recruits one route (n5407) that natural sentences do not use. This
   fits Codex's worry from the shuffled-label result: an edit along a margin axis can drive a continuation pathway
   that the model's own contrast does not use.
