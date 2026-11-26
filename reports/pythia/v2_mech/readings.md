# Pythia reading mechanism (C4) with v2: readings against the declared rules (Nov 26)

Rules: `reports/pythia/mechanism_v2_rules.md` (declared Nov 25 before any run, commit 0c8f4e0). Run: TSUBAME jobs
9566080–9566087 at commit 0c8f4e0. Outputs: `reports/pythia/v2_mech/{d11_v2, d11_umd_frozen, d11_shuf_frozen,
mediation}.md`, `results/pythia/das_v2/mech/`. Half A (43 pairs) selects; every number below is on half B (46 pairs,
8,414 keys) unless marked otherwise.

## Checks: all pass

| check | rule | result |
|---|---|---|
| Analysis-code reproduction on v's stored `conjunction.npz` | every stored decision exact | 17 / 17 keys equal, switch-neuron table byte-identical |
| Inventory | 89 pairs, 43 / 46, 32 contexts, no empty cell | 89 pairs, 32,592 rows, 43 / 46; empty-cell assertion passed |
| Hooks (v2 and u_md mediation) | empty clamp = reference ≤ 1e-5; outside S unchanged; inside S = target ≤ 1e-6 | all 0.0; no NaN cells |
| C1 row check (v2 vs stored bf16 run) | no missing rows; median abs. difference ≤ 0.05 per arm; abs. pooled ΔD ≤ 0.02 | 0 missing; median 0.042 (T) / 0.046 (I); pooled ΔD 0.002 |
| Matched controls | shortfall recorded | 0 drawn from lower deciles (mag, de). Mean half-A magnitude: P 0.042 vs mag 0.039 (rand 0.020). Mean half-A abs. direct effect: P 0.00092 vs de 0.00064 |

The de match is by decile, so the de controls stay less coupled on average than P (0.70×). That only weakens "beyond de"
a little: the de controls mediate 0.028 against P's 0.188.

## Declared readings

| reading | rule | v2 | holds |
|---|---|---|---|
| **N1** neurons exist | ≥ 10; sign agreement ≥ 0.8; aligned I_n and was-effect CIs > 0; abs. has < 0.5 × was | 2,986 (MLP11 1,488 / MLP14 1,498); sign agreement 0.99; I_n +0.036 [+0.033, +0.038]; was +0.039, has +0.003 | yes |
| **N2** they carry part of the switch | switch lo95 > 0 and share lo95 > 0 | switch +1.057 [+0.987, +1.136]; share **0.62 [0.58, 0.66]** | yes |
| "at least a quarter" | share lo95 ≥ 0.25 | 0.58 | yes |
| **U1** u_md, frozen set | sign agreement ≥ 0.8; aligned I_n CI > 0 | 0.92; +0.027 [+0.025, +0.029] | yes |
| **U2** u_md's own switch | switch lo95 > 0; share lo95 > 0 (≥ 0.25 for "a quarter") | switch +0.802 [+0.744, +0.866]; share 0.70 [0.65, 0.75] | yes (and a quarter) |
| Gate (v2 / u_md) | D lo95 > 0 | +0.779 [+0.679, +0.882] / +0.630 [+0.531, +0.736] | yes |
| **M1** P mediates (v2) | lo95 m(P) > 0 and lo95(m(P) − mean m(mag)) > 0 | m(P) **0.188 [0.148, 0.228]** (0.147 nats); P − mag 0.175 [0.134, 0.219] | yes |
| M1 beyond de (v2) | lo95(m(P) − mean m(de)) > 0 | 0.160 [0.119, 0.205] | yes |
| **M2** P restores (v2) | the same for r | r(P) **0.279 [0.232, 0.326]** (0.217 nats); P − mag 0.269 [0.216, 0.321]; P − de 0.243 [0.194, 0.295] | yes |
| M1 / M2 (u_md, frozen sets) | as above | m(P) 0.186 [0.144, 0.228], P − mag 0.184, P − de 0.169; r(P) 0.258 [0.211, 0.301], P − mag 0.248, P − de 0.225 | yes / yes |

P ranks first among all 20 sets of every control family, for both instruments, for both B and R. The mean control shares
(v2, B) are rand 0.021, mag 0.013, de 0.028, and the largest single control set reaches 0.109 (a de set).

## What the numbers say, and what they do not

**The bridge sentence (M1, as declared):** on held-out verbs (half B), clamping the 2,986 selected neurons to their
values in the I-edit run removes 0.15 nats (19%, [15, 23]) of the v2 edit's T − I effect on *by*, against 1% for
activation-matched neurons and 3% for direct-effect-matched neurons. The converse, setting them to their T-run values
inside the I edit, restores 28% [23, 33].

Context for that number, all on the same rows:
- **All of MLPs 11 and 14 together** mediate 0.26 [0.21, 0.30] (restore 0.35). P is 0.73 [0.66, 0.81] of what the two
  layers carry (restore 0.81). Most of the edit's effect on *by* therefore runs outside these two MLPs at this token
  (other MLPs, attention, the residual stream). The 0.62 switch share is a share of the MLP 11 + 14 direct-effect switch.
  It is not a share of the edit's effect, and the text must not read it as one.
- **By layer:** MLP 14 alone 0.14, MLP 11 alone 0.06 (restore 0.21 / 0.13). The joint clamp (0.19) is near their sum.
- **Exploratory:** the top 10 neurons by half-A direct effect mediate 0.23 [0.20, 0.26], more than the full set; top 50
  give 0.26 and top 100 give 0.28. The rest of P partly cancels the top neurons. So a 10-neuron core carries the
  mediated part. This ranking was chosen on half A and is not a declared test.
- **New and lemma-fresh pairs (descriptive):** m(P) 0.22 [0.16, 0.29] on the 19 new pairs and 0.23 [0.15, 0.31] on the
  15 lemma-fresh pairs, P − mag 0.21 / 0.22. That is if anything above the half-B average (0.19), which speaks to the forking path
  flagged in the rules (layers and rule set on v and the original pairs).
- **u_md:** with the frozen v2 sets, the training-free direction gives the same mediated share (0.186 vs 0.188) and a
  similar switch share (0.70). This is convergence of two class-derived directions, not proof that no trained
  direction is involved.

**The shuffled-label direction (descriptive; the expectation in the rules was not met).** With v2's own shuffled-label
direction, the frozen set responds with sign agreement 0.97 and aligned I_n +0.020 [+0.019, +0.022]. That is 0.56 of
v2's response, with a switch 0.60 of v2's (+0.635 vs +1.057) and the same share of it (0.65). The rules expected "a
much smaller response". The likely reason is upstream of C4: in v2's own final, the shuffled-label direction already
recovers 0.45 of the held-out pc_frac against v2's 0.80 (ratio 0.57; IIA 0.25 vs 0.79). The set's response scales with
how much class difference the instrument moves at site 8. That fits downstream readers of a class signal, but it
means this check does not test whether the neurons are specific to v2. Two consequences:
1. In C4 the shuffled run is reported as "responds in proportion to the T − I difference each direction transmits",
   not as a specificity control.
2. For C1 it is a flag: a rank-1 direction trained on shuffled labels moves about half as much class information as
   v2 on held-out swaps. The paper's C1 controls are random directions (pc_frac ≈ 0). The shuffled-label number belongs
   beside them, and it connects to the pending random / shuffled-direction (illusion) check for Gemma and OLMo.

**Overlap with v's set (descriptive):** 1,918 of the 2,986 v2 neurons are in v's 2,782 (Jaccard 0.50; chance about 507).
Top 10 by switch: 9 shared; top 50: 37; top 100: 70.

## Consequences for the paper (from the rules' outcome table)

- N1–N2 hold: C4's neuron claim is restated with v2's numbers (2,986 neurons; switch share 0.62 [0.58, 0.66]). v's
  2,782 / 0.65 appear nowhere.
- U1–U2 hold: one sentence in §5, "the selected readers also respond to the training-free class contrast (switch share
  0.70; mediated share 0.19)".
- M1 holds, beyond mag and beyond de: the bridge sentence goes in §5, with the MLP 11 + 14 total (0.26) beside it so
  the 0.19 is not read as "most of the effect". The text may say the conjunction set matters beyond output coupling.
- No serial head → neuron → *by* claim. Heads (B7 / D10) remain deferred; until ported they appear at most as one
  labelled appendix sentence on v.
- Still on v's set, and to decide: the wh-gap reading (R5-C9), the Pile-text activations (R5-B7) and the natural
  necessity ablation (R5-B5, cut as a new run). Porting C9 and B7 to the v2 set is forward passes only. The top-10
  overlap is 9 / 10, so the Pile-text description of the top neurons largely carries over, but it was measured on v's
  ranking.
