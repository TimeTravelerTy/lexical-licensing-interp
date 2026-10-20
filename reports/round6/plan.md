# Round 6: declared plan (paper 1: frequency and entrenchment)

Builds on rounds 1–5 (`../results_index.md`). Round 6 serves paper 1 (the
rare-verb deficit: where it is, what exposure explains it, whether it is
specific to the passive, and whether it can be restored or preempted).
Paper-2 items (LEACE necessity, gap-neuron mediation, human annotation of B7,
6.9B DAS) are parked.

Same rules as round 5: each part is written and committed before any of its
runs, with predictions and decision rules stated before the data exist;
where a prediction leans on earlier outputs, that is said. An outside
reviewer (Codex) reviews each part's draft and code before the runs.

Run order: 0, A1, A2, B3, C4, C5, A6, A7. Parts are declared in that order,
in groups, each group in its own commit before its runs.

Notation as in rounds 4–5 (`../round5/plan.md`): T / I = transitive
(passivizable, "good") / intransitive ("bad") verb of a pair; Δ_pass(V, c) =
log P(V | "The N was") − log P(V | "The N has") (full participle log-prob,
summed over subtokens, no BOS); L = Δ_pass(T) − Δ_pass(I) per pair; a_V =
u_V1 · Δh̄ (passive-slot affinity of a verb's first subtoken); site s =
residual output of layer s − 1; z = a verb's site-8 coordinate on the
active-trained DAS direction d. D11 = the round-5 expanded inventory (281
T/I pairs: Head 47, Tail 132, XTail 102; 62 held out;
`data/round5/verbs_expanded/pairs.csv`, loader `scripts/verbs_expanded.py`).
"Curated" = the 126 band-cross pairs of rounds 4–5. CIs are 95%, 2,000
draws, seed 17, unless stated otherwise. Layout: plan and reports in
`reports/round6/`, regenerable inputs in `data/round6/<part>/`, outputs in
`results/round6/<part>/`.


# Part 0: housekeeping (declared 2026-10-20)

No model runs, no predictions.

1. **Results index.** Round 5 is already listed in `../results_index.md`
   (rows R5-*). Add a column tagging each experiment as paper 1 (P1:
   frequency / entrenchment), paper 2 (P2: mechanism) or both, and list the
   parked paper-2 items. Round-6 rows are added when round 6 reports.
2. **Human data.** Check the format of the FreqBLiMP human study
   (`freq-blimp/human_validation_ui`; returned three-annotator export, the
   corrected merge of 2026-08-25) for `passive_1` / `passive_2`.
   - Checked before this declaration (format only, no outcome tabulated
     beyond response counts): it is **forced choice** (A / B / can't
     decide, with reasons), followed by a 1–5 naturalness rating of the
     intended acceptable sentence only. There are no per-sentence ratings.
   - So, per the round-6 rule, **accuracy by band only, descriptive**; no
     bad-item acceptance. Per paradigm × condition (Original, Head, Tail,
     XTail; 15 pairs × 3 annotators): correct among decided responses,
     can't-decide share, accuracy with can't-decide counted as half, items
     with 2-of-3 support, mean naturalness of the good sentence; Pythia-1.4B
     LP accuracy on the same released items (round-5 A4 unedited scores)
     beside it. Items resampled within cell for 95% CIs (15 items per cell,
     so these are wide). Script `scripts/analyze_human_passives.py`; report
     `human_passives.md`; per-item table (no annotator-level data)
     `data/round6/human/passive_items.csv`. The raw export stays outside the
     repo; its sha256 is recorded.
