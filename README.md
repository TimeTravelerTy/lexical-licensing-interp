# Lexical Licensing Interventions

This repo contains the intervention-facing layer for the FreqBLiMP lexical
licensing project. It treats the existing sibling repos as inputs:

- `/Users/tyronewhite/masters_research_code/freq-blimp`: final FreqBLiMP data.
- `/Users/tyronewhite/masters_research_code/blimp-rare`: scoring and analysis
  code, plus richer generated metadata files.

The immediate workflow is:

1. Run the Pythia behavioral gate on FreqBLiMP with `lp_readout` and
   `in_template_lp`.
2. Build rigid aligned lexical-licensing templates.
3. Verify Pythia tokenizer alignment before any attribution patching or DAS.
4. Use attribution patching to localize layers/sites.
5. Train DAS only on localized candidates.

See `docs/experiment_spec.md` for the concrete plan and command layout.

The real-verb passive frequency pilot is documented in
`docs/matched_passives.md`.

Later work (passive DAS rounds 2–7 and the Gemma-3-1B / OLMo-2-1B
replications) is indexed in `reports/README.md`; `reports/results_index.md`
lists every experiment with its key number and report.

The mechanism paper (ARR, Jan 2027) is planned in `docs/paper_plan.md`:
claims, status per model, where each result lives, work ordered by risk and
decision rules. Keep its status table current.

## Layout

- `scripts/`: flat; prefixes say the role (`build_*` make inputs under `data/`,
  `run_*` produce raw outputs under `results/`, `analyze_*` / `summarize_*` /
  `plot_*` write reports and figures). Shared helpers: `model_arch.py`,
  `verb_forms.py`, `verbs_expanded.py`.
- `data/`, `results/`, `reports/`: one folder per stage or round (`round4/` …,
  `gemma/`, `olmo/`). Large raw dumps are gitignored; their SHA-256s are in the
  run metadata or the reports that use them.
- `docs/`: early plans and the TSUBAME workflow (`docs/tsubame_workflow.md`).
