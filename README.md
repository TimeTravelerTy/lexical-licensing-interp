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

