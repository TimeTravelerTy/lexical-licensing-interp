# Lexical Licensing DAS Experiment Spec

## Behavioral Gate

Run `lp_readout` and `in_template_lp` on:

- `EleutherAI/pythia-1.4b`
- `EleutherAI/pythia-2.8b`
- `EleutherAI/pythia-6.9b`

Regimes:

- `head`
- `tail`
- `xtail`

Initial phenomena:

- `causative`
- `inchoative`
- `transitive`
- `intransitive`
- `drop_argument`
- controls: `passive_1`, `passive_2`, `principle_A_domain_3`

Use `/Users/tyronewhite/masters_research_code/blimp-rare/scripts/score_acceptability_methods.py`.
Generated commands should write outputs under this repo's `results/` directory,
so the sibling repos can remain read-only.

## Aligned Template Gate

Raw FreqBLiMP sentences are not used directly for interventions because they
vary in determiner length, auxiliary choice, tense, negation, and verb
tokenization. The first intervention deliverable is a tokenizer-verified
aligned template dataset.

The current template family is intentionally simple:

| frame target | prompt skeleton | expected next token |
| --- | --- | --- |
| object frame licensed | `In the scene, the technician will {verb}` | ` the` |
| no-object frame licensed | `In the scene, the glass will {verb}` | `.` |
| drop-object licensed | `In the scene, the artist will {verb}` | `.` |

The verifier must prove, for a chosen tokenizer:

- same prefix token count within a template;
- same verb-region token count for the retained subset;
- same decision-token position across retained items;
- target continuations are single tokens.

The aligned preview can include unverified candidates, but DAS and attribution
patching must consume only `alignment_status == "aligned"` records.

## Attribution Patching

Use aligned prompt pairs where the high-level variable is object/no-object frame
licensing. The metric should match the downstream DAS objective:

```text
log p(target_source | intervened base) - log p(target_base | intervened base)
```

Attribution patching is only a localization screen. Confirm top sites with exact
activation patching before training DAS.

## DAS

Train head-regime DAS on localized residual-stream sites. Validate on held-out
head verbs, then test transfer to tail and xtail.

Report:

- headline: ODDS/log-odds shift;
- secondary: IIA-style thresholded success;
- qualitative: raw judgment flips.

Controls:

- random rotation;
- shuffled labels;
- held-out verbs;
- passive/principle-A negative controls;
- degenerate-output/OOD checks.

