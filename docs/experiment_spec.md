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
- same decision-token position within each retained template/regime/subtask
  group;
- target continuations are single tokens.

Important: the causal intervention site should be a named region, not a global
absolute index. The intended site is the final token of the prompt, i.e. the
verb's final subtoken. This is robust to head/tail/xtail token fragmentation as
long as the implementation passes per-example `verb_final_subtoken` locations to
the patching/DAS code. Absolute decision indices may differ across frequency
bands; this is acceptable for analysis grouped by named region, but not for code
that assumes one global token index.

The aligned preview can include unverified candidates, but DAS and attribution
patching must consume only `alignment_status == "aligned"` records.

## Current Template Status

The current template file is a preview and behavioral-proxy scaffold, not the
final causal dataset. Its strengths:

- fixed local decision format;
- single-token Pythia decision targets;
- balanced good/bad labels within each fixed prefix/template;
- explicit aligned/rejected tokenization status.

Known limitations before DAS:

- `the` versus `.` is a proxy for object-frame expectation, not the full frame
  distribution. A later behavioral analysis should compare this against grouped
  continuations such as object introducers (` the`, ` a`, ` an`, ` it`, ` them`)
  versus sentence-ending/adverbial continuations (`.`, ` yesterday`, ` quietly`)
  where tokenization permits.
- semantic fit is weak with fully fixed subjects such as `the glass` or
  `the technician`. For causal claims, either choose per-verb subject nouns using
  the FreqBLiMP semantic machinery or report relative intervention shifts rather
  than absolute grammaticality scores.
- subject/prefix leakage must be ruled out with controls. The final DAS dataset
  should include prefix-only and shuffled-verb baselines, and should train/eval
  only on splits where each prefix/template has both target labels.

## Attribution Patching

Use aligned prompt pairs where the high-level variable is object/no-object frame
licensing. The metric should match the downstream DAS objective:

```text
log p(target_source | intervened base) - log p(target_base | intervened base)
```

Attribution patching is only a localization screen. Confirm top sites with exact
activation patching before training DAS.

Minimum controls before declaring a localization:

- prefix-only baseline: score target prediction from the prefix without the
  target verb signal, or with a neutral/dummy verb;
- shuffled verb-label baseline within the same prefix/template;
- exact activation patching at the top attribution sites;
- held-out verb split by lemma, not by surface item.

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
- prefix/subject-only controls;
- passive/principle-A negative controls;
- degenerate-output/OOD checks.

## Recommended Next Dataset Version

Build `v2` as a causal dataset, not just a next-token proxy:

1. Keep the prompt-final verb design, but define the intervention location as
   `verb_final_subtoken` per example.
2. Use FreqBLiMP metadata to choose semantically compatible subjects/objects per
   verb where possible.
3. Preserve balance: every prefix/context family must contain both object and
   no-object target labels, so the subject or fixed prefix cannot predict the
   answer.
4. Include explicit control rows:
   - `prefix_only`;
   - `shuffled_label`;
   - `random_verb_same_frequency_band`;
   - passive/principle-A negative controls.
5. Treat drop-argument as its own contrast. Do not assume it shares a site with
   causative/inchoative until attribution patching says so.
