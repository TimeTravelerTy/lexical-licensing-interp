# OLMo-2-1B: nonce verbs and DAS controls (rules, declared Nov 23, 2026, before any run)

`finals_rules.md` §4 left these for their own rules after the finals. Both follow the Gemma procedure unchanged (`reports/gemma/exploratory.md` §8–10 for nonce, `reports/gemma/das_controls.md` for controls), with OLMo's sites mapped by the same rules. Exploratory, as Gemma's were; the paper reports them as curves with the declared onset reading.

## Nonce verbs

**Pipeline:** Pythia's two nonce analyses, unchanged, on the same 80 lemmas, metrics, bootstrap (lemma bootstrap, 2000 draws, seed 17) and decision rules.
- Round-3 C8 (`run_nonce_passive.py`, `analyze_nonce_passive.py`): z along d per site; matched / mismatched / balanced contrasts; log P(" by"), PREP without *by*, O, "."; the causal swap; decisions (a) context-sensitive separation (Δ_matched) and (b) verb-specific (Δ_balanced).
- Round-4 C8 / C9 (`run_nonce_cues.py`, `analyze_nonce_cues.py`): inflection mismatch and cue contrasts.

**Stimuli:** rebuilt with the unchanged builders into `data/olmo/nonce/{nonce_passive,nonce_cues}` with OLMo's DAS items (the z ruler; 29 pairs, none dropped) and OLMo's passive-test items.

**Sites.** Projection at **every site 0–8** (the plan's range) plus the sites mapped from Pythia's by the rule that picked them (Gemma §9 table):

| Pythia site | why | rule | OLMo site |
|---:|---|---|---:|
| 17 | primary | frozen rule | 13 |
| 12 | declared secondary | first site with best active IIA ≈ 0.9 | 11 |
| 8 | declared secondary, decision, causal swap | first site with best active IIA ≈ 0.8 | 9 |
| 6 | fill-in, decision, causal swap | IIA-matched (0.669 → OLMo 5, 0.653) | 5 |
| 4 | fill-in | IIA-matched (0.454 → OLMo 3, 0.434) | 3 |
| 10 / 14 / 16 | fill-in | IIA-matched | 9 / 11 / 12 |

- Projection sites: 0–8, 9, 11, 12, 13.
- **Decision and causal-swap sites: 5 and 9** (Pythia's 6 and 8), plus 3 (Pythia's 4), as Gemma added its site 3.
- Round-4 z columns: 3, 5, 9, 11, 13 (Pythia 4 / 6 / 8 / 12 / 17).
- Bases: the OLMo finals at those sites (`results/olmo/das/transfer/final_strict_site*`, `final_strict` = 13), exported with `export_bases.py`'s conversion to `results/olmo/das/nonce/bases_rank1.npz`; bases already in `transfer/bases_rank1.npz` must be bit-identical.

**Readings.**
- The declared decisions (a) and (b) at sites 3, 5 and 9, as in Pythia and Gemma.
- **Onset by depth** (the paper's "shallow, context-derived" claim, as §10a): the first site in 0–8 where (a) holds, and the first where (b) holds, as a relative depth, next to Pythia (a: 3 = 0.12; b: 6 = 0.25) and Gemma (both 6 = 0.23). The claim replicates in OLMo if (a) holds by relative depth ≤ 0.375 (site ≤ 6) and stays "yes" at every later projection site. Otherwise the curve is reported as it is.

**Sensitivity splits** (OLMo's own tokenizer): lemmas whose -ed form's last piece is inflection-only ("ed") vs carries stem letters, and lemmas with the same -ed token count as Pythia. Computed in the job from `results/gemma/nonce_tokenization/forms.csv` (the -ed words) with OLMo's tokenizer.

**Checks:** the runners' token-suffix check; the bases bit-identity check; no traceback.

## DAS controls (active task)

`run_das_controls.py active --sweep-dir results/olmo/das/sweep_strict`: per site 0–16, 20 untrained random unit directions through the same interchange on the sweep's held-out swaps (split 0, 5 folds), the unpatched IIA of the same swaps, and a logistic-regression T vs I probe. Trained = the sweep's best-epoch held-out IIA.

- **Reading:** at each site, trained IIA vs the random directions' 95th percentile; the probe accuracy shows the class is linearly present where DAS IIA is low (as in Gemma). Reported next to Pythia and Gemma by relative depth.
- The passive-task controls need a passive (reverse) sweep, which OLMo does not have; they are not run.
