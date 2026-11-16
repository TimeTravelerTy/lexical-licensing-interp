# OLMo-2-1B final runs, passive depth curve and transfer: rules, declared before running (2026-11-16)

Follows `rules.md` (model, BOS, precision, gates) and `das_sweep.md` (the sweep, frozen site 13, 3 epochs). Paper
plan task 2 (`docs/paper_plan.md`). Nothing below has been run. Branch `paper`.

Everything mirrors the Gemma replication: the same runners, builders and analyses, with
`--model allenai/OLMo-2-0425-1B` and the `olmo2` map defaults (bf16, fp32 unembedding, no BOS).

- Outputs: `results/olmo/das/transfer/` (T) and `data/olmo/transfer/` (D).
- Reports: `reports/olmo/transfer/` and `reports/olmo/depth_curve/`.

## 1. Sites and configs

**Primary: site 13** (frozen rule; `results/olmo/das/frozen_config.json`, 3 epochs). The final run picks the rank by
its own rule, as for Pythia and Gemma.

**Declared secondary sites**, by the rules used for Pythia and Gemma, read off `results/olmo/das/sweep_by_site.csv`
(best-epoch fold-mean held-out cross IIA, last site 16 excluded):

| Rule | Pythia | Gemma | OLMo |
|---|---|---|---|
| First site with IIA ≥ 0.8 | 8 | 5 | **9** (0.818; site 8 is 0.794) |
| First site with IIA ≥ 0.9 | 12 | 8 | **11** (0.940; site 10 is 0.873) |
| Earliest site within 0.05 of the max IIA | 12 | 12 | **12** (0.978 ≥ 0.997 − 0.05; site 11 is 0.940) |

**Depth curve:** every other site 0–15 gets a final run and the passive test only.

**Configs:** `scripts/make_site_configs.py` (rank 1; epochs by the frozen epoch rule: the smallest epoch within 0.01
of the site's best).
- It reproduces all 15 stored Gemma transfer configs' epochs from Gemma's sweep (checked before this declaration).
- OLMo epochs it gives, sites 0–15: 1, 3, 2, 4, 4, 4, 3, 3, 5, 4, 4, 4, 4, 3, 2, 2. Site 13 gives 3, the frozen
  config.

## 2. Runs

1. **Finals:** `run_das_round2.py final` at site 13 (frozen config, out `T/final_strict`) and at every other site 0–15
   (`T/frozen_config_site{s}.json`, out `T/final_strict_site{s}`). Same splits, folds and seeds as Pythia and Gemma.
2. **Transfer chain** (Gemma's `transfer_chain.sh`, stages prep / main / fg), with PRIM = 13, SECS = 9, 11, 12 and
   S = 9, 11, 12, 13:
   - prep: export bases, `frozen_config_final.json`, δ_s by the TOST rule, and build the passive-test,
     construction and filler-gap inputs;
   - main: passive test, constructions (OR, TC), in-range patch;
   - fg: filler-gap gates and patching (embedded and matrix wh).
3. **Passive depth curve:**
   - `analyze_passive_test.py delta` at the remaining sites;
   - `run_passive_test.py` and `analyze_passive_test.py passive` at sites 0–8, 10, 14, 15 (Gemma's
     `g3_pass_gemma.sh`);
   - the plan is rebuilt in a temp dir and checked equal to `D/passive_test` (plan hash, prompts, items).
4. **Not in this declaration** (each gets its own rules after the finals): nonce at sites 0–8, DAS controls (random /
   shuffled), the split-edit test, LEACE.

**Gate:** finals reproduce the sweep. The held-out IIA of each final at split 0 is within 0.03 of the sweep's
chosen-epoch IIA (fold mean). A site that misses is reported, not dropped.

## 3. Replication criteria (paper plan task 5)

Declared here for the transfer and staging claims. The same criteria are applied to Pythia, Gemma and OLMo, from
the stored and new tables, and each model gets pass / fail per criterion.

- **T1. Passive *by* transfer:** D(by) classed RISE at both declared secondary sites from the 0.8 and 0.9 rules
  (Pythia 8 / 12, Gemma 5 / 8, OLMo 9 / 11).
- **T2. Constructions:** at the same two sites, D(O) classed RISE for both OR and TC (T − I donors, bad bases).
- **T3. Filler-gap:** the gates pass, and the in-range patch is classed licensing-like at one or more of the
  declared sites, in one or more of the two experiments (embedded, matrix).
- **S1. Staging (v):** one or more sites classed abstract (O NO RISE, *by* RISE). Every site classed surface (O
  RISE, *by* not RISE) is deeper than every abstract site. Read over the depth curve.
  - The same criterion with u_md (and the split-edit on-axis piece) is applied when those runs exist; it is the
    one the paper uses if the split-edit decision rule fires.
- **Depth mapping:** reported by both relative depth and absolute layer index. No criterion depends on the mapping.

**Paper rule:** OLMo "replicates" the transfer claim if T1 and T2 pass. T3 and S1 are reported per model. If T1 or
T2 fails, the paper plan rule applies: "Pythia and Gemma, with a partial third-family check".
