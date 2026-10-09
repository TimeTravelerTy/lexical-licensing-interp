# Round 4, D12. Translation gain per pair

Spec: `plan.md`, D12. Run: `run_translation_gain.py` (fp32, natural pass); analysis: `analyze_translation_gain.py`. 64 primary pairs. TG = direct effect of MLPs 11–17 on the centered " by" logit (through the final-LN scale), good − bad, mean over contexts. gain = TG / site-8 z gap (pairs with z gap ≥ 0.1); TG_res = TG − -0.215 × z gap. CIs: pair bootstrap (2,000 draws).

Means: TG +0.945; z gap +0.363; gain +3.397. Per MLP (mean TG): MLP11 +0.002, MLP12 -0.031, MLP13 +0.068, MLP14 +0.551, MLP15 +0.100, MLP16 +0.127, MLP17 +0.128.

| Measure | split-half R | slope on participle Zipf |
|---|---:|---|
| TG | 0.998 | -0.169 [-0.800, +0.402] (n 64) |
| gain | 0.997 | -0.805 [-2.252, +0.693] (n 61) |
| TG_res | 0.998 | -0.155 [-0.751, +0.416] (n 64) |
| by logit gap (all components) | 0.996 | +0.309 [-0.245, +0.926] (n 64) |
| z gap (site 8) | 0.999 | +0.065 [+0.039, +0.101] (n 64) |

| Measure | Behaviour | pairs | r observed [95% CI] | r disattenuated [95% CI] | verdict |
|---|---|---:|---|---|---|
| TG | single-prompt " by" preference | 64 | +0.65 [+0.48, +0.79] | +0.65 [+0.48, +0.79] | significant positive link |
| TG | released by margin (task 0) | 59 | +0.57 [+0.35, +0.75] | +0.57 [+0.35, +0.75] | significant positive link |
| TG | curated passive_2 LP margin | 59 | +0.17 [-0.12, +0.45] | +0.17 [-0.12, +0.46] | no detectable link; moderate not excluded |
| gain | single-prompt " by" preference | 61 | +0.52 [+0.35, +0.66] | +0.52 [+0.35, +0.66] | significant positive link |
| gain | released by margin (task 0) | 56 | +0.47 [+0.28, +0.64] | +0.47 [+0.28, +0.64] | significant positive link |
| gain | curated passive_2 LP margin | 56 | +0.23 [-0.08, +0.52] | +0.23 [-0.08, +0.52] | no detectable link; moderate not excluded |
| TG_res | single-prompt " by" preference | 64 | +0.65 [+0.48, +0.79] | +0.65 [+0.48, +0.79] | significant positive link |
| TG_res | released by margin (task 0) | 59 | +0.57 [+0.35, +0.75] | +0.57 [+0.35, +0.75] | significant positive link |
| TG_res | curated passive_2 LP margin | 59 | +0.17 [-0.12, +0.45] | +0.17 [-0.12, +0.46] | no detectable link; moderate not excluded |
| by logit gap (all components) | single-prompt " by" preference | 64 | +0.78 [+0.65, +0.87] | +0.78 [+0.65, +0.87] | significant positive link |
| by logit gap (all components) | released by margin (task 0) | 59 | +0.79 [+0.67, +0.88] | +0.79 [+0.68, +0.88] | significant positive link |
| by logit gap (all components) | curated passive_2 LP margin | 59 | +0.23 [-0.05, +0.46] | +0.23 [-0.05, +0.46] | no detectable link; moderate not excluded |
| z gap (site 8) | single-prompt " by" preference | 64 | +0.10 [-0.15, +0.35] | +0.10 [-0.15, +0.35] | no detectable link; moderate not excluded |
| z gap (site 8) | released by margin (task 0) | 59 | +0.02 [-0.27, +0.30] | +0.02 [-0.27, +0.30] | no detectable link; moderate not excluded |
| z gap (site 8) | curated passive_2 LP margin | 59 | +0.04 [-0.19, +0.28] | +0.04 [-0.19, +0.28] | no moderate link |

TG against the d gap as a predictor of behaviour (paired bootstrap):

| Behaviour | pairs | r(TG) | r(z gap) | r(TG) − r(z gap) [95% CI] | partial r(TG, behaviour | z gap) [95% CI] |
|---|---:|---:|---:|---|---|
| single-prompt " by" preference (part-whole: TG is a component of this readout) | 64 | +0.65 | +0.10 | +0.54 [+0.25, +0.84] | +0.65 [+0.48, +0.79] |
| released by margin (task 0) | 59 | +0.57 | +0.02 | +0.55 [+0.20, +0.88] | +0.57 [+0.37, +0.73] |
| curated passive_2 LP margin | 59 | +0.17 | +0.04 | +0.13 [-0.31, +0.55] | +0.18 [-0.11, +0.46] |

## Declared decisions

- **TG tracks frequency:** no (slope -0.169 [-0.800, +0.402]).
- **Frequency dependence of TG beyond the d gap** (Zipf coefficient in TG ~ z gap + Zipf, refit per draw): no (-0.179 [-0.798, +0.459]).
- **TG predicts behaviour beyond the d gap** (paired Δr and partial r both above 0; part-whole readout excluded): ['by_margin_released'].
- Reading (observational): no evidence of a frequency dependence of translation beyond the d gap (this does not show there is none).

