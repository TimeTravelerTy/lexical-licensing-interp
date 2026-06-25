# DAS Projection Diagnostic Results, 2026-06-25

## Setup

- model: `EleutherAI/pythia-1.4b`
- data: `data/aligned_templates_v2/lexical_licensing_v2_aligned.jsonl`
- split: eval only, `head,low`
- subtasks: `causative,inchoative`
- subspaces: head-trained rank-1 DAS directions at layers 18, 20, and 23
- TSUBAME job: `8010961` (`das_proj_eval`)

This diagnostic validates the direction as an instrument. It does not
distinguish lexical lookup from online computation, because real-verb
separation is predicted by both.

## Headline Result

The current head-trained directions separate eval examples very strongly, but
the separation is aligned with the expected next-token object/no-object target,
not with a stable licensed/unlicensed orientation.

The sign below is calibrated so that the causative good-minus-bad margin is the
reference orientation. A licensing-aligned direction should then give a positive
calibrated inchoative good-minus-bad margin. It is negative at every tested
site and regime.

| layer | regime | causative good-bad | inchoative good-bad | calibrated inchoative margin | calibrated object-target margin | inchoative orientation-free AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 18 | head | -14.065 | 17.344 | -17.344 | 15.284 | 1.000 |
| 18 | low | -17.862 | 15.203 | -15.203 | 16.532 | 0.979 |
| 20 | head | -16.120 | 19.605 | -19.605 | 17.416 | 0.999 |
| 20 | low | -20.834 | 17.151 | -17.151 | 18.992 | 0.974 |
| 23 | head | -20.609 | 21.918 | -21.918 | 21.095 | 1.000 |
| 23 | low | -24.660 | 17.169 | -17.169 | 20.915 | 0.947 |

## Interpretation

This fails the proposed `inchoative_good` instrument check. The learned `d`
does validate causative verb separation, but that is the easy reference cell:
causative-good is both licensed and object-expecting. The useful pressure test
is inchoative-good, where the licensed answer is no object. There, the direction
flips relative to causative-good and follows the no-object target.

So the current projection evidence says:

- `d` is a strong target/proxy axis for ` the` versus `.`;
- `d` is not cleanly validated as a licensing-goodness axis on real verbs;
- this remains a d-validation result, not an H1/H2 storage-vs-computation
  result.

## Local Outputs

The copied result CSVs are under `results/das_projection/` with prefix:

```text
20260625-pythia14b-v2-l{18,20,23}-d-projection-eval
```

Use `*.validation_cell_summary.csv` as the primary table. The label-separation
and pair-delta tables are secondary debugging views.
