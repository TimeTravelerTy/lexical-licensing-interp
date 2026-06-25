# DAS Projection Diagnostic Plan

## Purpose

This is a validation check for the learned DAS direction `d`, not evidence for
the storage-vs-computation question.

Real-verb projection separation cannot distinguish:

- H1 lookup: `d` reads a stored lexical entry.
- H2 rule/computation: `d` reads a computed licensing value.

Both predict that real held-out verbs can separate along `h · d`. The nonce
version is the discriminating test for H1/H2. This diagnostic only asks whether
the current real-verb `d` behaves like a licensing instrument rather than a
plain object-continuation proxy.

## Primary Cell

The main check is the inchoative-good cell.

The script calibrates the arbitrary rank-1 sign using:

```text
causative_good_minus_bad = mean(d · h_causative_good) - mean(d · h_causative_bad)
```

Then it applies that orientation to:

```text
inchoative_good_minus_bad = mean(d · h_inchoative_good) - mean(d · h_inchoative_bad)
```

Interpretation:

- positive calibrated inchoative margin: `d` treats inchoative-good as licensed,
  despite the alternator/object-propensity pressure;
- negative or collapsed calibrated inchoative margin: `d` is behaving more like
  an object/no-object continuation proxy.

The contextwise version of the same check is the first robustness test. Generic
good/bad AUC and pair deltas are secondary diagnostics.

## Run Command

Run in the environment that has the verified v2 aligned JSONL, Pythia weights,
`torch`, and `transformers`:

```bash
python3 scripts/project_eval_on_das_subspace.py \
  --data data/aligned_templates_v2/lexical_licensing_v2_aligned.jsonl \
  --subspace results/das_v2_transfer/20260623-pythia14b-v2-das-head-to-low-l23-none-s17-verb_to_subject_anchor.subspace.pt \
  --site resid_post_layer_23 \
  --rank 1 \
  --subtasks causative,inchoative \
  --train-regimes head \
  --eval-regimes head,low \
  --project-split eval \
  --pairing balanced_pool \
  --seed 17 \
  --batch-size 16 \
  --device cuda \
  --dtype bfloat16 \
  --out-dir results/das_projection \
  --run-name 20260625-pythia14b-v2-l23-d-projection-eval
```

If the original trained subspace file is present, it can be used directly:

```text
results/das_v2_transfer/20260623-pythia14b-v2-das-head-to-low-l23-none-s17.subspace.pt
```

The `verb_to_subject_anchor` subspace file is also valid for `d`; that run loaded
the original l23 lexical subspace and only changed the evaluation-time injection
anchor.

## Outputs

- `*.validation_cell_summary.csv`: headline diagnostic.
- `*.projection_detail.csv`: one row per unique projected prompt/verb/context.
- `*.label_separation_summary.csv`: generic AUC and threshold sweeps.
- `*.pair_delta_summary.csv`: clean-corrupt movement along `d`.

Use `validation_cell_summary.csv` first. Treat all other tables as robustness or
debugging views.
