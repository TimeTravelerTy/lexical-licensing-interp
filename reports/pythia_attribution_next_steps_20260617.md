# Pythia 1.4B Attribution Patching Next Steps

Run order:

1. `primary`: `causative,inchoative` (strongest and most stable behavioral gate signal).
2. `secondary`: `transitive,intransitive` (above chance but weaker behavioral gate signal).
3. `drop_argument`: `drop_argument` (separate contrast; tail behavior is weak/near chance).

Expected matched-pair scope:

- `primary`: 287 aligned rows, 80 matched prompt pairs, 160 directed pairs (127 unpaired aligned side groups).
- `secondary`: 297 aligned rows, 85 matched prompt pairs, 170 directed pairs (127 unpaired aligned side groups).
- `drop_argument`: 142 aligned rows, 36 matched prompt pairs, 72 directed pairs (70 unpaired aligned side groups).

All runs use aligned rows only, both patch directions, all regimes, and the per-example `verb_final_subtoken` anchor.
The output files are detail CSV, summary CSV, and manifest JSON under `results/attribution_patching/`.

## Commands

### Attribution Screen

```bash
cd '/Users/tyronewhite/masters_research_code/lexical_licensing_interp'
python3 '/Users/tyronewhite/masters_research_code/lexical_licensing_interp/scripts/run_pythia_attribution_patching.py' --data '/Users/tyronewhite/masters_research_code/lexical_licensing_interp/data/aligned_templates/lexical_licensing_aligned.jsonl' --model 'EleutherAI/pythia-1.4b' --subtasks 'causative,inchoative' --regimes head,tail,xtail --directions good_to_bad,bad_to_good --batch-size 8 --device 'cuda' --dtype 'bfloat16' --out-dir '/Users/tyronewhite/masters_research_code/lexical_licensing_interp/results/attribution_patching' --run-name '20260617-pythia14b-ap-primary'
python3 '/Users/tyronewhite/masters_research_code/lexical_licensing_interp/scripts/run_pythia_attribution_patching.py' --data '/Users/tyronewhite/masters_research_code/lexical_licensing_interp/data/aligned_templates/lexical_licensing_aligned.jsonl' --model 'EleutherAI/pythia-1.4b' --subtasks 'transitive,intransitive' --regimes head,tail,xtail --directions good_to_bad,bad_to_good --batch-size 8 --device 'cuda' --dtype 'bfloat16' --out-dir '/Users/tyronewhite/masters_research_code/lexical_licensing_interp/results/attribution_patching' --run-name '20260617-pythia14b-ap-secondary'
python3 '/Users/tyronewhite/masters_research_code/lexical_licensing_interp/scripts/run_pythia_attribution_patching.py' --data '/Users/tyronewhite/masters_research_code/lexical_licensing_interp/data/aligned_templates/lexical_licensing_aligned.jsonl' --model 'EleutherAI/pythia-1.4b' --subtasks 'drop_argument' --regimes head,tail,xtail --directions good_to_bad,bad_to_good --batch-size 8 --device 'cuda' --dtype 'bfloat16' --out-dir '/Users/tyronewhite/masters_research_code/lexical_licensing_interp/results/attribution_patching' --run-name '20260617-pythia14b-ap-drop_argument'
```

### Exact Patching Confirmation

Run this after the primary attribution summary exists. It confirms the top residual-stream sites with real activation replacement.

```bash
cd '/Users/tyronewhite/masters_research_code/lexical_licensing_interp'
python3 '/Users/tyronewhite/masters_research_code/lexical_licensing_interp/scripts/run_pythia_exact_patching.py' --data '/Users/tyronewhite/masters_research_code/lexical_licensing_interp/data/aligned_templates/lexical_licensing_aligned.jsonl' --model 'EleutherAI/pythia-1.4b' --subtasks causative,inchoative --regimes head,tail,xtail --directions good_to_bad,bad_to_good --summary-csv '/Users/tyronewhite/masters_research_code/lexical_licensing_interp/results/attribution_patching/20260617-pythia14b-ap-primary.summary.csv' --top-k 6 --batch-size 8 --device 'cuda' --dtype 'bfloat16' --out-dir '/Users/tyronewhite/masters_research_code/lexical_licensing_interp/results/exact_patching' --run-name '20260617-pythia14b-ap-primary-exact'
```

### TSUBAME Job Scripts

Generate temporary SGE scripts after the repo is present on TSUBAME. Copy them to the remote `jobs/` directory and submit with `qsub -g tga-sip_arase`.

```bash
cd '/Users/tyronewhite/masters_research_code/lexical_licensing_interp'
python3 '/Users/tyronewhite/masters_research_code/lexical_licensing_interp/scripts/make_tsubame_v1_job.py' primary_attribution
python3 '/Users/tyronewhite/masters_research_code/lexical_licensing_interp/scripts/make_tsubame_v1_job.py' primary_exact
python3 '/Users/tyronewhite/masters_research_code/lexical_licensing_interp/scripts/make_tsubame_v1_job.py' secondary_attribution
python3 '/Users/tyronewhite/masters_research_code/lexical_licensing_interp/scripts/make_tsubame_v1_job.py' drop_argument_attribution
```

## Interpretation Gate

- Treat these as localization screens only.
- Promote a site to exact patching only if it is consistent across direction and at least the head/tail regimes.
- Do not pool `drop_argument` with causative/inchoative unless its top sites independently match.
- Do not train DAS on this fixed-subject scaffold for final claims; use it to debug localization and the patching infrastructure.
