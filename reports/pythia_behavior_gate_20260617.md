# Pythia Behavioral Gate, 2026-06-17

Remote output root:
`/gs/bs/tga-sip_arase/tyrone/blimp-rare-evals/20260617-pythia-gate`

The TSUBAME array job completed 9 tasks:

- 3 regimes: `head`, `tail`, `xtail`
- 3 models: `EleutherAI/pythia-1.4b`, `EleutherAI/pythia-2.8b`,
  `EleutherAI/pythia-6.9b`
- 8 phenomena
- 2 scoring methods: `nll`, `in_template_lp`

Outputs observed:

- score files: 144
- manifests: 9
- fatal error signatures: none found in logs

## Core Lexical Phenomena

Cells show accuracy as `nll / in_template_lp`.

| Model | Regime | Causative | Inchoative | Transitive | Intransitive | Drop argument |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1.4B | head | 0.803 / 0.798 | 0.864 / 0.844 | 0.718 / 0.693 | 0.673 / 0.626 | 0.750 / 0.745 |
| 1.4B | tail | 0.789 / 0.774 | 0.819 / 0.813 | 0.709 / 0.721 | 0.701 / 0.658 | 0.543 / 0.546 |
| 1.4B | xtail | 0.710 / 0.699 | 0.716 / 0.740 | 0.664 / 0.684 | 0.653 / 0.619 | 0.669 / 0.635 |
| 2.8B | head | 0.796 / 0.763 | 0.842 / 0.860 | 0.696 / 0.700 | 0.678 / 0.671 | 0.727 / 0.752 |
| 2.8B | tail | 0.792 / 0.786 | 0.829 / 0.842 | 0.700 / 0.680 | 0.720 / 0.710 | 0.550 / 0.598 |
| 2.8B | xtail | 0.656 / 0.649 | 0.720 / 0.738 | 0.666 / 0.634 | 0.662 / 0.682 | 0.638 / 0.645 |
| 6.9B | head | 0.799 / 0.767 | 0.854 / 0.853 | 0.691 / 0.642 | 0.678 / 0.675 | 0.725 / 0.752 |
| 6.9B | tail | 0.741 / 0.743 | 0.800 / 0.830 | 0.669 / 0.652 | 0.698 / 0.684 | 0.509 / 0.515 |
| 6.9B | xtail | 0.675 / 0.653 | 0.729 / 0.728 | 0.670 / 0.650 | 0.631 / 0.640 | 0.660 / 0.670 |

## Readout

The model-size pattern is qualitatively similar across 1.4B, 2.8B, and 6.9B.
There is no clear reason to pay the larger-model cost for the first localization
pass.

Recommended next step:

1. Use Pythia 1.4B for attribution patching.
2. Prioritize causative/inchoative first. These are the strongest and most
   stable lexical-licensing signals across regimes.
3. Treat transitive/intransitive as secondary. They are above chance but weaker.
4. Keep drop-argument separate. It behaves differently and is near chance for
   tail in all three model sizes.
5. Do not train DAS on the current fixed-subject preview templates as a final
   causal claim. Use them only for infrastructure/proxy localization until the
   `v2` semantic-fit and prefix-control dataset is built.
