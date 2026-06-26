# V2 Whole-Pair LP Regime Oddity Analysis, 2026-06-26

## Question

Whole-pair LP accuracy is higher in the low regime than in the head regime:

| regime | subtask | n | mean good-bad LP | accuracy |
| --- | --- | ---: | ---: | ---: |
| head | causative | 440 | 4.654 | 0.893 |
| head | inchoative | 320 | 3.026 | 0.809 |
| low | causative | 280 | 5.506 | 0.989 |
| low | inchoative | 280 | 5.568 | 0.932 |

This is not the expected frequency effect. The main causes look like dataset
difficulty and subject sampling, not a real low-frequency advantage.

## Main Finding

The inchoative low advantage is mostly a subject-mix artifact.

| regime | subtask | raw accuracy | subject-class balanced accuracy | raw margin | subject-class balanced margin |
| --- | --- | ---: | ---: | ---: | ---: |
| head | causative | 0.893 | 0.883 | 4.654 | 4.610 |
| low | causative | 0.989 | 0.982 | 5.506 | 5.480 |
| head | inchoative | 0.809 | 0.837 | 3.026 | 3.435 |
| low | inchoative | 0.932 | 0.837 | 5.568 | 3.909 |

For inchoatives, subject-class balancing removes the accuracy gap: both head
and low become 0.837. The raw low score is higher because low inchoative has
many more inanimate-subject examples.

## Subject Mix

| regime | subtask | subject class | n | accuracy | mean good-bad LP |
| --- | --- | --- | ---: | ---: | ---: |
| head | inchoative | manual animate | 192 | 0.698 | 1.390 |
| head | inchoative | manual inanimate | 128 | 0.977 | 5.480 |
| low | inchoative | manual animate | 56 | 0.679 | 1.143 |
| low | inchoative | manual inanimate | 224 | 0.996 | 6.674 |

So the model behaves as expected within subject class: animate-subject
inchoatives are harder, inanimate-subject inchoatives are much easier. The
regime comparison is currently contaminated by how often each subject class was
sampled.

Examples of hard animate-subject inchoatives:

```text
G: During the check, the doctor will fray.
B: During the check, the doctor will admonish.

G: Tomorrow, the doctor will bake.
B: Tomorrow, the doctor will notify.

G: In the lab, the teacher will dim.
B: In the lab, the teacher will require.
```

These are formally in the intended frame, but the subject makes the good
sentence semantically odd. That lowers full-sentence LP for reasons orthogonal
to lexical licensing.

## Causative Difference

Subject balancing does not remove the causative gap. The remaining difference
comes from verb-inventory difficulty.

Head causative bad verbs include very common intransitives such as `appear`,
`disappear`, and `arrive`. These are ungrammatical with a direct object, but
their high standalone probability can still make the full bad sentence
competitive under raw summed LP.

Worst head causative bad verbs:

| bad verb | n | accuracy | mean good-bad LP |
| --- | ---: | ---: | ---: |
| disappear | 20 | 0.600 | 1.163 |
| appear | 20 | 0.650 | 1.327 |
| arrive | 20 | 0.700 | 1.879 |
| lie | 20 | 0.800 | 2.443 |
| vanish | 20 | 0.850 | 2.533 |

Low causative bad verbs are mostly supplemental and often easier:

```text
hibernate, abscond, shiver, hobble, dither, capitulate, frolic, tremble
```

The low causative failure count is only 3/280, compared with 47/440 for head.

## Frequency Gap

The mean good-minus-bad Zipf gap is not enough to explain the full pattern:

| regime | subtask | mean good Zipf - bad Zipf |
| --- | --- | ---: |
| head | causative | 0.085 |
| head | inchoative | 0.176 |
| low | causative | 0.090 |
| low | inchoative | 0.349 |

Low inchoative has a bigger good-bad frequency gap, which may contribute, but
the subject-class reweighting shows the raw accuracy gap is primarily subject
composition.

## Recommendation

Do not interpret the low > head whole-pair result as a meaningful frequency
effect. Treat this run as a behavioral sanity check that full-sentence scoring
contains the contrast.

Before using whole-pair LP as a frequency comparison or as a DAS objective,
regenerate the dataset with fixed subject schedules:

- each `regime/subtask/context_schema` should receive the same small subject set;
- subject choice should remain independent of side and verb metadata;
- report animate/inanimate subject breakdowns and subject-balanced accuracy;
- consider a length-normalized or continuation-only LP variant as a secondary
  check, because full-sentence LP includes verb prior as well as frame fit.
