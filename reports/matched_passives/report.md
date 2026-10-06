# Pythia 1.4B matched-context passive pilot

Positive margins favor the passivizable verb. The lexical unit is one verb pair; eight shared contexts are averaged within each pair. Intervals bootstrap verb pairs.

## Regime summaries

| Paradigm | Band | Verb pairs | Accuracy | Mean log-probability margin | 95% interval for margin |
| --- | --- | ---: | ---: | ---: | ---: |
| passive_1 | head | 11 | 0.841 | 3.795 | [2.064, 5.672] |
| passive_1 | tail | 15 | 0.992 | 5.970 | [4.704, 7.310] |
| passive_1 | xtail | 12 | 0.875 | 4.640 | [2.868, 6.499] |
| passive_2 | head | 11 | 0.989 | 3.443 | [2.667, 4.232] |
| passive_2 | tail | 15 | 0.867 | 4.148 | [2.589, 5.621] |
| passive_2 | xtail | 12 | 0.865 | 4.269 | [2.721, 5.786] |

## Head minus xtail sensitivity

| Paradigm | Metric | Difference | 95% interval | Leave-one-out range | Sign changes |
| --- | --- | ---: | ---: | ---: | ---: |
| passive_1 | mean_margin | -0.845 | [-3.375, 1.685] | [-1.456, -0.322] | 0/23 |
| passive_1 | accuracy | -0.034 | [-0.212, 0.152] | [-0.091, 0.025] | 1/23 |
| passive_2 | mean_margin | -0.826 | [-2.523, 0.873] | [-1.276, -0.383] | 0/23 |
| passive_2 | accuracy | 0.124 | [0.008, 0.280] | [0.057, 0.136] | 0/23 |

## Individual verb pairs

Each row is the mean across the eight contexts. See `per_verb_pair.csv` for participle frequencies, token counts, and verb/suffix contributions.

### passive_1

| Band | Passivizable / intransitive | Accuracy | Mean margin |
| --- | --- | ---: | ---: |
| head | accompany / emerge | 1.000 | 9.907 |
| head | avoid / struggle | 1.000 | 4.189 |
| head | blame / laugh | 0.875 | 1.327 |
| head | bless / complain | 0.250 | -0.887 |
| head | condemn / testify | 0.625 | 1.126 |
| head | confront / remark | 1.000 | 5.580 |
| head | defend / reply | 1.000 | 1.871 |
| head | monitor / compete | 0.875 | 4.138 |
| head | punish / proceed | 1.000 | 6.373 |
| head | remind / respond | 0.625 | 0.997 |
| head | welcome / lie | 1.000 | 7.122 |
| tail | befriend / reappear | 1.000 | 6.778 |
| tail | berate / disembark | 1.000 | 6.545 |
| tail | chastise / collude | 1.000 | 7.031 |
| tail | defame / proliferate | 1.000 | 4.057 |
| tail | delude / mutate | 0.875 | 1.999 |
| tail | deride / dwindle | 1.000 | 7.302 |
| tail | flog / recede | 1.000 | 4.834 |
| tail | interrogate / emigrate | 1.000 | 11.360 |
| tail | laud / conspire | 1.000 | 5.023 |
| tail | malign / stagnate | 1.000 | 4.247 |
| tail | pacify / gush | 1.000 | 3.722 |
| tail | patronize / balk | 1.000 | 1.985 |
| tail | scrutinize / wane | 1.000 | 6.638 |
| tail | terrorize / diverge | 1.000 | 9.943 |
| tail | vilify / languish | 1.000 | 8.084 |
| xtail | cajole / hitchhike | 1.000 | 6.917 |
| xtail | discomfit / debark | 1.000 | 4.369 |
| xtail | edify / flounce | 1.000 | 4.777 |
| xtail | flummox / ossify | 1.000 | 5.693 |
| xtail | importune / putrefy | 0.750 | 3.338 |
| xtail | overawe / vegetate | 1.000 | 5.255 |
| xtail | reprove / remonstrate | 0.250 | -0.748 |
| xtail | satirize / decamp | 0.750 | 0.571 |
| xtail | stupefy / intermarry | 1.000 | 10.395 |
| xtail | tantalize / toddle | 1.000 | 4.214 |
| xtail | titillate / calve | 0.750 | 1.208 |
| xtail | upbraid / burgeon | 1.000 | 9.692 |

### passive_2

| Band | Passivizable / intransitive | Accuracy | Mean margin |
| --- | --- | ---: | ---: |
| head | accompany / emerge | 1.000 | 2.930 |
| head | avoid / struggle | 1.000 | 2.658 |
| head | blame / laugh | 1.000 | 4.478 |
| head | bless / complain | 1.000 | 2.181 |
| head | condemn / testify | 1.000 | 3.285 |
| head | confront / remark | 1.000 | 4.782 |
| head | defend / reply | 0.875 | 1.206 |
| head | monitor / compete | 1.000 | 3.408 |
| head | punish / proceed | 1.000 | 5.976 |
| head | remind / respond | 1.000 | 2.433 |
| head | welcome / lie | 1.000 | 4.541 |
| tail | befriend / reappear | 0.250 | -1.436 |
| tail | berate / disembark | 0.875 | 1.757 |
| tail | chastise / collude | 1.000 | 7.114 |
| tail | defame / proliferate | 1.000 | 4.759 |
| tail | delude / mutate | 1.000 | 3.817 |
| tail | deride / dwindle | 1.000 | 4.710 |
| tail | flog / recede | 0.875 | 4.779 |
| tail | interrogate / emigrate | 1.000 | 7.168 |
| tail | laud / conspire | 1.000 | 4.307 |
| tail | malign / stagnate | 1.000 | 2.021 |
| tail | pacify / gush | 1.000 | 3.224 |
| tail | patronize / balk | 0.000 | -1.355 |
| tail | scrutinize / wane | 1.000 | 4.680 |
| tail | terrorize / diverge | 1.000 | 9.807 |
| tail | vilify / languish | 1.000 | 6.871 |
| xtail | cajole / hitchhike | 1.000 | 4.765 |
| xtail | discomfit / debark | 1.000 | 6.098 |
| xtail | edify / flounce | 1.000 | 4.404 |
| xtail | flummox / ossify | 1.000 | 6.179 |
| xtail | importune / putrefy | 0.875 | 1.618 |
| xtail | overawe / vegetate | 1.000 | 4.577 |
| xtail | reprove / remonstrate | 0.625 | 0.416 |
| xtail | satirize / decamp | 0.125 | -0.679 |
| xtail | stupefy / intermarry | 1.000 | 9.143 |
| xtail | tantalize / toddle | 1.000 | 5.265 |
| xtail | titillate / calve | 0.750 | 2.703 |
| xtail | upbraid / burgeon | 1.000 | 6.743 |

## Interpretation notes

The selected verbs are unique within each paradigm and reused across the two paradigms by design. Good and bad verbs are matched within 0.25 lemma Zipf and 0.35 participle Zipf. Shared human-patient frames are broadly plausible for the selected good verbs, though individual meanings may still make a sentence unusual, especially among xtail verbs. This is a small curated gate; a flat or noisy effect is inconclusive.
