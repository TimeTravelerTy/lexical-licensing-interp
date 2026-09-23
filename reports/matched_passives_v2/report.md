# Pythia 1.4B diverse-context passive pilot

Positive margins favor the passivizable verb. The lexical unit is one verb pair. Two auxiliary templates are averaged within each pair. Patient and agent nouns were manually selected for each good verb. Intervals bootstrap verb pairs.

## Regime summaries

| Paradigm | Band | Verb pairs | Accuracy | Mean log-probability margin | 95% interval for margin |
| --- | --- | ---: | ---: | ---: | ---: |
| passive_1 | head | 26 | 1.000 | 9.330 | [8.285, 10.353] |
| passive_1 | tail | 50 | 0.990 | 9.969 | [8.815, 11.073] |
| passive_1 | xtail | 50 | 0.990 | 7.178 | [6.234, 8.120] |
| passive_2 | head | 26 | 1.000 | 6.890 | [5.720, 8.063] |
| passive_2 | tail | 50 | 0.990 | 7.888 | [7.030, 8.715] |
| passive_2 | xtail | 50 | 0.960 | 5.142 | [4.365, 5.908] |

## Head minus xtail sensitivity

| Paradigm | Metric | Difference | 95% interval | Leave-one-out range | Sign changes |
| --- | --- | ---: | ---: | ---: | ---: |
| passive_1 | mean_margin | 2.152 | [0.809, 3.526] | [1.955, 2.417] | 0/76 |
| passive_1 | accuracy | 0.010 | [0.000, 0.030] | [0.000, 0.010] | 1/76 |
| passive_2 | mean_margin | 1.748 | [0.323, 3.164] | [1.509, 1.986] | 0/76 |
| passive_2 | accuracy | 0.040 | [0.000, 0.090] | [0.020, 0.041] | 0/76 |

## Individual verb pairs

Each row is the mean across the tested templates. See `per_verb_pair.csv` for participle frequencies, token counts, and verb/suffix contributions.

### passive_1

| Band | Passivizable / bad verb | Accuracy | Mean margin |
| --- | --- | ---: | ---: |
| head | bet / appear | 1.000 | 9.195 |
| head | blame / struggle | 1.000 | 7.159 |
| head | celebrate / reply | 1.000 | 12.598 |
| head | confront / testify | 1.000 | 9.917 |
| head | crown / object | 1.000 | 9.262 |
| head | defend / participate | 1.000 | 7.552 |
| head | earn / occur | 1.000 | 12.126 |
| head | exhibit / proceed | 1.000 | 10.655 |
| head | hate / listen | 1.000 | 4.922 |
| head | host / exist | 1.000 | 14.251 |
| head | impose / emerge | 1.000 | 11.986 |
| head | include / happen | 1.000 | 9.529 |
| head | insure / vanish | 1.000 | 9.382 |
| head | introduce / arrive | 1.000 | 7.249 |
| head | lease / apologize | 1.000 | 10.624 |
| head | lend / opt | 1.000 | 10.258 |
| head | load / respond | 1.000 | 7.561 |
| head | measure / laugh | 1.000 | 13.015 |
| head | rate / result | 1.000 | 7.182 |
| head | rob / belong | 1.000 | 13.086 |
| head | score / remain | 1.000 | 7.872 |
| head | send / die | 1.000 | 10.203 |
| head | sum / compete | 1.000 | 2.720 |
| head | target / lie | 1.000 | 10.632 |
| head | thank / matter | 1.000 | 7.383 |
| head | upgrade / complain | 1.000 | 6.267 |
| tail | approximate / muse | 1.000 | 5.697 |
| tail | audit / interact | 1.000 | 14.751 |
| tail | awe / glow | 1.000 | 13.098 |
| tail | blurt / stagnate | 1.000 | 8.816 |
| tail | bribe / correspond | 1.000 | 12.228 |
| tail | christen / dwindle | 1.000 | 14.705 |
| tail | coerce / emigrate | 1.000 | 16.616 |
| tail | commemorate / brag | 1.000 | 8.990 |
| tail | deter / flirt | 1.000 | 10.255 |
| tail | dice / cooperate | 1.000 | 8.355 |
| tail | dope / bargain | 1.000 | 10.381 |
| tail | duck / dawn | 1.000 | 2.835 |
| tail | eschew / languish | 1.000 | 7.811 |
| tail | evade / inquire | 0.500 | -0.961 |
| tail | exert / slump | 1.000 | 14.273 |
| tail | fleece / vie | 1.000 | 10.105 |
| tail | forest / function | 1.000 | 1.525 |
| tail | gag / blossom | 1.000 | 11.361 |
| tail | galvanize / transpire | 1.000 | 14.201 |
| tail | garnish / toil | 1.000 | 8.528 |
| tail | heap / grin | 1.000 | 7.231 |
| tail | imitate / linger | 1.000 | 5.136 |
| tail | ingest / conspire | 1.000 | 12.389 |
| tail | nab / retaliate | 1.000 | 7.658 |
| tail | orphan / adhere | 1.000 | 16.404 |
| tail | outdo / recede | 1.000 | 12.477 |
| tail | overestimate / diverge | 1.000 | 9.941 |
| tail | perfume / reel | 1.000 | 16.520 |
| tail | preclude / fluctuate | 1.000 | 12.817 |
| tail | quarry / rave | 1.000 | 7.030 |
| tail | quell / sympathize | 1.000 | 9.594 |
| tail | rear / sin | 1.000 | 6.983 |
| tail | recast / reappear | 1.000 | 8.908 |
| tail | refund / clash | 1.000 | 12.551 |
| tail | reinstall / seep | 1.000 | 14.773 |
| tail | rouse / lurk | 1.000 | 13.716 |
| tail | safeguard / blush | 1.000 | 5.993 |
| tail | scrub / thrive | 1.000 | 17.167 |
| tail | slander / glare | 1.000 | 2.071 |
| tail | smother / subside | 1.000 | 13.103 |
| tail | spice / pee | 1.000 | 7.896 |
| tail | stockpile / shudder | 1.000 | 10.439 |
| tail | stow / abstain | 1.000 | 9.065 |
| tail | subjugate / collude | 1.000 | 7.963 |
| tail | subvert / persevere | 1.000 | 8.230 |
| tail | tame / chuckle | 1.000 | 8.000 |
| tail | trademark / bloom | 1.000 | 13.039 |
| tail | transcend / concur | 1.000 | 5.174 |
| tail | utter / creep | 1.000 | 13.577 |
| tail | whitewash / wane | 1.000 | 9.013 |
| xtail | blab / loll | 1.000 | 6.208 |
| xtail | blaspheme / seethe | 1.000 | 8.859 |
| xtail | clout / grouse | 1.000 | 3.778 |
| xtail | crate / lumber | 1.000 | 3.122 |
| xtail | daub / resound | 1.000 | 7.474 |
| xtail | debug / tingle | 1.000 | 8.712 |
| xtail | disabuse / sidle | 1.000 | 6.019 |
| xtail | disgorge / exult | 1.000 | 8.818 |
| xtail | disquiet / goggle | 1.000 | 6.673 |
| xtail | doff / salivate | 1.000 | 8.784 |
| xtail | edify / toddle | 1.000 | 5.568 |
| xtail | espy / scram | 0.500 | 0.655 |
| xtail | extirpate / vegetate | 1.000 | 2.653 |
| xtail | filch / smolder | 1.000 | 10.113 |
| xtail | garland / banter | 1.000 | 6.881 |
| xtail | gull / desist | 1.000 | 10.129 |
| xtail | guzzle / commiserate | 1.000 | 5.346 |
| xtail | heft / hibernate | 1.000 | 9.308 |
| xtail | immolate / vacillate | 1.000 | 11.083 |
| xtail | incriminate / sulk | 1.000 | 8.632 |
| xtail | lard / fizz | 1.000 | 2.450 |
| xtail | lather / slouch | 1.000 | 5.785 |
| xtail | lug / drool | 1.000 | 9.676 |
| xtail | mistrust / coexist | 1.000 | 8.145 |
| xtail | mulch / bustle | 1.000 | 2.241 |
| xtail | obviate / teem | 1.000 | 12.255 |
| xtail | ogle / wallow | 1.000 | 5.786 |
| xtail | outshine / gape | 1.000 | 3.944 |
| xtail | perm / fawn | 1.000 | 13.525 |
| xtail | pigeonhole / hitchhike | 1.000 | 5.405 |
| xtail | placate / crackle | 1.000 | 7.173 |
| xtail | prise / jut | 1.000 | 3.468 |
| xtail | rearm / totter | 1.000 | 12.110 |
| xtail | recompense / leer | 1.000 | 9.025 |
| xtail | redress / pant | 1.000 | 7.144 |
| xtail | regale / writhe | 1.000 | 9.332 |
| xtail | reproach / whiz | 1.000 | 8.661 |
| xtail | reprove / remonstrate | 1.000 | 0.981 |
| xtail | resupply / slog | 1.000 | 12.682 |
| xtail | shoo / quiver | 1.000 | 4.410 |
| xtail | shoplift / eventuate | 1.000 | 12.663 |
| xtail | stopper / quack | 1.000 | 6.156 |
| xtail | suckle / defecate | 1.000 | 2.303 |
| xtail | sunder / eavesdrop | 1.000 | 13.923 |
| xtail | swaddle / decamp | 1.000 | 8.918 |
| xtail | tithe / putter | 1.000 | 1.118 |
| xtail | unlatch / burgeon | 1.000 | 9.378 |
| xtail | wad / shimmer | 1.000 | 3.327 |
| xtail | wag / bask | 1.000 | 12.553 |
| xtail | wallop / gloat | 1.000 | 5.551 |

### passive_2

| Band | Passivizable / bad verb | Accuracy | Mean margin |
| --- | --- | ---: | ---: |
| head | bet / appear | 1.000 | 4.119 |
| head | blame / struggle | 1.000 | 4.980 |
| head | celebrate / reply | 1.000 | 10.415 |
| head | confront / testify | 1.000 | 7.219 |
| head | crown / object | 1.000 | 9.190 |
| head | defend / participate | 1.000 | 7.003 |
| head | earn / occur | 1.000 | 9.866 |
| head | exhibit / proceed | 1.000 | 6.102 |
| head | hate / listen | 1.000 | 3.210 |
| head | host / exist | 1.000 | 5.678 |
| head | impose / emerge | 1.000 | 9.759 |
| head | include / happen | 1.000 | 3.406 |
| head | insure / vanish | 1.000 | 2.633 |
| head | introduce / arrive | 1.000 | 2.428 |
| head | lease / apologize | 1.000 | 9.815 |
| head | lend / opt | 1.000 | 5.897 |
| head | load / respond | 1.000 | 9.805 |
| head | measure / laugh | 1.000 | 11.977 |
| head | rate / result | 1.000 | 5.211 |
| head | rob / belong | 1.000 | 12.861 |
| head | score / remain | 1.000 | 6.048 |
| head | send / die | 1.000 | 7.551 |
| head | sum / compete | 1.000 | 0.950 |
| head | target / lie | 1.000 | 8.373 |
| head | thank / matter | 1.000 | 5.449 |
| head | upgrade / complain | 1.000 | 9.198 |
| tail | approximate / muse | 1.000 | 5.769 |
| tail | audit / interact | 1.000 | 7.517 |
| tail | awe / glow | 1.000 | 11.918 |
| tail | blurt / stagnate | 1.000 | 3.927 |
| tail | bribe / correspond | 1.000 | 9.647 |
| tail | christen / dwindle | 1.000 | 8.932 |
| tail | coerce / emigrate | 1.000 | 8.804 |
| tail | commemorate / brag | 1.000 | 8.181 |
| tail | deter / flirt | 1.000 | 12.814 |
| tail | dice / cooperate | 1.000 | 10.969 |
| tail | dope / bargain | 1.000 | 8.676 |
| tail | duck / dawn | 1.000 | 2.836 |
| tail | eschew / languish | 1.000 | 4.836 |
| tail | evade / inquire | 0.500 | 1.098 |
| tail | exert / slump | 1.000 | 8.077 |
| tail | fleece / vie | 1.000 | 7.468 |
| tail | forest / function | 1.000 | 7.373 |
| tail | gag / blossom | 1.000 | 10.705 |
| tail | galvanize / transpire | 1.000 | 12.643 |
| tail | garnish / toil | 1.000 | 6.893 |
| tail | heap / grin | 1.000 | 7.600 |
| tail | imitate / linger | 1.000 | 3.817 |
| tail | ingest / conspire | 1.000 | 7.743 |
| tail | nab / retaliate | 1.000 | 8.531 |
| tail | orphan / adhere | 1.000 | 8.072 |
| tail | outdo / recede | 1.000 | 7.552 |
| tail | overestimate / diverge | 1.000 | 9.130 |
| tail | perfume / reel | 1.000 | 8.347 |
| tail | preclude / fluctuate | 1.000 | 8.229 |
| tail | quarry / rave | 1.000 | 6.903 |
| tail | quell / sympathize | 1.000 | 12.382 |
| tail | rear / sin | 1.000 | 2.637 |
| tail | recast / reappear | 1.000 | 6.464 |
| tail | refund / clash | 1.000 | 9.818 |
| tail | reinstall / seep | 1.000 | 8.767 |
| tail | rouse / lurk | 1.000 | 11.906 |
| tail | safeguard / blush | 1.000 | 5.584 |
| tail | scrub / thrive | 1.000 | 14.299 |
| tail | slander / glare | 1.000 | 2.922 |
| tail | smother / subside | 1.000 | 10.539 |
| tail | spice / pee | 1.000 | 8.059 |
| tail | stockpile / shudder | 1.000 | 5.069 |
| tail | stow / abstain | 1.000 | 13.052 |
| tail | subjugate / collude | 1.000 | 3.386 |
| tail | subvert / persevere | 1.000 | 3.686 |
| tail | tame / chuckle | 1.000 | 8.175 |
| tail | trademark / bloom | 1.000 | 8.835 |
| tail | transcend / concur | 1.000 | 4.159 |
| tail | utter / creep | 1.000 | 10.836 |
| tail | whitewash / wane | 1.000 | 8.803 |
| xtail | blab / loll | 1.000 | 4.935 |
| xtail | blaspheme / seethe | 1.000 | 5.770 |
| xtail | clout / grouse | 1.000 | 3.627 |
| xtail | crate / lumber | 1.000 | 2.247 |
| xtail | daub / resound | 1.000 | 3.590 |
| xtail | debug / tingle | 1.000 | 6.633 |
| xtail | disabuse / sidle | 1.000 | 7.743 |
| xtail | disgorge / exult | 1.000 | 5.753 |
| xtail | disquiet / goggle | 1.000 | 6.275 |
| xtail | doff / salivate | 1.000 | 8.845 |
| xtail | edify / toddle | 1.000 | 7.049 |
| xtail | espy / scram | 0.000 | -2.904 |
| xtail | extirpate / vegetate | 1.000 | 2.212 |
| xtail | filch / smolder | 1.000 | 4.735 |
| xtail | garland / banter | 1.000 | 8.807 |
| xtail | gull / desist | 1.000 | 3.870 |
| xtail | guzzle / commiserate | 1.000 | 5.393 |
| xtail | heft / hibernate | 1.000 | 4.833 |
| xtail | immolate / vacillate | 1.000 | 6.558 |
| xtail | incriminate / sulk | 1.000 | 4.522 |
| xtail | lard / fizz | 1.000 | 4.371 |
| xtail | lather / slouch | 1.000 | 0.384 |
| xtail | lug / drool | 1.000 | 4.798 |
| xtail | mistrust / coexist | 1.000 | 6.742 |
| xtail | mulch / bustle | 1.000 | 3.333 |
| xtail | obviate / teem | 1.000 | 9.425 |
| xtail | ogle / wallow | 1.000 | 4.264 |
| xtail | outshine / gape | 1.000 | 1.014 |
| xtail | perm / fawn | 1.000 | 9.148 |
| xtail | pigeonhole / hitchhike | 1.000 | 4.182 |
| xtail | placate / crackle | 1.000 | 4.767 |
| xtail | prise / jut | 1.000 | 1.628 |
| xtail | rearm / totter | 1.000 | 8.375 |
| xtail | recompense / leer | 1.000 | 7.163 |
| xtail | redress / pant | 1.000 | 8.520 |
| xtail | regale / writhe | 1.000 | 4.896 |
| xtail | reproach / whiz | 1.000 | 5.248 |
| xtail | reprove / remonstrate | 1.000 | 1.326 |
| xtail | resupply / slog | 1.000 | 8.578 |
| xtail | shoo / quiver | 1.000 | 4.731 |
| xtail | shoplift / eventuate | 1.000 | 10.319 |
| xtail | stopper / quack | 1.000 | 5.020 |
| xtail | suckle / defecate | 0.500 | -0.138 |
| xtail | sunder / eavesdrop | 1.000 | 9.205 |
| xtail | swaddle / decamp | 1.000 | 8.342 |
| xtail | tithe / putter | 1.000 | 2.171 |
| xtail | unlatch / burgeon | 1.000 | 6.349 |
| xtail | wad / shimmer | 0.500 | 1.522 |
| xtail | wag / bask | 1.000 | 7.453 |
| xtail | wallop / gloat | 1.000 | 3.464 |

## Interpretation notes

The selected verbs are unique within each paradigm and reused across the two paradigms by design. Good and bad verbs are matched within 0.25 lemma Zipf and 0.35 participle Zipf. Bands follow realised participle frequency. The reviewed head bad inventory contains fewer than 50 defensible verbs; the head estimate is less precise. Patient nouns vary by pair, so semantic context may still contribute to differences across regimes. Some head bad verbs select prepositions in active sentences, but their bare passives omit the preposition. A flat or noisy effect is inconclusive.
