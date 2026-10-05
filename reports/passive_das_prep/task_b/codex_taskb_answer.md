**I would use target 3, omit `" by"` from both sets, and patch the last verb subtoken.** Train swaps in both directions, validate on held-out verbs, and treat passive transfer as evidence for context-sensitive reuse of the learned subspace. The pattern “by rises, object starts do not” would support an abstract interpretation, but would not uniquely identify “takes a direct object.”

Your intended dissociation is sensible. The weak assumption is that the two proposed readings exhaust the mechanisms that could produce it.

**1. Target: choose O versus I, with a narrower interpretation of what it measures.**

Define

\[
M(x)=\log P(O\mid x)-\log P(I\mid x),\qquad
q(x)=\frac{P(O\mid x)}{P(O\mid x)+P(I\mid x)}=\sigma(M(x)).
\]

Use binary cross-entropy on the patched run’s \(q\), with the independently assigned source class as the label. This makes the objective explicit: distinguish object-like starts from specified competing continuations.

Target 3 is my preferred operational target because:

- It addresses the central failure of target 1: `"."` is a poor representative of the alternatives available to intransitives.
- It pools multiple object starts, reducing dependence on one determiner.
- Unlike target 2, it does not require object starts to beat **the entire vocabulary**. A verb can license a direct object without putting more than half its next-token probability on your O set.
- Unlike full-distribution KL, it does not explicitly demand transfer of the source verb’s lexical preferences.

Your measurements support its feasibility: Head-train ordering is 99.8%, and the two classes mostly fall on opposite sides of zero.

However, **the “both” percentages are partly calibration statistics**. Zero means equal mass in two manually chosen sets. It is not a linguistic boundary. Pronouns-only achieves 99.6% ordering despite only 46% “both”; that is evidence against rejecting it solely because its transitive mean is negative.

I would amend O to include `" me"`, `" us"`, and reflexives, after checking their actual tokenizations. Keep determiner and pronoun results separately as diagnostic readouts. Pronouns are cleaner syntactically, but their probabilities depend strongly on discourse and selection: a garment-taking verb need not predict `" him"`.

Also record

\[
C(x)=P(O\mid x)+P(I\mid x).
\]

A convincing \(q\) with tiny \(C\) can describe a contest between two negligible alternatives while the model mostly predicts a clause, particle, or word extension.

**The strongest argument against my choice:** target 3 identifies an immediate continuation preference, not direct-object licensing itself. Determiners can introduce temporal adjuncts; transitive verbs can omit or delay objects; particles can intervene. Thus, it could recover a mixture of argument structure, lexical collocations, and construction preference. Its strong separation does not resolve that ambiguity.

A better construct check would use **short, independently annotated continuations**: complete nominal objects versus complete non-object continuations, including determiner-initial temporal adjuncts. Pool several continuations per category and test across different object choices. This adds selectional and sequence-length complications, so I would use it to validate target 3 rather than assume it automatically gives a purer target.

Finally, your present **intransitive base → transitive source only** design is underconstrained. Add the reverse direction and same-class swaps, and cross donors beyond their original verb-pair partners. Otherwise, DAS can succeed by finding a convenient way to boost noun starts. Training constrains intervention behavior; it does not establish the subspace’s meaning “by construction.” DAS’s causal-abstraction claim is relative to specified counterfactual predictions. [Geiger et al.](https://proceedings.mlr.press/v236/geiger24a.html)

**2. Omit `" by"` from both training sets; resolve the active ambiguity structurally.**

For target 3,

\[
M=\operatorname{LSE}(z_O)-\operatorname{LSE}(z_I).
\]

The softmax normalizer cancels. Consequently, an omitted `" by"` logit has no direct contribution to this loss. That is useful when `" by"` is your transfer readout.

Including `" by"` in I makes the active objective explicitly reward object starts relative to `" by"` for transitive labels. It also treats `" by"` as an acceptable competing continuation for intransitive labels. This mixes the transfer readout into training without making it a better diagnostic of intransitivity.

Target 2 has this problem implicitly: `" by"` belongs to “rest.”

**Excluding `" by"` does not remove passive information from the representation.** A transitive verb may naturally encode information useful for both active objects and passive licensing—that is partly what you hope to find. The avoidable confound is that your nominally active prefix can already be interpreted as a passive reduced relative.

I would compare these active frames:

| Frame | Role |
|---|---|
| `"She targeted"` / `"She salivated"` | Primary simple-past frame; pronoun subjects strongly reduce the reduced-relative reading |
| `"She has targeted"` / `"She has salivated"` | Strong disambiguation control; preserves the participial verb form |
| `"Maria targeted"` / `"Maria salivated"` | Additional subject-template generalization |
| `"The soldier targeted"` | Ambiguous-frame robustness condition |

Proper names reduce some ambiguity but are less decisive than an explicit active auxiliary. `"She has targeted"` commits the prefix to an active perfect construction; the corresponding passive would require *been*.

Train or replicate across simple-past and perfect-active frames, and hold out templates. This prevents an auxiliary-specific solution from carrying the whole argument.

Also, the 90 nouns are not uniformly “agent nouns.” `"The war"`, `"The news"`, and `"The gentleman"` create very different plausibility and interpretation pressures. Use independently judged subject–verb compatibility. A pronoun frame is cleaner structurally, but it still must suit the verb’s subject requirements.

Keep ambiguous `" that"` neutral in the first-token objective: placing it in I would misclassify demonstrative objects such as “took that book.”

**3. Last subtoken is the defensible primary position.**

At the last subtoken, the causal model has processed the entire written verb prefix, and that position produces the next-token prediction you are training.

At the first subtoken of a multi-token verb, the representation cannot yet depend on subsequent subtokens. You would be comparing a complete Head verb with an incomplete Tail word. That is especially problematic for shared prefixes.

“All subtokens” is a different intervention:

- Broadcasting one Head source state into several recipient positions changes intervention strength and introduces a questionable positional correspondence.
- Patching several positions creates additional routes through later computation.
- Learning separate position-specific subspaces increases intervention capacity.

Therefore, use **one rank-\(d\) intervention at the final verb subtoken**, at the same specified layer/component in both voices.

But your Head data cannot adjudicate first versus last versus all: every Head verb has one token. The position rule should therefore be specified before examining passive transfer.

The essential bridge condition is **common multi-token verbs**. Test held-out active transfer on those before interpreting rare-passive transfer. With your current distribution, frequency and token count are almost confounded; regression adjustment cannot recover a clean frequency comparison without overlap.

Report effects by token count and, where possible, final-token morphology. A shared suffix token could contribute to apparent transfer.

Finally, failure at the final position would reject this particular alignment and intervention site. It would not show that the model lacks abstract transitivity information; relevant information may remain elsewhere in the verb span.

**4. Exclusions should follow independent linguistic or technical criteria.**

The continuations suggest different problems, requiring different handling.

| Case | Recommended treatment | Basis |
|---|---|---|
| **jutted → “his…”** | Remove from the globally “strictly intransitive” primary inventory, or construct an independently disambiguated sense condition | “Jut one’s chin” is an ordinary transitive use |
| **scrammed** | Separate its departure sense from its technical transitive sense | “Scram a reactor” is documented |
| **crackled, resounded** | Audit their senses; do not assume globally strict intransitivity | Dictionaries list transitive uses |
| **fizzed, vegetated → “the…”** | Retain unless independent annotation establishes a label or plausibility problem | High model O mass alone establishes neither transitivity nor invalidity |
| **bet extending into another word** | Flag or exclude under a uniform technical criterion, or analyze with a boundary-aware diagnostic | Word-extension ambiguity is distinct from grammatical classification |

The lexical issue is broader than *jut*: documented transitive uses exist for [jut](https://www.oxfordlearnersdictionaries.com/us/definition/english/jut), [scram](https://www.collinsdictionary.com/dictionary/english/scram), [crackle](https://www.merriam-webster.com/dictionary/crackle), and [resound](https://www.collinsdictionary.com/us/dictionary/english/resound). Those entries do not prove that the model’s observed continuation selects the transitive sense. They do undermine an unqualified “strictly intransitive verb” label.

For the primary inventory, have annotators who are blind to model scores judge whether plausible direct-object continuations are available **from the actual prefix**. Post-verbal material cannot retrospectively disambiguate the sense available at the intervention point.

Choose the exclusion level accordingly:

- **Verb level:** the uncontextualized verb supports relevant transitive and intransitive senses.
- **Item level:** a particular subject/template is independently implausible or technically defective.
- **No exclusion:** the linguistic label is sound and the model simply predicts poorly.

Apply item exclusions symmetrically to the paired comparison.

Do not remove rare items merely because \(M>0\). That would select the evaluation population using the behavior you are trying to explain. You can additionally report a prespecified “model distinguishes these active classes” subset, but retain the full linguistically validated set and state the conditional scope of that subset.

**5. Passive transfer needs specificity controls and a defensible readout.**

Three assumptions need particular attention.

First, **direct-object licensing and passivizability are not equivalent across English verbs and constructions**. Restrict the primary claim to ordinary monotransitive verbs supporting the simple verbal passive. Separate verbs with retained objects, predicative complements, or required particles/complements.

Second, **immediate `" by"` is not a universal passive-licensing measure**. Your examples already establish this. “Summed up” predicts a particle before an agent phrase; “larded with” involves additional complement structure. Negative immediate-by margins can reflect these constructional preferences.

For the primary analysis, define a linguistically selected bare-participle passive subset. Analyze particle/complement verbs separately with appropriate continuation sequences. Do not discard them simply because their by margins are negative.

Third, **no rise in `" the"` does not uniquely imply abstraction**. Downstream passive syntax could suppress an object-start tendency imported from an active. Conversely, an abstract feature could increase determiner starts through a temporal adjunct or another licensed complement. Your passive baseline makes the dissociation worth testing, but does not identify its mechanism.

The minimum controls I would require are:

| Control | What it resolves |
|---|---|
| **Held-out active swaps in both directions** | Establishes that \(d\) controls the active contrast beyond training verbs |
| **Transitive → transitive and intransitive → intransitive** | Tests whether changes track class differences rather than arbitrary donor differences |
| **Same passive base, many transitive versus intransitive active donors** | Separates donor class from a particular verb identity or original pairing |
| **Active intransitive → bad passive, including the same verb** | Measures generic transfer between voices without changing lexical class |
| **Random subspaces at matched layer and dimension** | Establishes specificity; report both ordinary and displacement-norm-matched controls |
| **Shuffled-label DAS with the same search budget** | Tests how readily the procedure finds steering solutions without the intended labels |
| **Good-passive → bad-passive patching** | Provides a positive control for the recipient site and passive readout |
| **Common multi-token active and passive tests** | Separates tokenization transfer from rarity |
| **Several seeds and lexical splits** | Exposes instability with only 20 training pairs |

For a fixed passive base \(b\), a useful primary contrast is

\[
D_R(b)=
\mathbb E_{s\in T}R(b\leftarrow s)
-
\mathbb E_{s\in I}R(b\leftarrow s),
\]

with donor subjects/templates matched. Report this alongside each donor class’s change from the unpatched base. It controls for effects common to inserting an active-derived state into a passive.

Use multiple readouts: immediate `" by"`, complete agent-phrase continuations, total O mass, determiner mass, and pronoun/reflexive mass. Inspect where probability comes from: a by increase driven by broad punctuation suppression is weaker evidence than a selective change toward appropriate passive continuations.

For “no object-start gain,” use an **equivalence criterion with a prespecified meaningful bound**, rather than a nonsignificant test. Pooling `"the"` and `"a"` alone is insufficient.

Freeze the target, layer, dimension, position rule, and stopping criterion using active development data. Choosing them by the strongest passive-by effect would compromise the transfer test. Estimate uncertainty over lexical units, accounting for repeated donors and recipients; 90 subjects per pair do not give you 90 independent tests of lexical generalization.

A result would be uninterpretable as abstract transitivity transfer if it depended on passive-based selection, a few donors, equivalent effects from same-class/random controls, unresolved tokenization differences, or an invalid by readout. A null result would also be inconclusive if the positive control failed.

For a stronger claim about the model’s naturally used mechanism, add downstream tracing or complementary interventions. Successful subspace steering alone leaves that question open; the methodological interpretation is explicitly debated in [Makelov et al.](https://proceedings.iclr.cc/paper_files/paper/2024/hash/70b8505ac79e3e131756f793cd80eb8d-Abstract-Conference.html) and [Wu et al.’s reply](https://arxiv.org/abs/2401.12631).

The claim I would aim to support is: **an active-trained subspace causally controls object-continuation preference and transfers selectively to passive agent-continuation preference on held-out verbs.** Calling it an abstract direct-object-licensing variable requires the additional lexical, constructional, and mechanistic evidence above.