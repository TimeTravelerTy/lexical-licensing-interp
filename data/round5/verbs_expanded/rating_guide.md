# D11 rating guide (shared by both raters; final version, before any frozen rating)

Each item is an English verb: its lemma and its past participle. Judge the verb in its ordinary present-day
senses, from your own knowledge only (no files or web). Give exactly one label:

- `T`: the verb's ordinary use includes a direct object, with a natural verbal passive in that sense
  ("The letter was signed (by the clerk)"). This holds even if the verb also has a common intransitive use
  (break, open, prepare). Flag `obligatory` if the verb normally needs its object (destroy, devour); flag
  `optional` if the object is often omitted or the verb also has a common intransitive use (eat, read, break).
- `I`: the verb is dominantly intransitive and has no common passivizable transitive use
  ("arrive", "sneeze", "dwindle"). A prepositional complement ("rely on") does not count as a direct object.
  Flag `rare_trans` if some transitive use exists but is rare, archaic, technical, idiomatic or not
  passivizable ("sneeze one's head off", "wait tables"); leave the flag empty otherwise.
- `reject`: neither fits: the verb is dominantly intransitive but has a common passivizable transitive use
  ("walk the dog", "march the troops"); it is mainly copular, raising or clause-taking, or a light verb; its
  participle is mainly an adjective or a noun; or you do not know the verb well enough.

Output: CSV with columns item,label,flag,reason; one row per item; flag is `obligatory` / `optional` for T,
`rare_trans` or empty for I, empty for reject; reason = at most 8 words.
