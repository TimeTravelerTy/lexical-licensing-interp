# B7 annotation guide (shared by both annotators; written before any labelling)

Each item is a text snippet (from web text) with one word marked [[...]]. Label the grammatical role of the
marked word in its context with exactly one of these labels (copy the label text exactly):

- `verbal passive participle`: a past participle in a passive construction: after a form of be/get
  ("was approved by ...", "has been replaced"), or in a reduced relative or participial clause whose head
  noun or understood subject is the participle's understood object ("the information provided by X",
  "the taxes paid", "Surrounded by friends, I ...", "..., followed by X").
- `adjectival / copular participle`: a participle used as an adjective describing a state, without an event or
  agent reading ("the broken window", "I am interested in", "very tired", "the door was closed all day").
- `active perfect or simple past`: the verb of an active clause in the perfect (have/has/had + participle) or the
  simple past, with its object (if any) in place after it.
- `object gap after the verb`: a verb (in any form other than a passive participle) whose object is missing
  after it because that object is the head of a relative clause, a wh-phrase, or the head of an infinitival
  relative / purpose clause ("the book I read __", "what you want __", "a tool to use __").
- `other`: anything else (nouns, adjectives, adverbs, prepositions, punctuation, active verbs with their object in
  place in the present tense or infinitive, ...).
- `uncertain`: the snippet does not allow a decision.

If both "active perfect or simple past" and "object gap after the verb" apply ("the advice you've heard __"),
choose `object gap after the verb`. Judge only the marked word; snippets may be cut mid-word at the edges;
"⏎" marks a line break.

Output: CSV with columns item,label,note, one row per item; note = a free description of the construction in at
most 12 words.
