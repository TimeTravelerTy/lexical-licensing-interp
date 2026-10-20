"""Inflected forms of the D11 / curated verbs (round 6, B3 and C5): base, 3sg present, -ing, simple past, participle.

forms(lemma, participle) uses the participle to decide consonant doubling and -ck (flopped -> flopping, picnicked ->
picnicking) and a short exception list for irregular verbs. check() flags forms with wordfreq Zipf 0 for hand review.
"""

from __future__ import annotations

PAST = {"arisen": "arose", "striven": "strove", "browbeaten": "browbeat", "outdone": "outdid"}
# irregular participles whose -ing / 3sg come from the lemma by the ordinary rules
IRREGULAR_PART = {"bet", "crept", "lent", "outshone", "recast", "sent", "slept", "dwelt", "knelt", "slunk", "beheld",
                  "waylaid", "arisen", "striven", "browbeaten", "outdone"}


def third_sg(lemma: str) -> str:
    if lemma.endswith("oo"):
        return lemma + "s"  # shampoo -> shampoos
    if lemma.endswith(("s", "sh", "ch", "x", "z", "o")):
        if lemma.endswith("z") and not lemma.endswith("zz"):
            return lemma + "zes"  # whiz -> whizzes
        return lemma + "es"
    if lemma.endswith("y") and lemma[-2:-1] not in set("aeiou"):
        return lemma[:-1] + "ies"
    return lemma + "s"


ING_EXCEPT = {"bet": "betting"}  # irregular participle = lemma with consonant doubling


def ing(lemma: str, participle: str) -> str:
    if lemma in ING_EXCEPT:
        return ING_EXCEPT[lemma]
    if lemma.endswith("ie"):
        return lemma[:-2] + "ying"
    if lemma.endswith(("ee", "ye", "oe")):
        return lemma + "ing"
    if lemma.endswith("y"):
        return lemma + "ing"
    if participle in IRREGULAR_PART or not participle.endswith("ed"):
        assert participle in IRREGULAR_PART, f"unsupported irregular participle: {participle}"
        return (lemma[:-1] if lemma.endswith("e") else lemma) + "ing"
    if lemma.endswith("e") and participle == lemma + "d":
        return lemma[:-1] + "ing"
    stem = participle[:-2]  # keeps doubling (flopp-) and -ck (picnick-)
    return stem + "ing"


def forms(lemma: str, participle: str) -> dict:
    return {"base": lemma, "3sg": third_sg(lemma), "ing": ing(lemma, participle),
            "past": PAST.get(participle, participle), "participle": participle}


def check(table):
    """Rows whose generated 3sg or -ing form has wordfreq Zipf 0 (for hand review)."""
    from wordfreq import zipf_frequency

    out = []
    for r in table:
        for k in ("3sg", "ing"):
            if zipf_frequency(r[k], "en") == 0:
                out.append((r["base"], k, r[k]))
    return out
