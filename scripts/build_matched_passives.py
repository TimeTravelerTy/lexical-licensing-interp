#!/usr/bin/env python3
"""Build verb-frequency-controlled passive pairs from canonical FreqBLiMP.

The FreqBLiMP sentence files supply the licensed/unlicensed participle
inventories. Its vocabulary overlay supplies source lemmas for those forms.
Every selected verb pair is crossed with exactly the same common noun frames.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from wordfreq import zipf_frequency


BANDS = {"head": (3.5, 5.5), "tail": (2.4, 3.2), "xtail": (1.2, 2.2)}
PARADIGMS = ("passive_1", "passive_2")
# Broad human patients. The same frames are used for every verb. The curated
# good verbs below can all plausibly take a human patient in these frames.
FRAMES = (
    ("student_teacher", "The student", "the teacher"),
    ("teacher_doctor", "The teacher", "the doctor"),
    ("driver_worker", "The driver", "the worker"),
    ("artist_farmer", "The artist", "the farmer"),
    ("worker_student", "The worker", "the student"),
    ("farmer_artist", "The farmer", "the artist"),
    ("doctor_driver", "The doctor", "the driver"),
    ("teacher_worker", "The teacher", "the worker"),
)
CURATED_GOOD = {
    "head": {"accompany", "seize", "dismiss", "warn", "educate", "punish",
             "praise", "monitor", "remind", "defend", "blame", "welcome"},
    "tail": {"deride", "berate", "vilify", "delude", "chastise", "defame",
             "malign", "laud", "flog", "patronize", "pacify", "scrutinize",
             "terrorize", "interrogate", "befriend"},
    "xtail": {"discomfit", "overawe", "flummox", "importune", "upbraid",
              "stupefy", "tantalize", "reprove", "titillate", "edify",
              "satirize", "cajole"},
}
PARTICIPLE = re.compile(r"\b(?:was|were|is|are|wasn't|weren't|isn't|aren't)\s+([A-Za-z]+)(?=\s+by\b|[.?!])")


def source_lemma(root: str, expression: str) -> str:
    if "_overlay_" in root:
        return root.split("_overlay_", 1)[0].lower()
    if "_" in root:
        return root.rsplit("_", 1)[0].lower()
    return expression.lower()


def stable_hash(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()


def collect_forms(data_root: Path) -> tuple[dict, dict]:
    forms: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    audit = {"source_rows": {}, "unparsed_sentences": []}
    for band in BANDS:
        for paradigm in PARADIGMS:
            path = data_root / band / f"{paradigm}.jsonl"
            n = 0
            with path.open(encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    n += 1
                    for side in ("good", "bad"):
                        sentence = rec[f"sentence_{side}"]
                        match = PARTICIPLE.search(sentence)
                        if match:
                            forms[band, paradigm, side].add(match.group(1).lower())
                        else:
                            audit["unparsed_sentences"].append({"band": band, "paradigm": paradigm, "side": side, "sentence": sentence})
            audit["source_rows"][f"{band}/{paradigm}"] = n
    return forms, audit


def resolve_lemmas(freqblimp_root: Path, forms: dict) -> dict[str, set[str]]:
    wanted = set().union(*forms.values())
    resolved: dict[str, set[str]] = defaultdict(set)
    for source in ("vocabulary.csv", "vocabulary_overlay.csv"):
        with (freqblimp_root / source).open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                form = row["expression"].lower()
                if form not in wanted or row.get("en") != "1" or row.get("verb") != "1":
                    continue
                if row.get("category") not in {"(S\\NP)/NP", "S\\NP"}:
                    continue
                lemma = source_lemma(row.get("root", ""), form)
                if lemma.isalpha():
                    resolved[form].add(lemma)
    return resolved


def inventory(forms: dict, lemmas: dict, audit: dict) -> dict:
    out: dict[tuple[str, str, str], list[dict]] = {}
    audit["excluded"] = defaultdict(int)
    for key, selected_forms in sorted(forms.items()):
        band = key[0]
        lo, hi = BANDS[band]
        rows = []
        for form in sorted(selected_forms):
            options = lemmas.get(form, set())
            if len(options) != 1:
                audit["excluded"]["missing_or_ambiguous_lemma"] += 1
                continue
            lemma = next(iter(options))
            zipf = float(zipf_frequency(lemma, "en"))
            if not lo <= zipf <= hi:
                audit["excluded"]["lemma_outside_source_band"] += 1
                continue
            rows.append({"form": form, "lemma": lemma, "lemma_zipf": zipf,
                         "form_zipf": float(zipf_frequency(form, "en"))})
        # A lemma may have two participle spellings; retain one deterministically.
        by_lemma = {}
        for row in sorted(rows, key=lambda r: (-r["form_zipf"], r["form"])):
            by_lemma.setdefault(row["lemma"], row)
        out[key] = sorted(by_lemma.values(), key=lambda r: (r["lemma_zipf"], r["lemma"]))
    audit["excluded"] = dict(audit["excluded"])
    audit["inventory_counts"] = {"/".join(k): len(v) for k, v in out.items()}
    return out


def pair_inventory(good: list[dict], bad: list[dict], max_pairs: int, seed: int) -> list[tuple[dict, dict]]:
    # Match each eligible passivizable verb to its nearest unused intransitive.
    # The lexical class is frequency matched within each source band.
    good = sorted(good, key=lambda r: (r["lemma_zipf"], stable_hash(r["lemma"], seed)))
    if len(good) > max_pairs:
        indices = [round((i + 0.5) * len(good) / max_pairs - 0.5) for i in range(max_pairs)]
        good = [good[i] for i in indices]
    unused = {r["lemma"]: r for r in bad}
    pairs = []
    for g in good:
        candidates = [b for b in unused.values() if g["lemma"] != b["lemma"]]
        if not candidates:
            break
        b = min(candidates, key=lambda b: (abs(g["lemma_zipf"] - b["lemma_zipf"]),
                                          stable_hash(g["lemma"] + b["lemma"], seed)))
        pairs.append((g, b))
        del unused[b["lemma"]]
    return pairs


def build(args: argparse.Namespace) -> None:
    root = Path(args.freqblimp_root)
    forms, audit = collect_forms(root / "data" / "freqblimp")
    lemmas = resolve_lemmas(root, forms)
    inventories = inventory(forms, lemmas, audit)
    rows = []
    audit["pair_counts"] = {}
    for band in BANDS:
        shared = {}
        for side in ("good", "bad"):
            first = {r["lemma"]: r for r in inventories[band, "passive_1", side]}
            second = {r["lemma"]: r for r in inventories[band, "passive_2", side]}
            shared[side] = [first[lemma] for lemma in sorted(first.keys() & second.keys())
                            if first[lemma]["form"] == second[lemma]["form"]]
        good = shared["good"]
        if args.inventory == "curated":
            good = [r for r in good if r["lemma"] in CURATED_GOOD[band]]
        pairs = pair_inventory(good, shared["bad"], args.max_pairs_per_band, args.seed)
        for paradigm in PARADIGMS:
            audit["pair_counts"][f"{band}/{paradigm}"] = len(pairs)
            for pair_idx, (g, b) in enumerate(pairs):
                for frame_id, patient, agent in FRAMES[:args.contexts]:
                    suffix = f" by {agent}." if paradigm == "passive_1" else "."
                    prefix = f"{patient} was "
                    rows.append({
                        "pair_id": f"{band}/{paradigm}/{pair_idx}/{frame_id}",
                        "band": band, "paradigm": paradigm, "frame_id": frame_id,
                        "patient": patient, "agent": agent if paradigm == "passive_1" else "",
                        "prefix": prefix, "suffix": suffix,
                        "good_lemma": g["lemma"], "bad_lemma": b["lemma"],
                        "good_verb": g["form"], "bad_verb": b["form"],
                        "good_lemma_zipf": g["lemma_zipf"], "bad_lemma_zipf": b["lemma_zipf"],
                        "good_form_zipf": g["form_zipf"], "bad_form_zipf": b["form_zipf"],
                        "lemma_zipf_gap": g["lemma_zipf"] - b["lemma_zipf"],
                        "sentence_good": prefix + g["form"] + suffix,
                        "sentence_bad": prefix + b["form"] + suffix,
                    })
    if any(n == 0 for n in audit["pair_counts"].values()):
        raise SystemExit(f"Empty condition: {audit['pair_counts']}")
    audit.update({"seed": args.seed, "max_pairs_per_band": args.max_pairs_per_band,
                  "contexts": args.contexts, "total_pairs": len(rows),
                  "source_root": str(root), "inventory": args.inventory,
                  "design": "same human-patient frames crossed with lemma-Zipf-matched verb classes"})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    Path(args.audit_out).write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "pairs": len(rows), "pair_counts": audit["pair_counts"],
                      "inventory_counts": audit["inventory_counts"], "excluded": audit["excluded"]}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--freqblimp-root", default="../freq-blimp")
    ap.add_argument("--out", default="data/matched_passives/pairs.jsonl")
    ap.add_argument("--audit-out", default="data/matched_passives/audit.json")
    ap.add_argument("--contexts", type=int, default=8, choices=range(1, len(FRAMES) + 1))
    ap.add_argument("--max-pairs-per-band", type=int, default=20)
    ap.add_argument("--inventory", choices=("curated", "all"), default="curated")
    ap.add_argument("--seed", type=int, default=17)
    build(ap.parse_args())
