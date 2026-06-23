#!/usr/bin/env python3
"""Build tokenizer-verified v2 lexical-licensing templates from FreqBLiMP.

This builder uses the canonical `freq-blimp` outputs for regime/subtask verb
support and `freq-blimp/vocabulary_overlay.csv` for role-compatible noun
sampling. It intentionally avoids the archived `blimp-rare` swapping pipeline.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


FREQ_BLIMP_ROOT = Path(os.environ.get("FREQ_BLIMP_ROOT", "/Users/tyronewhite/masters_research_code/freq-blimp"))
OVERLAY_CSV = FREQ_BLIMP_ROOT / "vocabulary_overlay.csv"
DATA_ROOT = FREQ_BLIMP_ROOT / "data" / "freqblimp"
OVERLAY_GUARDS = FREQ_BLIMP_ROOT / "generation_projects" / "blimp" / "overlay_guards.py"
SUPPLEMENTAL_BAD_VERBS = Path("data/aligned_templates_v2/supplemental_bad_verbs.json")

REGIMES = ("head", "low")
SUBTASKS = {"causative", "inchoative", "transitive", "intransitive", "drop_argument"}
TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
ZIPF_BANDS = {
    "head": (3.5, 5.5),
    "tail": (2.4, 3.2),
    "xtail": (1.2, 2.2),
    "low": (1.2, 3.2),
}
AUX_FORMS = {
    "am", "are", "is", "was", "were", "be", "been", "being",
    "do", "does", "did", "done", "doing",
    "have", "has", "had", "having",
    "can", "could", "may", "might", "must", "shall", "should", "will", "would",
}

SUBTASK_LABELS = {
    "causative": ("object_licensed", "object_unlicensed", "object_frame"),
    "transitive": ("object_licensed", "object_unlicensed", "object_frame"),
    "inchoative": ("no_object_licensed", "no_object_unlicensed", "no_object_frame"),
    "intransitive": ("no_object_licensed", "no_object_unlicensed", "no_object_frame"),
    "drop_argument": ("drop_object_licensed", "drop_object_unlicensed", "drop_object_frame"),
}

OBJECT_SCHEMAS = (
    ("overlay_subject_lab", "In the lab, the {subject} will {verb}"),
    ("overlay_subject_visit", "After the visit, the {subject} will {verb}"),
    ("overlay_subject_check", "During the check, the {subject} will {verb}"),
    ("overlay_subject_tomorrow", "Tomorrow, the {subject} will {verb}"),
    ("overlay_subject_trial", "During the trial, the {subject} will {verb}"),
)

DROP_SCHEMAS = (
    ("overlay_subject_studio", "In the studio, the {subject} will {verb}"),
    ("overlay_subject_shift", "After the shift, the {subject} will {verb}"),
    ("overlay_subject_storage", "During storage, the {subject} will {verb}"),
    ("overlay_subject_later", "Later today, the {subject} will {verb}"),
    ("overlay_subject_rehearsal", "During rehearsal, the {subject} will {verb}"),
)

NOUN_FIELDS = (
    "agent", "animal", "animate", "artifact", "buildable", "cleanable",
    "climbable", "conceptual", "document", "drinkable", "food", "institution",
    "liquid", "locale", "mass", "person", "physical", "frequent", "sg", "pl",
    "topic", "vehicle",
)
_NOUN_MATCH_CACHE: dict[str, list[dict[str, str]]] = {}
_ZIPF_CACHE: dict[str, float] = {}


@dataclass(frozen=True)
class VerbProfile:
    lemma: str
    form: str
    root: str
    arg_1: str
    arg_2: str
    category: str
    category_2: str


@dataclass(frozen=True)
class VerbCandidate:
    regime: str
    subtask: str
    side: str
    lemma: str
    source_form: str
    source_idx: int
    source_text: str
    profile: VerbProfile | None
    source_zipf: float | None = None
    source_zipf_regime: str = ""
    inventory_source: str = "freqblimp_curated"
    curation_reason: str = ""


def stable_unit(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def ensure_freqblimp_on_path() -> None:
    root = str(FREQ_BLIMP_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def zipf_for_lemma(lemma: str) -> float:
    lemma = clean_expression(lemma)
    if not lemma:
        return 0.0
    if lemma not in _ZIPF_CACHE:
        ensure_freqblimp_on_path()
        from utils.frequency import zipf_for_expression

        _ZIPF_CACHE[lemma] = float(zipf_for_expression(lemma))
    return _ZIPF_CACHE[lemma]


def historical_zipf_regime(zipf: float | None) -> str:
    if zipf is None:
        return ""
    eps = 1e-6
    if ZIPF_BANDS["xtail"][0] - eps <= zipf <= ZIPF_BANDS["xtail"][1] + eps:
        return "xtail"
    if ZIPF_BANDS["tail"][0] - eps <= zipf <= ZIPF_BANDS["tail"][1] + eps:
        return "tail"
    if ZIPF_BANDS["low"][0] - eps <= zipf <= ZIPF_BANDS["low"][1] + eps:
        return "low_gap"
    if ZIPF_BANDS["head"][0] - eps <= zipf <= ZIPF_BANDS["head"][1] + eps:
        return "head"
    return "out_of_band"


def zipf_in_regime(zipf: float | None, regime: str) -> bool:
    if zipf is None or regime not in ZIPF_BANDS:
        return False
    lo, hi = ZIPF_BANDS[regime]
    eps = 1e-6
    return lo - eps <= zipf <= hi + eps


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def clean_word(value: object) -> str:
    text = str(value or "").strip().lower()
    if not text or " " in text or "-" in text or "'" in text:
        return ""
    if not text.isalpha():
        return ""
    return text


def clean_expression(value: object) -> str:
    text = str(value or "").strip().lower()
    if not text or "-" in text or "'" in text:
        return ""
    if not all(part.isalpha() for part in text.split()):
        return ""
    return text


def source_lemma_from_overlay_root(root: str) -> str:
    if "_overlay_" not in root:
        return ""
    tail = root.split("_overlay_", 1)[1]
    for prefix in ("SNP_NP_", "SNP_"):
        if tail.startswith(prefix):
            tail = tail[len(prefix):]
            break
    for suffix in ("_s_np_np", "_s_np"):
        if tail.endswith(suffix):
            return clean_expression(tail[: -len(suffix)].replace("_", " "))
    return ""


def source_lemma_from_freqblimp_root(root: str, expression: str) -> str:
    text = str(root or "").strip()
    if "_overlay_" in text:
        text = text.split("_overlay_", 1)[0]
    elif "_" in text:
        stem, suffix = text.rsplit("_", 1)
        if "\\" in suffix or "/" in suffix or suffix in {"S", "NP", "N"}:
            text = stem
    return clean_expression((text or expression).replace("_", " "))


def condition_matches(row: dict[str, str], condition: str) -> bool:
    if not condition:
        return True
    for disjunct in condition.split(";"):
        ok = True
        for part in disjunct.split("^"):
            if not part:
                continue
            if "=" not in part:
                ok = False
                break
            key, value = part.split("=", 1)
            if row.get(key, "") != value:
                ok = False
                break
        if ok:
            return True
    return False


def condition_with_conjunct(condition: str, conjunct: str) -> str:
    condition = condition or ""
    if not condition:
        return conjunct
    return ";".join(
        f"{disjunct}^{conjunct}" if disjunct else conjunct
        for disjunct in condition.split(";")
    )


def load_overlay(overlay_csv: Path) -> tuple[dict[str, str], dict[str, VerbProfile], list[dict[str, str]]]:
    root_to_bare: dict[str, str] = {}
    root_profiles: dict[str, VerbProfile] = {}
    form_to_root: dict[str, str] = {}
    expression_profiles: dict[str, VerbProfile] = {}
    nouns: list[dict[str, str]] = []

    with overlay_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            expression = clean_expression(row.get("expression"))
            if not expression:
                continue
            if row.get("verb") == "1" and row.get("root"):
                root = str(row["root"])
                form_to_root.setdefault(expression, root)
                if row.get("bare") == "1" and "_overlay_" not in root and expression not in expression_profiles:
                    expression_profiles[expression] = VerbProfile(
                        lemma=expression,
                        form=expression,
                        root=root,
                        arg_1=str(row.get("arg_1", "")),
                        arg_2=str(row.get("arg_2", "")),
                        category=str(row.get("category", "")),
                        category_2=str(row.get("category_2", "")),
                    )
                source_lemma = source_lemma_from_overlay_root(root)
                if row.get("bare") == "1" and source_lemma and source_lemma not in expression_profiles:
                    expression_profiles[source_lemma] = VerbProfile(
                        lemma=source_lemma,
                        form=expression,
                        root=root,
                        arg_1=str(row.get("arg_1", "")),
                        arg_2=str(row.get("arg_2", "")),
                        category=str(row.get("category", "")),
                        category_2=str(row.get("category_2", "")),
                    )
                if row.get("bare") == "1" and "_overlay_" not in root:
                    root_to_bare.setdefault(root, expression)
                    root_profiles.setdefault(
                        root,
                        VerbProfile(
                            lemma=expression,
                            form=expression,
                            root=root,
                            arg_1=str(row.get("arg_1", "")),
                            arg_2=str(row.get("arg_2", "")),
                            category=str(row.get("category", "")),
                            category_2=str(row.get("category_2", "")),
                        ),
                )
            elif row.get("noun") == "1" and row.get("category") == "N":
                expression = clean_word(row.get("expression"))
                if not expression:
                    continue
                if row.get("properNoun") == "1" or row.get("mass") == "1":
                    continue
                noun = {field: str(row.get(field, "")) for field in NOUN_FIELDS}
                noun["expression"] = expression
                nouns.append(noun)

    form_to_lemma = {
        form: root_to_bare[root]
        for form, root in form_to_root.items()
        if root in root_to_bare
    }
    profiles = {
        profile.lemma: profile
        for root, profile in root_profiles.items()
        if root_to_bare.get(root) == profile.lemma
    }
    profiles.update(expression_profiles)
    return form_to_lemma, profiles, nouns


def load_curated_tuple(name: str, source: Path = OVERLAY_GUARDS) -> tuple[str, ...]:
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            values = ast.literal_eval(node.value)
            return tuple(clean_expression(value) for value in values if clean_expression(value))
    raise ValueError(f"missing curated tuple {name} in {source}")


def curated_verb_inventory() -> dict[tuple[str, str], tuple[str, ...]]:
    alternating = load_curated_tuple("_CAUSATIVE_ALTERNATING_VERBS")
    bad_intransitives = tuple(dict.fromkeys(
        load_curated_tuple("_NONPASSIVIZABLE_PARTICIPLE_VERBS")
        + load_curated_tuple("_CAUSATIVE_BAD_EXTRA_INTRANSITIVES")
        + load_curated_tuple("_CAUSATIVE_BAD_SAFE_INTRANSITIVES")
    ))
    bad_transitives = load_curated_tuple("_DROP_ARGUMENT_BAD_VERBS")
    return {
        ("causative", "good"): alternating,
        ("causative", "bad"): bad_intransitives,
        ("inchoative", "good"): alternating,
        ("inchoative", "bad"): bad_transitives,
    }


def profile_from_freqblimp_row(row: Any, lemma: str, root_label: str | None = None) -> VerbProfile:
    names = set(getattr(getattr(row, "dtype", None), "names", ()) or ())

    def field(name: str) -> str:
        return str(row[name]) if name in names else ""

    return VerbProfile(
        lemma=lemma,
        form=field("expression") or lemma,
        root=root_label or field("root"),
        arg_1=field("arg_1"),
        arg_2=field("arg_2"),
        category=field("category"),
        category_2=field("category_2"),
    )


def filtered_curated_rows_for_regime(regime: str) -> dict[tuple[str, str], list[tuple[str, VerbProfile, float]]]:
    if regime not in ZIPF_BANDS:
        raise ValueError(f"unknown regime: {regime}")
    ensure_freqblimp_on_path()
    from utils.randomize import SamplingPolicy, clear_sampling_policy, configure_sampling_policy
    from generation_projects.blimp.overlay_guards import (
        causative_alternating_verb_rows,
        causative_bad_intransitive_rows,
        filter_rows_for_active_zipf,
        inchoative_bad_transitive_rows,
    )
    from utils.vocab_table import get_table_zipf_expression

    lo, hi = ZIPF_BANDS[regime]
    configure_sampling_policy(
        SamplingPolicy(
            seed=17,
            controlled_pos=("noun", "verb", "adjective"),
            zipf_min={"noun": lo, "verb": lo, "adjective": lo},
            zipf_max={"noun": hi, "verb": hi, "adjective": hi},
            overlay_enabled=True,
        )
    )
    specs = {
        ("causative", "good"): causative_alternating_verb_rows(),
        ("causative", "bad"): causative_bad_intransitive_rows(),
        ("inchoative", "good"): causative_alternating_verb_rows(),
        ("inchoative", "bad"): inchoative_bad_transitive_rows(),
    }
    out: dict[tuple[str, str], list[tuple[str, VerbProfile, float]]] = {}
    try:
        for key, table in specs.items():
            filtered = filter_rows_for_active_zipf(table, "verb", fallback_on_empty=False)
            filtered_zipf = get_table_zipf_expression(filtered) if len(filtered) else []
            by_lemma: dict[str, tuple[VerbProfile, float]] = {}
            for row, source_zipf in zip(filtered, filtered_zipf):
                expression = clean_expression(str(row["expression"]))
                lemma = source_lemma_from_freqblimp_root(str(row["root"]), expression)
                if not lemma or " " in lemma:
                    continue
                if lemma not in by_lemma or str(row["bare"]) == "1":
                    by_lemma[lemma] = (
                        profile_from_freqblimp_row(
                            row,
                            lemma,
                            root_label=f"freqblimp_zipf_{regime}:{row['root']}:{row['expression']}",
                        ),
                        float(source_zipf),
                    )
            out[key] = [(lemma, profile, source_zipf) for lemma, (profile, source_zipf) in sorted(by_lemma.items())]
    finally:
        clear_sampling_policy()
    return out


def load_supplemental_bad_verbs(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("verbs", [])
    if not isinstance(entries, list):
        raise ValueError(f"supplemental verb file has non-list 'verbs': {path}")
    cleaned: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        lemma = clean_expression(entry.get("lemma"))
        subtask = str(entry.get("subtask", "")).strip()
        side = str(entry.get("side", "")).strip()
        regimes = entry.get("regimes", [])
        if isinstance(regimes, str):
            regimes = [regimes]
        regimes = [str(regime).strip() for regime in regimes if str(regime).strip()]
        if not lemma or subtask not in SUBTASKS or side not in {"good", "bad"}:
            continue
        cleaned.append(
            {
                "lemma": lemma,
                "subtask": subtask,
                "side": side,
                "regimes": regimes,
                "curation_reason": str(entry.get("curation_reason", "")),
            }
        )
    return cleaned


def fallback_profile(lemma: str, subtask: str, side: str) -> VerbProfile:
    if subtask == "causative" and side == "good":
        return VerbProfile(lemma, lemma, "curated_causative_alternating", "animate=1", "physical=1", "(S\\NP)/NP", "TV")
    if subtask == "causative" and side == "bad":
        return VerbProfile(lemma, lemma, "curated_causative_bad_intransitive", "animate=1;physical=1", "", "S\\NP", "IV")
    if subtask == "inchoative" and side == "good":
        return VerbProfile(lemma, lemma, "curated_causative_alternating", "physical=1", "", "S\\NP", "IV")
    if subtask == "inchoative" and side == "bad":
        return VerbProfile(lemma, lemma, "curated_inchoative_bad_transitive", "animate=1", "physical=1", "(S\\NP)/NP", "TV")
    return VerbProfile(lemma, lemma, "curated_unknown", "", "", "", "")


def collect_curated_candidates(
    subtasks: set[str],
    profiles: dict[str, VerbProfile],
    supplemental_bad_verbs: list[dict[str, Any]],
) -> list[VerbCandidate]:
    candidates: list[VerbCandidate] = []
    for regime in REGIMES:
        inventory = filtered_curated_rows_for_regime(regime)
        for subtask in sorted(subtasks & {"causative", "inchoative"}):
            for side in ("good", "bad"):
                by_lemma: dict[str, VerbCandidate] = {}
                for lemma, zipf_profile, source_zipf in inventory[(subtask, side)]:
                    profile = zipf_profile or profiles.get(lemma) or fallback_profile(lemma, subtask, side)
                    by_lemma[lemma] = VerbCandidate(
                        regime=regime,
                        subtask=subtask,
                        side=side,
                        lemma=lemma,
                        source_form=profile.form or lemma,
                        source_idx=0,
                        source_text=f"curated_zipf:{regime}:{subtask}:{side}",
                        profile=profile,
                        source_zipf=source_zipf,
                        source_zipf_regime=historical_zipf_regime(source_zipf),
                        inventory_source="freqblimp_curated",
                    )
                for entry in supplemental_bad_verbs:
                    if entry["subtask"] != subtask or entry["side"] != side:
                        continue
                    if regime not in entry["regimes"]:
                        continue
                    lemma = str(entry["lemma"])
                    if lemma in by_lemma:
                        continue
                    source_zipf = zipf_for_lemma(lemma)
                    if not zipf_in_regime(source_zipf, regime):
                        continue
                    profile = profiles.get(lemma) or fallback_profile(lemma, subtask, side)
                    by_lemma[lemma] = VerbCandidate(
                        regime=regime,
                        subtask=subtask,
                        side=side,
                        lemma=lemma,
                        source_form=profile.form or lemma,
                        source_idx=0,
                        source_text=f"supplemental_zipf:{regime}:{subtask}:{side}",
                        profile=profile,
                        source_zipf=source_zipf,
                        source_zipf_regime=historical_zipf_regime(source_zipf),
                        inventory_source="supplemental_v2_bad_verbs",
                        curation_reason=str(entry.get("curation_reason", "")),
                    )
                for source_idx, candidate in enumerate(sorted(by_lemma.values(), key=lambda item: item.lemma)):
                    candidates.append(
                        VerbCandidate(
                            regime=candidate.regime,
                            subtask=candidate.subtask,
                            side=candidate.side,
                            lemma=candidate.lemma,
                            source_form=candidate.source_form,
                            source_idx=source_idx,
                            source_text=candidate.source_text,
                            profile=candidate.profile,
                            source_zipf=candidate.source_zipf,
                            source_zipf_regime=candidate.source_zipf_regime,
                            inventory_source=candidate.inventory_source,
                            curation_reason=candidate.curation_reason,
                        )
                    )
    return candidates


def extract_main_verb(sentence: str, form_to_lemma: dict[str, str]) -> tuple[str, str] | None:
    tokens = [m.group(0).lower() for m in TOKEN_RE.finditer(sentence)]
    candidates = [(token, form_to_lemma[token]) for token in tokens if token in form_to_lemma and token not in AUX_FORMS]
    if not candidates:
        return None
    return candidates[-1]


def collect_candidates(
    subtasks: set[str],
    form_to_lemma: dict[str, str],
    profiles: dict[str, VerbProfile],
    supplemental_bad_verbs_path: Path,
) -> list[VerbCandidate]:
    seen: set[tuple[str, str, str, str]] = set()
    supplemental_bad_verbs = load_supplemental_bad_verbs(supplemental_bad_verbs_path)
    candidates: list[VerbCandidate] = collect_curated_candidates(subtasks, profiles, supplemental_bad_verbs)
    generated_subtasks = subtasks - {"causative", "inchoative"}
    for regime in REGIMES:
        for subtask in sorted(generated_subtasks):
            path = DATA_ROOT / regime / f"{subtask}.jsonl"
            if not path.exists():
                continue
            for rec in iter_jsonl(path):
                source_idx = int(rec.get("pairID", rec.get("idx", len(candidates))))
                for side, text_key in (("good", "sentence_good"), ("bad", "sentence_bad")):
                    sentence = str(rec.get(text_key, ""))
                    extracted = extract_main_verb(sentence, form_to_lemma)
                    if extracted is None:
                        continue
                    source_form, lemma = extracted
                    key = (regime, subtask, side, lemma)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(
                        VerbCandidate(
                            regime=regime,
                            subtask=subtask,
                            side=side,
                            lemma=lemma,
                            source_form=source_form,
                            source_idx=source_idx,
                            source_text=sentence,
                            profile=profiles.get(lemma),
                        )
                    )
    return candidates


def target_for(subtask: str, side: str) -> tuple[str, str]:
    if subtask in {"causative", "transitive"}:
        good_target, bad_target = " the", "."
    elif subtask in {"inchoative", "intransitive", "drop_argument"}:
        good_target, bad_target = ".", " the"
    else:
        raise ValueError(f"unknown subtask: {subtask}")
    if side == "good":
        return good_target, bad_target
    return bad_target, good_target


def contexts_for_subtask(subtask: str) -> tuple[tuple[str, str], ...]:
    return DROP_SCHEMAS if subtask == "drop_argument" else OBJECT_SCHEMAS


def sample_matching_nouns(
    nouns: list[dict[str, str]],
    condition: str,
    count: int,
    seed_key: str,
    seed: int,
) -> list[dict[str, str]]:
    matches: list[dict[str, str]] = []
    for preferred_condition in (
        condition_with_conjunct(condition_with_conjunct(condition, "frequent=1"), "sg=1"),
        condition_with_conjunct(condition, "frequent=1"),
    ):
        cache_key = preferred_condition or "<any>"
        matches = _NOUN_MATCH_CACHE.get(cache_key, [])
        if not matches and cache_key not in _NOUN_MATCH_CACHE:
            matches = sorted(
                (row for row in nouns if condition_matches(row, preferred_condition)),
                key=lambda row: row["expression"],
            )
            _NOUN_MATCH_CACHE[cache_key] = matches
        if matches:
            break
    if not matches:
        fallback_key = condition or "<any>"
        matches = _NOUN_MATCH_CACHE.get(fallback_key)
        if matches is None:
            matches = sorted((row for row in nouns if condition_matches(row, condition)), key=lambda row: row["expression"])
            _NOUN_MATCH_CACHE[fallback_key] = matches
    if not matches:
        fallback_key = "<frequent-sg>"
        matches = _NOUN_MATCH_CACHE.get(fallback_key)
        if matches is None:
            matches = sorted(
                (row for row in nouns if row.get("sg") == "1" and row.get("frequent") == "1"),
                key=lambda row: row["expression"],
            )
            _NOUN_MATCH_CACHE[fallback_key] = matches
    if not matches:
        return []
    n = max(1, count)
    start = int(stable_unit(seed_key, seed) * len(matches)) % len(matches)
    return [matches[(start + i) % len(matches)] for i in range(min(n, len(matches)))]


def deterministic_sample(candidates: list[VerbCandidate], limit: int | None, seed: int) -> list[VerbCandidate]:
    if limit is None or len(candidates) <= limit:
        return sorted(candidates, key=lambda c: (c.lemma, c.source_idx))
    ordered = sorted(
        candidates,
        key=lambda c: (
            stable_unit(f"{c.regime}|{c.subtask}|{c.side}|{c.lemma}|{c.source_idx}", seed),
            c.lemma,
            c.source_idx,
        ),
    )
    return sorted(ordered[:limit], key=lambda c: (c.lemma, c.source_idx))


def make_row(
    candidate: VerbCandidate,
    schema_id: str,
    template: str,
    subject: dict[str, str],
    obj: dict[str, str] | None,
    row_index: int,
) -> dict[str, Any]:
    good_label, bad_label, frame_label = SUBTASK_LABELS[candidate.subtask]
    expected_target, contrast_target = target_for(candidate.subtask, candidate.side)
    subject_text = subject["expression"]
    object_text = obj["expression"] if obj else ""
    prompt = template.format(subject=subject_text, verb=candidate.lemma)
    subject_sig = subject_text
    context_id = f"{schema_id}_{subject_sig}"
    row_id = f"v2|{candidate.regime}|{candidate.subtask}|{context_id}|{candidate.side}|{candidate.lemma}|{row_index}"
    profile = candidate.profile
    return {
        "row_id": row_id,
        "regime": candidate.regime,
        "subtask": candidate.subtask,
        "template_id": f"{frame_label}_{schema_id}",
        "context_id": context_id,
        "context_schema_id": schema_id,
        "template_prompt": template,
        "prefix_family": frame_label,
        "prompt": prompt,
        "verb": candidate.lemma,
        "source_surface": candidate.source_form,
        "source_tag": "",
        "base_frame": profile.category_2 if profile else "",
        "src_lemma": profile.root if profile else "",
        "side": candidate.side,
        "licensing_label": good_label if candidate.side == "good" else bad_label,
        "frame_label": frame_label,
        "expected_target": expected_target,
        "contrast_target": contrast_target,
        "source_idx": candidate.source_idx,
        "source_text": candidate.source_text,
        "source_zipf": candidate.source_zipf,
        "source_zipf_regime": candidate.source_zipf_regime,
        "inventory_source": candidate.inventory_source,
        "curation_reason": candidate.curation_reason,
        "subject": subject_text,
        "subject_class": "overlay_arg_1_match",
        "subject_role_probe": "arg_1",
        "subject_source": "freqblimp_vocabulary_overlay",
        "subject_overlay_features": {field: subject.get(field, "") for field in NOUN_FIELDS if subject.get(field, "")},
        "object": object_text,
        "object_class": "overlay_arg_2_match" if obj else "",
        "object_role_probe": "arg_2" if obj else "",
        "object_source": "freqblimp_vocabulary_overlay" if obj else "",
        "object_overlay_features": {field: obj.get(field, "") for field in NOUN_FIELDS if obj and obj.get(field, "")},
        "verb_arg_1": profile.arg_1 if profile else "",
        "verb_arg_2": profile.arg_2 if profile else "",
        "semantic_fit_source": "freqblimp_overlay_arg_constraints",
        "split_lemma_key": candidate.lemma,
        "control_type": "observed_verb",
        "intervention_anchor": "verb_final_subtoken",
        "anchor_index_policy": "per_example_last_prompt_token",
        "decision_target_is_proxy": True,
        "alignment_status": "unverified",
    }


def build_rows(
    candidates: list[VerbCandidate],
    nouns: list[dict[str, str]],
    target_per_side_context: int | None,
    subject_variants_per_context: int,
    seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[VerbCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[(candidate.regime, candidate.subtask, candidate.side)].append(candidate)

    rows: list[dict[str, Any]] = []
    row_index = 0
    for (regime, subtask, side), group_candidates in sorted(grouped.items()):
        sampled = deterministic_sample(group_candidates, target_per_side_context, seed)
        for schema_id, template in contexts_for_subtask(subtask):
            # Keep subjects shared across good/bad sides for the core v2
            # causative/inchoative contrasts; per-verb role constraints are
            # still recorded on each row.
            if subtask == "causative":
                subject_condition = "animate=1"
            elif subtask == "inchoative":
                subject_condition = "physical=1^animate=0"
            else:
                profile_conditions = [c.profile.arg_1 for c in sampled if c.profile and c.profile.arg_1]
                subject_condition = profile_conditions[0] if profile_conditions else "noun=1"
            subjects = sample_matching_nouns(
                nouns,
                subject_condition,
                subject_variants_per_context,
                f"{regime}|{subtask}|{schema_id}|subject",
                seed,
            )
            for variant_idx, subject in enumerate(subjects):
                for candidate in sampled:
                    obj = None
                    if candidate.profile and candidate.profile.arg_2:
                        objects = sample_matching_nouns(
                            nouns,
                            candidate.profile.arg_2,
                            1,
                            f"{candidate.lemma}|{schema_id}|object|{variant_idx}",
                            seed,
                        )
                        obj = objects[0] if objects else None
                    rows.append(make_row(candidate, f"{schema_id}_s{variant_idx:02d}", template, subject, obj, row_index))
                    row_index += 1
    return rows


def load_tokenizer(model_name: str, local_files_only: bool) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install transformers to run tokenizer verification.") from exc
    return AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)


def verify_rows(rows: list[dict[str, Any]], model_name: str, local_files_only: bool) -> dict[str, Any]:
    tokenizer = load_tokenizer(model_name, local_files_only=local_files_only)
    summary: dict[str, Any] = {"model": model_name, "groups": {}}
    target_token_lengths = {}
    for target in sorted({r["expected_target"] for r in rows} | {r["contrast_target"] for r in rows}):
        target_token_lengths[target] = len(tokenizer.encode(target, add_special_tokens=False))

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["regime"], row["subtask"], row["template_id"])].append(row)

    for group_key, group_rows in grouped.items():
        prompt_counts = Counter()
        prefix_counts = Counter()
        verb_region_counts = Counter()
        for row in group_rows:
            prompt = row["prompt"]
            prefix = prompt[: -len(row["verb"])] if prompt.endswith(row["verb"]) else row["template_prompt"].split("{verb}")[0]
            prefix_len = len(tokenizer.encode(prefix, add_special_tokens=False))
            prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
            verb_region_len = len(tokenizer.encode(" " + row["verb"], add_special_tokens=False))
            row["tokenizer_model"] = model_name
            row["prefix_token_count"] = prefix_len
            row["prompt_token_count"] = prompt_len
            row["verb_region_token_count"] = verb_region_len
            row["anchor_token_index"] = prompt_len - 1
            row["decision_token_index"] = prompt_len - 1
            row["expected_target_token_count"] = target_token_lengths[row["expected_target"]]
            row["contrast_target_token_count"] = target_token_lengths[row["contrast_target"]]
            prompt_counts[prompt_len] += 1
            prefix_counts[prefix_len] += 1
            verb_region_counts[verb_region_len] += 1

        retained_prompt_len, retained_n = prompt_counts.most_common(1)[0]
        retained_prefix_len, _ = prefix_counts.most_common(1)[0]
        for row in group_rows:
            aligned = (
                row["prompt_token_count"] == retained_prompt_len
                and row["prefix_token_count"] == retained_prefix_len
                and row["expected_target_token_count"] == 1
                and row["contrast_target_token_count"] == 1
            )
            row["alignment_status"] = "aligned" if aligned else "rejected_tokenization"
        summary["groups"]["|".join(group_key)] = {
            "n": len(group_rows),
            "retained_n": retained_n,
            "retained_prompt_token_count": retained_prompt_len,
            "decision_token_index": retained_prompt_len - 1,
            "prefix_token_counts": dict(prefix_counts),
            "prompt_token_counts": dict(prompt_counts),
            "verb_region_token_counts": dict(verb_region_counts),
        }
    summary["target_token_lengths"] = target_token_lengths
    return summary


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_report(path: Path, candidates: list[VerbCandidate], rows: list[dict[str, Any]], summary: dict[str, Any] | None) -> None:
    candidate_counts = Counter((c.regime, c.subtask, c.side) for c in candidates)
    candidate_source_counts = Counter((c.regime, c.subtask, c.side, c.inventory_source) for c in candidates)
    candidate_band_counts = Counter((c.regime, c.subtask, c.side, c.source_zipf_regime) for c in candidates)
    row_counts = Counter((r["regime"], r["subtask"], r["side"], r["context_schema_id"], r["alignment_status"]) for r in rows)
    subject_counts = Counter((r["subject"], r["subject_class"], r["subject_source"]) for r in rows)
    lines = [
        "# V2 Aligned Template Build Report",
        "",
        "## Design Notes",
        "",
        "- Causative/inchoative verbs come from curated tuples in `freq-blimp/generation_projects/blimp/overlay_guards.py`.",
        "- Regime membership is assigned by Zipf bands `head=3.5-5.5` and `low=1.2-3.2`.",
        "- Low-frequency rows retain `source_zipf_regime` metadata: `xtail=1.2-2.2`, `tail=2.4-3.2`, and `low_gap=2.2-2.4`.",
        "- Supplemental v2 bad-side verbs are included from the repo-local JSON inventory after the same Zipf-band check.",
        "- Non-curated subtasks, when requested, fall back to canonical `freq-blimp` JSONL outputs.",
        "- Subject/object fillers come from `freq-blimp/vocabulary_overlay.csv`.",
        "- The archived `blimp-rare` swapping pipeline is not used.",
        "- Intervention anchor: `verb_final_subtoken`.",
        "- Subject/object fillers are sampled from overlay rows matching `arg_1`/`arg_2` constraints.",
        "- Noun sampling prefers `frequent=1` and `sg=1` overlay rows but falls back to the full role-matching pool.",
        "- Each context family is generated with both target labels.",
        "- Counts below distinguish row count from unique verified lemma count.",
        "",
        "## Candidate Verb Inventory",
        "",
    ]
    for key, value in sorted(candidate_counts.items()):
        lines.append(f"- `{key[0]}` `{key[1]}` `{key[2]}`: {value} unique candidate lemmas")

    lines.extend(["", "## Candidate Inventory Sources", ""])
    for key, value in sorted(candidate_source_counts.items()):
        regime, subtask, side, inventory_source = key
        lines.append(f"- `{regime}` `{subtask}` `{side}` `{inventory_source}`: {value}")

    lines.extend(["", "## Candidate Source Frequency Bands", ""])
    for key, value in sorted(candidate_band_counts.items()):
        regime, subtask, side, source_zipf_regime = key
        lines.append(f"- `{regime}` `{subtask}` `{side}` `{source_zipf_regime}`: {value}")

    lines.extend(["", "## Rows By Context Schema", ""])
    for key, value in sorted(row_counts.items()):
        regime, subtask, side, context_schema_id, status = key
        lines.append(f"- `{regime}` `{subtask}` `{side}` `{context_schema_id}` `{status}`: {value}")

    lines.extend(["", "## Sampled Subjects", ""])
    for (subject, subject_class, subject_source), value in sorted(subject_counts.items())[:100]:
        lines.append(f"- `{subject}` `{subject_class}` `{subject_source}`: {value}")

    if summary is not None:
        lines.extend(["", "## Tokenization Summary", ""])
        lines.append(f"- model: `{summary['model']}`")
        lines.append(f"- target token lengths: `{summary['target_token_lengths']}`")
        for group, payload in sorted(summary["groups"].items()):
            lines.append(
                f"- `{group}`: retained {payload['retained_n']}/{payload['n']} "
                f"at prompt_token_count={payload['retained_prompt_token_count']}; "
                f"all prompt counts={payload['prompt_token_counts']}; "
                f"verb-region counts={payload['verb_region_token_counts']}"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subtasks", default="causative,inchoative,transitive,intransitive,drop_argument")
    ap.add_argument("--target-per-side-context", type=int, default=100)
    ap.add_argument("--subject-variants-per-context", type=int, default=4)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--overlay-csv", default=str(OVERLAY_CSV))
    ap.add_argument("--supplemental-bad-verbs", default=str(SUPPLEMENTAL_BAD_VERBS))
    ap.add_argument("--model", default=None, help="Optional tokenizer model for alignment verification.")
    ap.add_argument("--allow-tokenizer-download", action="store_true")
    ap.add_argument("--out", default="data/aligned_templates_v2/lexical_licensing_v2_candidates.jsonl")
    ap.add_argument("--aligned-out", default="data/aligned_templates_v2/lexical_licensing_v2_aligned.jsonl")
    ap.add_argument("--report", default="reports/aligned_template_v2_report.md")
    args = ap.parse_args()

    subtasks = {part.strip() for part in args.subtasks.split(",") if part.strip()}
    unknown = subtasks - SUBTASKS
    if unknown:
        raise SystemExit(f"Unknown subtasks: {sorted(unknown)}")

    form_to_lemma, profiles, nouns = load_overlay(Path(args.overlay_csv))
    candidates = collect_candidates(subtasks, form_to_lemma, profiles, Path(args.supplemental_bad_verbs))
    rows = build_rows(candidates, nouns, args.target_per_side_context, args.subject_variants_per_context, args.seed)
    summary = (
        verify_rows(rows, args.model, local_files_only=not args.allow_tokenizer_download)
        if args.model
        else None
    )

    write_jsonl(Path(args.out), rows)
    if args.model:
        write_jsonl(Path(args.aligned_out), [r for r in rows if r["alignment_status"] == "aligned"])
    write_report(Path(args.report), candidates, rows, summary)
    print(args.out)
    if args.model:
        print(args.aligned_out)
    print(args.report)


if __name__ == "__main__":
    main()
