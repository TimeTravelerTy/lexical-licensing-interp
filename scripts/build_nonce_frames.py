#!/usr/bin/env python3
"""Build nonce-frame prompt pairs for frame-conditioned object expectation.

The probe string is identical for T+ and T- for a given nonce. Tokenization is
filtered only on the probe surface form, because the measurement is made at the
same probe-final nonce token in both frame conditions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable

from run_pythia_attribution_patching import DEFAULT_MODEL


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAYON_NUM_THREADS", "1")

DEFAULT_OUT_DIR = "data/nonce_frames"
DEFAULT_FRAMES = "data/nonce_frames/nonce_frames.jsonl"
DEFAULT_STEMS = "data/nonce_frames/nonce_stems.jsonl"
DEFAULT_BLOCKLISTS = (
    "data/nonce_frames/nonce_blocklist.txt",
    "/usr/share/dict/words",
    "/usr/dict/words",
)

ONSETS = (
    "b",
    "bl",
    "br",
    "ch",
    "d",
    "dr",
    "f",
    "fl",
    "g",
    "gl",
    "k",
    "kl",
    "m",
    "n",
    "p",
    "pl",
    "s",
    "sk",
    "sl",
    "t",
    "tr",
    "v",
    "z",
)
NUCLEI = ("a", "e", "i", "o", "u", "ai", "ee", "oo")
CODAS = ("b", "ck", "d", "f", "g", "k", "l", "m", "n", "p", "sh", "t", "v", "x", "z")
SUFFIXES = ("et", "en", "le", "ip", "op", "up", "iv", "ev", "om", "um")
WORD_RE = re.compile(r"^[a-z]+$")

PRIMING_SUBJECTS = (
    ("artist", "animate"),
    ("door", "inanimate"),
    ("worker", "animate"),
    ("glass", "inanimate"),
    ("teacher", "animate"),
    ("rope", "inanimate"),
)
PROBE_SUBJECTS = (
    ("chef", "animate"),
    ("box", "inanimate"),
    ("driver", "animate"),
    ("paper", "inanimate"),
    ("student", "animate"),
    ("shirt", "inanimate"),
)
OBJECTS = ("vase", "box", "cup", "plate", "toy", "rope", "lock", "towel")
LEAD_SCHEMAS = (
    "In the lab, the {subject} {past}",
    "During the check, the {subject} {past}",
    "After the visit, the {subject} {past}",
)
PROBE_SCHEMAS = (
    "Tomorrow, the {subject} will {lemma}",
    "Later today, the {subject} will {lemma}",
    "During the trial, the {subject} will {lemma}",
)


def stable_unit(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def stem_sort_key(stem: str, seed: int) -> tuple[float, str]:
    return stable_unit(stem, seed), stem


def suffix_for(stem: str) -> str:
    for suffix in sorted(SUFFIXES, key=len, reverse=True):
        if stem.endswith(suffix):
            return suffix
    return ""


def target_suffix_counts(n_stems: int) -> dict[str, int]:
    base = n_stems // len(SUFFIXES)
    remainder = n_stems % len(SUFFIXES)
    return {
        suffix: base + (1 if index < remainder else 0)
        for index, suffix in enumerate(SUFFIXES)
    }


def normalize_word(value: str) -> str:
    return value.strip().lower()


def parse_path_arg(value: str) -> tuple[Path, ...]:
    return tuple(Path(part.strip()) for part in value.split(",") if part.strip())


def load_blocklist(paths: Iterable[Path]) -> tuple[set[str], list[str]]:
    words: set[str] = set()
    loaded: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        loaded.append(str(path))
        with path.open(encoding="utf-8", errors="ignore") as f:
            for line in f:
                word = normalize_word(line)
                if WORD_RE.fullmatch(word):
                    words.add(word)
    return words, loaded


def deletion_signatures(words: set[str]) -> dict[int, set[str]]:
    signatures: dict[int, set[str]] = {}
    for word in words:
        if not 3 <= len(word) <= 9:
            continue
        bucket = signatures.setdefault(len(word), set())
        for idx in range(len(word)):
            bucket.add(word[:idx] + word[idx + 1 :])
    return signatures


def near_blocklisted_word(word: str, blocklist: set[str], signatures: dict[int, set[str]]) -> bool:
    if word in blocklist:
        return True
    if any(word[:idx] + word[idx + 1 :] in blocklist for idx in range(len(word))):
        return True
    return any(word[:idx] + word[idx + 1 :] in signatures.get(len(word), set()) for idx in range(len(word)))


def blocklist_reasons(stem: str, blocklist: set[str], signatures: dict[int, set[str]]) -> list[str]:
    reasons: list[str] = []
    if stem in blocklist:
        reasons.append("blocklisted_surface")
    elif near_blocklisted_word(stem, blocklist, signatures):
        reasons.append("near_blocklisted_surface")
    past = regular_past(stem)
    if past in blocklist:
        reasons.append("blocklisted_regular_past")
    for suffix in SUFFIXES:
        if not stem.endswith(suffix):
            continue
        base = stem[: -len(suffix)]
        if len(base) >= 3 and base in blocklist:
            reasons.append(f"english_base_plus_{suffix}")
        elif len(base) >= 4 and near_blocklisted_word(base, blocklist, signatures):
            reasons.append(f"near_english_base_plus_{suffix}")
    return reasons


def iter_candidate_stems(blocklist: set[str]) -> Iterable[tuple[str, list[str]]]:
    signatures = deletion_signatures(blocklist)
    seen: set[str] = set()
    for onset in ONSETS:
        for nucleus in NUCLEI:
            for coda in CODAS:
                for suffix in SUFFIXES:
                    stem = f"{onset}{nucleus}{coda}{suffix}"
                    if len(stem) < 3 or len(stem) > 9:
                        continue
                    if stem in seen:
                        continue
                    if stem.endswith(("eed", "aid", "ood")):
                        continue
                    seen.add(stem)
                    yield stem, blocklist_reasons(stem, blocklist, signatures)


def regular_past(stem: str) -> str:
    if stem.endswith("e"):
        return f"{stem}d"
    return f"{stem}ed"


def tokenization_record(
    tokenizer: Any,
    stem: str,
    max_probe_tokens: int,
    min_probe_tokens: int,
    lexical_reject_reasons: list[str],
) -> dict[str, Any]:
    probe_surface = f" {stem}"
    ids = tokenizer.encode(probe_surface, add_special_tokens=False)
    pieces = tokenizer.convert_ids_to_tokens(ids)
    decoded_pieces = [tokenizer.decode([token_id]) for token_id in ids]
    stripped_decoded = [piece.strip() for piece in decoded_pieces]

    reasons: list[str] = list(lexical_reject_reasons)
    if len(ids) < min_probe_tokens:
        reasons.append("too_few_probe_tokens")
    if len(ids) > max_probe_tokens:
        reasons.append("too_many_probe_tokens")
    if any(piece == "" for piece in stripped_decoded):
        reasons.append("empty_decoded_piece")
    if any("\ufffd" in piece for piece in decoded_pieces):
        reasons.append("replacement_char_piece")
    if any(piece.startswith("<0x") for piece in pieces):
        reasons.append("byte_fallback_piece")
    alpha_lengths = [len("".join(ch for ch in piece if ch.isalpha())) for piece in stripped_decoded]
    if any(length == 1 for length in alpha_lengths) and len(ids) > 1:
        reasons.append("single_letter_fragment")
    if len(ids) > 1 and max(alpha_lengths or [0]) <= 2:
        reasons.append("mushy_short_fragments")

    return {
        "lemma": stem,
        "suffix": suffix_for(stem),
        "probe_surface": probe_surface,
        "probe_token_ids": ids,
        "probe_token_pieces": pieces,
        "probe_decoded_pieces": decoded_pieces,
        "probe_token_count": len(ids),
        "accepted": not reasons,
        "selected": False,
        "reject_reasons": reasons,
    }


def pick_index(key: str, n: int, seed: int) -> int:
    return int(math.floor(stable_unit(key, seed) * n)) % n


def lead_sentences(lemma: str, frame: str, seed: int, n_leads: int) -> tuple[list[str], list[dict[str, str]]]:
    rows: list[str] = []
    subjects: list[dict[str, str]] = []
    start_subject = pick_index(f"{lemma}|{frame}|lead_subject", len(PRIMING_SUBJECTS), seed)
    start_object = pick_index(f"{lemma}|{frame}|object", len(OBJECTS), seed)
    past = regular_past(lemma)
    for i in range(n_leads):
        subject, animacy = PRIMING_SUBJECTS[(start_subject + i) % len(PRIMING_SUBJECTS)]
        obj = OBJECTS[(start_object + i) % len(OBJECTS)]
        schema = LEAD_SCHEMAS[i % len(LEAD_SCHEMAS)]
        core = schema.format(subject=subject, past=past)
        if frame == "T+":
            sentence = f"{core} the {obj}."
        elif frame == "T-":
            sentence = f"{core}."
        else:
            raise ValueError(f"unknown frame: {frame}")
        rows.append(sentence)
        subjects.append({"subject": subject, "animacy": animacy, "object": obj if frame == "T+" else ""})
    return rows, subjects


def build_rows(
    tokenizer: Any,
    accepted_stems: list[dict[str, Any]],
    seed: int,
    n_leads: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stem_index, stem_record in enumerate(accepted_stems):
        lemma = str(stem_record["lemma"])
        probe_subject, probe_animacy = PROBE_SUBJECTS[
            pick_index(f"{lemma}|probe_subject", len(PROBE_SUBJECTS), seed)
        ]
        probe_schema = PROBE_SCHEMAS[pick_index(f"{lemma}|probe_schema", len(PROBE_SCHEMAS), seed)]
        probe_text = probe_schema.format(subject=probe_subject, lemma=lemma)
        probe_context_ids = tokenizer.encode(f" {probe_text}", add_special_tokens=False)
        probe_token_count = len(probe_context_ids)
        for frame in ("T+", "T-"):
            leads, lead_subjects = lead_sentences(lemma, frame, seed, n_leads)
            priming_text = " ".join(leads)
            full_context = f"{priming_text} {probe_text}"
            full_ids = tokenizer.encode(full_context, add_special_tokens=False)
            if len(full_ids) < len(probe_context_ids) or full_ids[-len(probe_context_ids) :] != probe_context_ids:
                raise RuntimeError(f"probe text is not a suffix after tokenization for lemma={lemma} frame={frame}")
            row_id = f"nonce|{stem_index:04d}|{lemma}|{frame}"
            rows.append(
                {
                    "row_id": row_id,
                    "lemma": lemma,
                    "stem_index": stem_index,
                    "frame": frame,
                    "priming_text": priming_text,
                    "probe_text": probe_text,
                    "full_context": full_context,
                    "verb_final_tok_idx": len(full_ids) - 1,
                    "full_context_token_count": len(full_ids),
                    "probe_token_count": probe_token_count,
                    "probe_surface": stem_record["probe_surface"],
                    "probe_form_token_count": stem_record["probe_token_count"],
                    "probe_form_token_ids": stem_record["probe_token_ids"],
                    "probe_form_token_pieces": stem_record["probe_token_pieces"],
                    "probe_subject": probe_subject,
                    "probe_subject_animacy": probe_animacy,
                    "lead_subjects": lead_subjects,
                    "n_leads": n_leads,
                    "tokenizer_model": tokenizer.name_or_path,
                    "alignment_status": "aligned",
                    "intervention_anchor": "verb_final_subtoken",
                }
            )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def run(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=not args.allow_download)
    blocklist, blocklist_paths_loaded = load_blocklist(parse_path_arg(args.blocklists))
    candidates = sorted(iter_candidate_stems(blocklist), key=lambda item: stem_sort_key(item[0], args.seed))
    suffix_targets = target_suffix_counts(args.n_stems)
    suffix_selected_counts = {suffix: 0 for suffix in SUFFIXES}

    stem_records: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    for stem, lexical_reject_reasons in candidates:
        record = tokenization_record(
            tokenizer,
            stem,
            args.max_probe_tokens,
            args.min_probe_tokens,
            lexical_reject_reasons,
        )
        stem_records.append(record)
        suffix = str(record["suffix"])
        if record["accepted"] and suffix_selected_counts.get(suffix, 0) < suffix_targets.get(suffix, 0):
            record["selected"] = True
            selected.append(record)
            suffix_selected_counts[suffix] += 1
        if len(selected) >= args.n_stems:
            break
    if len(selected) < args.n_stems:
        raise SystemExit(f"Only selected {len(selected)} stems under suffix quotas; requested {args.n_stems}.")

    rows = build_rows(tokenizer, selected, args.seed, args.n_leads)
    stem_path = Path(args.stem_out)
    frame_path = Path(args.out)
    write_jsonl(stem_path, stem_records)
    write_jsonl(frame_path, rows)
    manifest = {
        "model": args.model,
        "n_stems_requested": args.n_stems,
        "n_stems_selected": len(selected),
        "n_stem_records": len(stem_records),
        "n_frame_rows": len(rows),
        "suffix_targets": suffix_targets,
        "suffix_selected_counts": suffix_selected_counts,
        "frames": ["T+", "T-"],
        "n_leads": args.n_leads,
        "seed": args.seed,
        "max_probe_tokens": args.max_probe_tokens,
        "min_probe_tokens": args.min_probe_tokens,
        "blocklist_paths_requested": [str(path) for path in parse_path_arg(args.blocklists)],
        "blocklist_paths_loaded": blocklist_paths_loaded,
        "blocklist_word_count": len(blocklist),
        "stem_jsonl": str(stem_path),
        "frame_jsonl": str(frame_path),
        "note": "Nonce stems are filtered only on the probe surface form; T+/T- share the identical probe text.",
    }
    manifest_path = frame_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n-stems", type=int, default=80)
    ap.add_argument("--n-leads", type=int, default=3)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--max-probe-tokens", type=int, default=4)
    ap.add_argument("--min-probe-tokens", type=int, default=1)
    ap.add_argument("--blocklists", default=",".join(DEFAULT_BLOCKLISTS))
    ap.add_argument("--allow-download", action="store_true")
    ap.add_argument("--out", default=DEFAULT_FRAMES)
    ap.add_argument("--stem-out", default=DEFAULT_STEMS)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
