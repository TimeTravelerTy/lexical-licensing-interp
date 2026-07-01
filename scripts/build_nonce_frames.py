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
from pathlib import Path
from typing import Any, Iterable

from run_pythia_attribution_patching import DEFAULT_MODEL


DEFAULT_OUT_DIR = "data/nonce_frames"
DEFAULT_FRAMES = "data/nonce_frames/nonce_frames.jsonl"
DEFAULT_STEMS = "data/nonce_frames/nonce_stems.jsonl"

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
SUFFIXES = ("", "et", "en", "le", "ip", "op", "up")

RESERVED_WORDS = {
    "be",
    "do",
    "go",
    "make",
    "take",
    "have",
    "get",
    "put",
    "run",
    "set",
    "let",
    "cut",
    "hit",
    "bet",
    "bid",
    "fit",
    "bit",
    "sit",
    "kit",
    "pit",
    "bat",
    "cat",
    "dog",
    "box",
    "cup",
    "toy",
}

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


def iter_candidate_stems() -> Iterable[str]:
    manual = (
        "blicket",
        "dax",
        "wug",
        "glorp",
        "mip",
        "norp",
        "plim",
        "sprock",
        "tav",
        "zindle",
    )
    seen: set[str] = set()
    for stem in manual:
        if stem not in RESERVED_WORDS:
            seen.add(stem)
            yield stem
    for onset in ONSETS:
        for nucleus in NUCLEI:
            for coda in CODAS:
                for suffix in SUFFIXES:
                    stem = f"{onset}{nucleus}{coda}{suffix}"
                    if len(stem) < 3 or len(stem) > 9:
                        continue
                    if stem in RESERVED_WORDS or stem in seen:
                        continue
                    if stem.endswith(("eed", "aid", "ood")):
                        continue
                    seen.add(stem)
                    yield stem


def regular_past(stem: str) -> str:
    if stem.endswith("e"):
        return f"{stem}d"
    return f"{stem}ed"


def tokenization_record(tokenizer: Any, stem: str, max_probe_tokens: int, min_probe_tokens: int) -> dict[str, Any]:
    probe_surface = f" {stem}"
    ids = tokenizer.encode(probe_surface, add_special_tokens=False)
    pieces = tokenizer.convert_ids_to_tokens(ids)
    decoded_pieces = [tokenizer.decode([token_id]) for token_id in ids]
    stripped_decoded = [piece.strip() for piece in decoded_pieces]

    reasons: list[str] = []
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
        "probe_surface": probe_surface,
        "probe_token_ids": ids,
        "probe_token_pieces": pieces,
        "probe_decoded_pieces": decoded_pieces,
        "probe_token_count": len(ids),
        "accepted": not reasons,
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
        probe_token_count = len(tokenizer.encode(probe_text, add_special_tokens=False))
        for frame in ("T+", "T-"):
            leads, lead_subjects = lead_sentences(lemma, frame, seed, n_leads)
            priming_text = " ".join(leads)
            full_context = f"{priming_text} {probe_text}"
            full_ids = tokenizer.encode(full_context, add_special_tokens=False)
            probe_ids = tokenizer.encode(probe_text, add_special_tokens=False)
            if len(full_ids) < len(probe_ids) or full_ids[-len(probe_ids) :] != probe_ids:
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
    candidates = sorted(iter_candidate_stems(), key=lambda stem: stem_sort_key(stem, args.seed))

    stem_records: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    for stem in candidates:
        record = tokenization_record(tokenizer, stem, args.max_probe_tokens, args.min_probe_tokens)
        stem_records.append(record)
        if record["accepted"]:
            accepted.append(record)
        if len(accepted) >= args.n_stems:
            break
    if len(accepted) < args.n_stems:
        raise SystemExit(f"Only accepted {len(accepted)} stems; requested {args.n_stems}.")

    rows = build_rows(tokenizer, accepted, args.seed, args.n_leads)
    stem_path = Path(args.stem_out)
    frame_path = Path(args.out)
    write_jsonl(stem_path, stem_records)
    write_jsonl(frame_path, rows)
    manifest = {
        "model": args.model,
        "n_stems_requested": args.n_stems,
        "n_stems_accepted": len(accepted),
        "n_stem_records": len(stem_records),
        "n_frame_rows": len(rows),
        "frames": ["T+", "T-"],
        "n_leads": args.n_leads,
        "seed": args.seed,
        "max_probe_tokens": args.max_probe_tokens,
        "min_probe_tokens": args.min_probe_tokens,
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
    ap.add_argument("--allow-download", action="store_true")
    ap.add_argument("--out", default=DEFAULT_FRAMES)
    ap.add_argument("--stem-out", default=DEFAULT_STEMS)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
