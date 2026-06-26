#!/usr/bin/env python3
"""Whole-pair DAS injection and read-side projection tests for v2 rows.

The intervention site is the prompt-final verb token. A causal intervention at
that site cannot affect the model's probability for the verb token itself, so
the optimized injection metric is the post-anchor suffix log probability of the
full sentence. The output also reports full-sentence log probabilities for
comparison; only the suffix component is intervention-sensitive.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from run_pythia_attribution_patching import DEFAULT_MODEL, parse_csv_arg, shell_safe_slug, stable_unit
from run_pythia_exact_patching import normalize_model_output, parse_site, replace_model_output, site_name
from run_pythia_whole_pair_lp import build_pairs, load_rows


DEFAULT_DATA = "data/aligned_templates_v2/lexical_licensing_v2_aligned.jsonl"
DEFAULT_SUBTASKS = ("causative", "inchoative")
DEFAULT_REGIMES = ("head", "low")


@dataclass(frozen=True)
class WholePairDirected:
    pair_key: str
    regime: str
    subtask: str
    template_id: str
    context_id: str
    source_idx: int
    direction: str
    clean_side: str
    corrupt_side: str
    objective_clean_side: str
    objective_corrupt_side: str
    clean_sentence: str
    corrupt_sentence: str
    clean_prompt: str
    corrupt_prompt: str
    anchor_token_index: int
    prompt_token_count: int
    clean_verb: str
    corrupt_verb: str
    subject: str
    subject_class: str
    clean_source_zipf_regime: str
    corrupt_source_zipf_regime: str
    clean_inventory_source: str
    corrupt_inventory_source: str
    frame_continuation: str
    whole_pair_object: str
    clean_row_id: str
    corrupt_row_id: str


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return math.nan
    mu = mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))


def sem(values: list[float]) -> float:
    return stdev(values) / math.sqrt(len(values)) if len(values) >= 2 else math.nan


def side_sign(side: str) -> float:
    if side == "good":
        return 1.0
    if side == "bad":
        return -1.0
    raise ValueError(f"unknown side: {side}")


def binary_auc(values: list[float], labels: list[str], positive_label: str) -> float:
    pos = [v for v, label in zip(values, labels) if label == positive_label]
    neg = [v for v, label in zip(values, labels) if label != positive_label]
    if not pos or not neg:
        return math.nan
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def best_threshold_accuracy(values: list[float], labels: list[str], positive_label: str) -> tuple[float, float, str]:
    if not values:
        return math.nan, math.nan, ""
    unique = sorted(set(values))
    if len(unique) == 1:
        thresholds = unique
    else:
        thresholds = [unique[0] - 1.0]
        thresholds.extend((a + b) / 2.0 for a, b in zip(unique, unique[1:]))
        thresholds.append(unique[-1] + 1.0)
    best = (-1.0, thresholds[0], "positive_high")
    for threshold in thresholds:
        for direction in ("positive_high", "positive_low"):
            correct = 0
            for value, label in zip(values, labels):
                if direction == "positive_high":
                    pred = positive_label if value >= threshold else "__negative__"
                else:
                    pred = positive_label if value <= threshold else "__negative__"
                actual = positive_label if label == positive_label else "__negative__"
                correct += int(pred == actual)
            acc = correct / len(values)
            if acc > best[0]:
                best = (acc, threshold, direction)
    return best


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_directed_pairs(
    whole_pairs: list[dict[str, Any]],
    directions: set[str],
    control: str,
    seed: int,
) -> list[WholePairDirected]:
    directed: list[WholePairDirected] = []
    for index, pair in enumerate(whole_pairs):
        pair_key = str(pair["pair_key"])

        def objective_sides(clean_side: str, corrupt_side: str, direction: str) -> tuple[str, str]:
            if control != "shuffled_label":
                return clean_side, corrupt_side
            should_swap = stable_unit(f"{pair_key}|{direction}|objective", seed) < 0.5
            return (corrupt_side, clean_side) if should_swap else (clean_side, corrupt_side)

        if "good_to_bad" in directions:
            obj_clean, obj_corrupt = objective_sides("good", "bad", "good_to_bad")
            directed.append(
                WholePairDirected(
                    pair_key=pair_key,
                    regime=str(pair["regime"]),
                    subtask=str(pair["subtask"]),
                    template_id=str(pair["template_id"]),
                    context_id=str(pair["context_id"]),
                    source_idx=index,
                    direction="good_to_bad",
                    clean_side="good",
                    corrupt_side="bad",
                    objective_clean_side=obj_clean,
                    objective_corrupt_side=obj_corrupt,
                    clean_sentence=str(pair["good_sentence"]),
                    corrupt_sentence=str(pair["bad_sentence"]),
                    clean_prompt=str(pair["good_prompt"]),
                    corrupt_prompt=str(pair["bad_prompt"]),
                    anchor_token_index=int(pair["good_anchor_token_index"]),
                    prompt_token_count=int(pair["good_anchor_token_index"]) + 1,
                    clean_verb=str(pair["good_verb"]),
                    corrupt_verb=str(pair["bad_verb"]),
                    subject=str(pair["subject"]),
                    subject_class=str(pair["subject_class"]),
                    clean_source_zipf_regime=str(pair["good_source_zipf_regime"]),
                    corrupt_source_zipf_regime=str(pair["bad_source_zipf_regime"]),
                    clean_inventory_source=str(pair["good_inventory_source"]),
                    corrupt_inventory_source=str(pair["bad_inventory_source"]),
                    frame_continuation=str(pair["frame_continuation"]),
                    whole_pair_object=str(pair["whole_pair_object"]),
                    clean_row_id=str(pair["good_row_id"]),
                    corrupt_row_id=str(pair["bad_row_id"]),
                )
            )
        if "bad_to_good" in directions:
            obj_clean, obj_corrupt = objective_sides("bad", "good", "bad_to_good")
            directed.append(
                WholePairDirected(
                    pair_key=pair_key,
                    regime=str(pair["regime"]),
                    subtask=str(pair["subtask"]),
                    template_id=str(pair["template_id"]),
                    context_id=str(pair["context_id"]),
                    source_idx=index,
                    direction="bad_to_good",
                    clean_side="bad",
                    corrupt_side="good",
                    objective_clean_side=obj_clean,
                    objective_corrupt_side=obj_corrupt,
                    clean_sentence=str(pair["bad_sentence"]),
                    corrupt_sentence=str(pair["good_sentence"]),
                    clean_prompt=str(pair["bad_prompt"]),
                    corrupt_prompt=str(pair["good_prompt"]),
                    anchor_token_index=int(pair["bad_anchor_token_index"]),
                    prompt_token_count=int(pair["bad_anchor_token_index"]) + 1,
                    clean_verb=str(pair["bad_verb"]),
                    corrupt_verb=str(pair["good_verb"]),
                    subject=str(pair["subject"]),
                    subject_class=str(pair["subject_class"]),
                    clean_source_zipf_regime=str(pair["bad_source_zipf_regime"]),
                    corrupt_source_zipf_regime=str(pair["good_source_zipf_regime"]),
                    clean_inventory_source=str(pair["bad_inventory_source"]),
                    corrupt_inventory_source=str(pair["good_inventory_source"]),
                    frame_continuation=str(pair["frame_continuation"]),
                    whole_pair_object=str(pair["whole_pair_object"]),
                    clean_row_id=str(pair["bad_row_id"]),
                    corrupt_row_id=str(pair["good_row_id"]),
                )
            )
    return directed


def split_key(pair: WholePairDirected, split_by: str) -> str:
    if split_by == "pair":
        return pair.pair_key
    if split_by == "lemma_pair":
        return "|".join(sorted((pair.clean_verb, pair.corrupt_verb)))
    raise ValueError(f"unknown split policy: {split_by}")


def make_transfer_splits(
    pairs: list[WholePairDirected],
    train_regimes: set[str],
    eval_regimes: set[str],
    eval_frac: float,
    split_by: str,
    seed: int,
) -> tuple[list[WholePairDirected], list[WholePairDirected]]:
    train: list[WholePairDirected] = []
    eval_: list[WholePairDirected] = []
    for pair in pairs:
        in_train = pair.regime in train_regimes
        in_eval = pair.regime in eval_regimes
        if in_train and in_eval:
            if stable_unit(split_key(pair, split_by), seed) < eval_frac:
                eval_.append(pair)
            else:
                train.append(pair)
        elif in_train:
            train.append(pair)
        elif in_eval:
            eval_.append(pair)
    if not train or not eval_:
        raise SystemExit(f"Empty split: train={len(train)} eval={len(eval_)}")
    return train, eval_


def retokenize_pairs(tokenizer: Any, pairs: list[WholePairDirected]) -> tuple[list[WholePairDirected], dict[str, int]]:
    checked: list[WholePairDirected] = []
    skips: dict[str, int] = defaultdict(int)
    for pair in pairs:
        clean_prompt_len = len(tokenizer.encode(pair.clean_prompt, add_special_tokens=False))
        corrupt_prompt_len = len(tokenizer.encode(pair.corrupt_prompt, add_special_tokens=False))
        clean_sentence_len = len(tokenizer.encode(pair.clean_sentence, add_special_tokens=False))
        corrupt_sentence_len = len(tokenizer.encode(pair.corrupt_sentence, add_special_tokens=False))
        if clean_prompt_len != corrupt_prompt_len:
            skips["prompt_token_count_mismatch"] += 1
            continue
        if clean_sentence_len != corrupt_sentence_len:
            skips["full_sentence_token_count_mismatch"] += 1
            continue
        if clean_prompt_len <= 0 or clean_prompt_len > clean_sentence_len:
            skips["bad_anchor_range"] += 1
            continue
        checked.append(
            WholePairDirected(
                **{
                    **pair.__dict__,
                    "prompt_token_count": clean_prompt_len,
                    "anchor_token_index": clean_prompt_len - 1,
                }
            )
        )
    return checked, dict(skips)


def orthonormal_basis(raw: Any) -> Any:
    import torch

    q, _r = torch.linalg.qr(raw.float(), mode="reduced")
    return q


def capture_site(model: Any, inputs: Any, site_index: int, anchor: int) -> tuple[Any, Any]:
    captured: dict[str, Any] = {}

    def capture_tensor(hidden: Any) -> None:
        captured["site"] = hidden[:, anchor, :].detach()

    if site_index == 0:
        handle = model.gpt_neox.embed_in.register_forward_hook(lambda _module, _inputs, output: capture_tensor(output))
    else:
        layer_idx = site_index - 1

        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            capture_tensor(normalize_model_output(output))

        handle = model.gpt_neox.layers[layer_idx].register_forward_hook(hook)
    try:
        outputs = model(**inputs, output_hidden_states=False, use_cache=False)
    finally:
        handle.remove()
    return outputs, captured["site"]


def patched_forward(
    model: Any,
    corrupt_inputs: Any,
    site_index: int,
    anchor: int,
    clean_site: Any,
    basis: Any,
) -> Any:
    import torch

    row_indices = torch.arange(clean_site.shape[0], device=clean_site.device)

    def patch_tensor(hidden: Any) -> Any:
        hidden_f = hidden.float()
        clean_f = clean_site.float()
        delta = clean_f[row_indices, :] - hidden_f[row_indices, anchor, :]
        projected = (delta @ basis) @ basis.T
        patched = hidden_f.clone()
        patched[row_indices, anchor, :] = hidden_f[row_indices, anchor, :] + projected
        return patched.to(dtype=hidden.dtype)

    if site_index == 0:
        handle = model.gpt_neox.embed_in.register_forward_hook(lambda _module, _inputs, output: patch_tensor(output))
    else:
        layer_idx = site_index - 1

        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            hidden = normalize_model_output(output)
            return replace_model_output(output, patch_tensor(hidden))

        handle = model.gpt_neox.layers[layer_idx].register_forward_hook(hook)
    try:
        return model(**corrupt_inputs, output_hidden_states=False, use_cache=False)
    finally:
        handle.remove()


def encode_batch(tokenizer: Any, texts: list[str], device: Any) -> Any:
    return tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)


def sequence_lp_parts(logits: Any, input_ids: Any, attention_mask: Any, anchor: int) -> tuple[Any, Any, Any]:
    import torch
    import torch.nn.functional as F

    logprobs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    labels = input_ids[:, 1:]
    token_logprobs = logprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    mask = attention_mask[:, 1:].float()
    label_positions = torch.arange(1, input_ids.shape[1], device=input_ids.device)
    suffix_mask = (label_positions > anchor).float().unsqueeze(0) * mask
    prefix_mask = mask - suffix_mask
    full_lp = (token_logprobs * mask).sum(dim=-1)
    prefix_lp = (token_logprobs * prefix_mask).sum(dim=-1)
    suffix_lp = (token_logprobs * suffix_mask).sum(dim=-1)
    return full_lp, prefix_lp, suffix_lp


def batch_groups(pairs: list[WholePairDirected]) -> dict[tuple[str, str, str, int, int], list[WholePairDirected]]:
    grouped: dict[tuple[str, str, str, int, int], list[WholePairDirected]] = defaultdict(list)
    for pair in pairs:
        grouped[(pair.regime, pair.subtask, pair.direction, pair.prompt_token_count, pair.anchor_token_index)].append(pair)
    return grouped


def train_epoch(
    model: Any,
    tokenizer: Any,
    train_groups: dict[tuple[str, str, str, int, int], list[WholePairDirected]],
    train_keys: list[tuple[str, str, str, int, int]],
    device: Any,
    site_index: int,
    raw: Any,
    opt: Any,
    batch_size: int,
    grad_clip: float,
) -> tuple[float, int]:
    import torch
    import torch.nn.functional as F

    random.shuffle(train_keys)
    total_loss = 0.0
    total_n = 0
    for key in train_keys:
        group = train_groups[key][:]
        random.shuffle(group)
        for start in range(0, len(group), batch_size):
            batch = group[start : start + batch_size]
            opt.zero_grad(set_to_none=True)
            clean_inputs = encode_batch(tokenizer, [p.clean_sentence for p in batch], device)
            corrupt_inputs = encode_batch(tokenizer, [p.corrupt_sentence for p in batch], device)
            anchor = batch[0].anchor_token_index
            with torch.no_grad():
                _clean_outputs, clean_site = capture_site(model, clean_inputs, site_index, anchor)
                corrupt_outputs = model(**corrupt_inputs, output_hidden_states=False, use_cache=False)
                _full, _prefix, corrupt_suffix = sequence_lp_parts(
                    corrupt_outputs.logits,
                    corrupt_inputs.input_ids,
                    corrupt_inputs.attention_mask,
                    anchor,
                )
            basis = orthonormal_basis(raw)
            patched_outputs = patched_forward(model, corrupt_inputs, site_index, anchor, clean_site, basis)
            _p_full, _p_prefix, patched_suffix = sequence_lp_parts(
                patched_outputs.logits,
                corrupt_inputs.input_ids,
                corrupt_inputs.attention_mask,
                anchor,
            )
            signs = torch.tensor([side_sign(p.objective_clean_side) for p in batch], device=device)
            signed_effect = signs * (patched_suffix - corrupt_suffix.detach())
            loss = F.softplus(-signed_effect).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([raw], grad_clip)
            opt.step()
            total_loss += float(loss.detach().cpu()) * len(batch)
            total_n += len(batch)
    return total_loss / max(total_n, 1), total_n


def evaluate(
    model: Any,
    tokenizer: Any,
    pairs: list[WholePairDirected],
    device: Any,
    site_index: int,
    raw: Any,
    batch_size: int,
    split_name: str,
) -> list[dict[str, Any]]:
    import torch

    rows: list[dict[str, Any]] = []
    basis = orthonormal_basis(raw)
    with torch.no_grad():
        for _key, group in sorted(batch_groups(pairs).items()):
            for start in range(0, len(group), batch_size):
                batch = group[start : start + batch_size]
                clean_inputs = encode_batch(tokenizer, [p.clean_sentence for p in batch], device)
                corrupt_inputs = encode_batch(tokenizer, [p.corrupt_sentence for p in batch], device)
                anchor = batch[0].anchor_token_index
                clean_outputs, clean_site = capture_site(model, clean_inputs, site_index, anchor)
                corrupt_outputs = model(**corrupt_inputs, output_hidden_states=False, use_cache=False)
                patched_outputs = patched_forward(model, corrupt_inputs, site_index, anchor, clean_site, basis)
                clean_full, _clean_prefix, clean_suffix = sequence_lp_parts(
                    clean_outputs.logits,
                    clean_inputs.input_ids,
                    clean_inputs.attention_mask,
                    anchor,
                )
                corrupt_full, corrupt_prefix, corrupt_suffix = sequence_lp_parts(
                    corrupt_outputs.logits,
                    corrupt_inputs.input_ids,
                    corrupt_inputs.attention_mask,
                    anchor,
                )
                patched_full, patched_prefix, patched_suffix = sequence_lp_parts(
                    patched_outputs.logits,
                    corrupt_inputs.input_ids,
                    corrupt_inputs.attention_mask,
                    anchor,
                )
                for idx, pair in enumerate(batch):
                    clean_suf = float(clean_suffix[idx].detach().cpu())
                    corrupt_suf = float(corrupt_suffix[idx].detach().cpu())
                    patched_suf = float(patched_suffix[idx].detach().cpu())
                    clean_f = float(clean_full[idx].detach().cpu())
                    corrupt_f = float(corrupt_full[idx].detach().cpu())
                    patched_f = float(patched_full[idx].detach().cpu())
                    sign = side_sign(pair.objective_clean_side)
                    effect = patched_suf - corrupt_suf
                    full_effect = patched_f - corrupt_f
                    clean_gap = clean_suf - corrupt_suf
                    signed_effect = sign * effect
                    signed_clean_gap = sign * clean_gap
                    rows.append(
                        {
                            "split": split_name,
                            "pair_key": pair.pair_key,
                            "regime": pair.regime,
                            "subtask": pair.subtask,
                            "template_id": pair.template_id,
                            "context_id": pair.context_id,
                            "subject": pair.subject,
                            "subject_class": pair.subject_class,
                            "direction": pair.direction,
                            "clean_side": pair.clean_side,
                            "corrupt_side": pair.corrupt_side,
                            "objective_clean_side": pair.objective_clean_side,
                            "objective_corrupt_side": pair.objective_corrupt_side,
                            "site": site_name(site_index),
                            "anchor_token_index": pair.anchor_token_index,
                            "clean_verb": pair.clean_verb,
                            "corrupt_verb": pair.corrupt_verb,
                            "clean_source_zipf_regime": pair.clean_source_zipf_regime,
                            "corrupt_source_zipf_regime": pair.corrupt_source_zipf_regime,
                            "clean_inventory_source": pair.clean_inventory_source,
                            "corrupt_inventory_source": pair.corrupt_inventory_source,
                            "clean_suffix_lp": clean_suf,
                            "corrupt_suffix_lp": corrupt_suf,
                            "patched_suffix_lp": patched_suf,
                            "suffix_effect": effect,
                            "signed_suffix_effect": signed_effect,
                            "clean_minus_corrupt_suffix_lp": clean_gap,
                            "signed_clean_minus_corrupt_suffix_lp": signed_clean_gap,
                            "normalized_suffix_effect": ""
                            if abs(signed_clean_gap) < 1e-8
                            else signed_effect / signed_clean_gap,
                            "clean_full_lp": clean_f,
                            "corrupt_full_lp": corrupt_f,
                            "patched_full_lp": patched_f,
                            "full_effect": full_effect,
                            "signed_full_effect": sign * full_effect,
                            "corrupt_prefix_lp": float(corrupt_prefix[idx].detach().cpu()),
                            "patched_prefix_lp": float(patched_prefix[idx].detach().cpu()),
                            "moves_toward_objective_clean": int(signed_effect > 0),
                        }
                    )
    return rows


def summarize_injection(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["split"]), str(row["regime"]), str(row["subtask"]), str(row["direction"]))].append(row)
    out: list[dict[str, Any]] = []
    for (split, regime, subtask, direction), group in sorted(grouped.items()):
        signed = [float(r["signed_suffix_effect"]) for r in group]
        raw = [float(r["suffix_effect"]) for r in group]
        norm = [float(r["normalized_suffix_effect"]) for r in group if r["normalized_suffix_effect"] != ""]
        out.append(
            {
                "split": split,
                "regime": regime,
                "subtask": subtask,
                "direction": direction,
                "n": len(group),
                "mean_suffix_effect": mean(raw),
                "mean_signed_suffix_effect": mean(signed),
                "sem_signed_suffix_effect": sem(signed),
                "mean_normalized_suffix_effect": mean(norm),
                "move_success_rate": mean([float(r["moves_toward_objective_clean"]) for r in group]),
                "mean_clean_minus_corrupt_suffix_lp": mean([float(r["clean_minus_corrupt_suffix_lp"]) for r in group]),
                "mean_signed_clean_minus_corrupt_suffix_lp": mean(
                    [float(r["signed_clean_minus_corrupt_suffix_lp"]) for r in group]
                ),
                "mean_signed_full_effect": mean([float(r["signed_full_effect"]) for r in group]),
            }
        )
    return out


def unique_projection_records(pairs_by_split: dict[str, list[WholePairDirected]]) -> list[dict[str, Any]]:
    records: dict[tuple[Any, ...], dict[str, Any]] = {}
    for split, pairs in pairs_by_split.items():
        for pair in pairs:
            for role in ("clean", "corrupt"):
                side = getattr(pair, f"{role}_side")
                sentence = getattr(pair, f"{role}_sentence")
                prompt = getattr(pair, f"{role}_prompt")
                verb = getattr(pair, f"{role}_verb")
                key = (split, pair.regime, pair.subtask, pair.template_id, pair.context_id, side, verb, sentence)
                records.setdefault(
                    key,
                    {
                        "split": split,
                        "regime": pair.regime,
                        "subtask": pair.subtask,
                        "template_id": pair.template_id,
                        "context_id": pair.context_id,
                        "subject": pair.subject,
                        "subject_class": pair.subject_class,
                        "side": side,
                        "licensing_status": "licensed" if side == "good" else "unlicensed",
                        "verb": verb,
                        "sentence": sentence,
                        "prompt": prompt,
                        "prompt_token_count": pair.prompt_token_count,
                        "anchor_token_index": pair.anchor_token_index,
                    },
                )
    return list(records.values())


def project_records(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    basis: Any,
    site_index: int,
    batch_size: int,
    device: Any,
) -> list[dict[str, Any]]:
    import torch

    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(int(record["prompt_token_count"]), int(record["anchor_token_index"]))].append(record)
    out: list[dict[str, Any]] = []
    with torch.no_grad():
        for (_prompt_len, anchor), group in sorted(grouped.items()):
            for start in range(0, len(group), batch_size):
                batch = group[start : start + batch_size]
                inputs = encode_batch(tokenizer, [str(r["sentence"]) for r in batch], device)
                _outputs, hidden = capture_site(model, inputs, site_index, anchor)
                coords = hidden.float() @ basis.float()
                projected = coords @ basis.float().T
                projection_norm = projected.norm(dim=-1)
                residual_norm = hidden.float().norm(dim=-1)
                for row, coord, proj_n, resid_n in zip(
                    batch,
                    coords.detach().float().cpu().tolist(),
                    projection_norm.detach().float().cpu().tolist(),
                    residual_norm.detach().float().cpu().tolist(),
                ):
                    item = dict(row)
                    if not isinstance(coord, list):
                        coord = [float(coord)]
                    for idx, value in enumerate(coord):
                        item[f"coord_{idx}"] = value
                    item["projection_norm"] = proj_n
                    item["residual_norm"] = resid_n
                    item["projection_norm_frac"] = "" if resid_n == 0 else proj_n / resid_n
                    out.append(item)
    return out


def summarize_projection(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["split"]), str(row["regime"]), str(row["subtask"]))].append(row)
    out: list[dict[str, Any]] = []
    for (split, regime, subtask), group in sorted(grouped.items()):
        values = [float(r["coord_0"]) for r in group]
        labels = [str(r["side"]) for r in group]
        good = [v for v, label in zip(values, labels) if label == "good"]
        bad = [v for v, label in zip(values, labels) if label == "bad"]
        if not good or not bad:
            continue
        auc = binary_auc(values, labels, "good")
        best_acc, threshold, direction = best_threshold_accuracy(values, labels, "good")
        out.append(
            {
                "split": split,
                "regime": regime,
                "subtask": subtask,
                "label_column": "side",
                "n_good": len(good),
                "n_bad": len(bad),
                "mean_coord_good": mean(good),
                "mean_coord_bad": mean(bad),
                "mean_good_minus_bad": mean(good) - mean(bad),
                "auc_good_gt_bad": auc,
                "orientation_free_auc": max(auc, 1.0 - auc),
                "best_threshold_acc": best_acc,
                "best_threshold": threshold,
                "best_threshold_direction_for_good": direction,
            }
        )
    return out


def coord_values(rows: list[dict[str, Any]], **filters: str) -> list[float]:
    return [
        float(row["coord_0"])
        for row in rows
        if all(str(row.get(key, "")) == value for key, value in filters.items())
    ]


def summarize_validation_cells(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["split"]), str(row["regime"]))].append(row)
    out: list[dict[str, Any]] = []
    for (split, regime), group in sorted(grouped.items()):
        caus_good = coord_values(group, subtask="causative", side="good")
        caus_bad = coord_values(group, subtask="causative", side="bad")
        inch_good = coord_values(group, subtask="inchoative", side="good")
        inch_bad = coord_values(group, subtask="inchoative", side="bad")
        if not all([caus_good, caus_bad, inch_good, inch_bad]):
            continue
        caus_margin = mean(caus_good) - mean(caus_bad)
        orientation = 1.0 if caus_margin >= 0 else -1.0
        inch_margin = mean(inch_good) - mean(inch_bad)
        calibrated_inch = orientation * inch_margin
        out.append(
            {
                "split": split,
                "regime": regime,
                "cell": "causative_calibrated_inchoative_good_vs_bad",
                "n_causative_good": len(caus_good),
                "n_causative_bad": len(caus_bad),
                "n_inchoative_good": len(inch_good),
                "n_inchoative_bad": len(inch_bad),
                "causative_good_minus_bad": caus_margin,
                "inchoative_good_minus_bad": inch_margin,
                "calibrated_inchoative_good_minus_bad": calibrated_inch,
                "licensing_axis_pass": int(calibrated_inch > 0),
            }
        )
    return out


def run(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    subtasks = set(parse_csv_arg(args.subtasks))
    train_regimes = set(parse_csv_arg(args.train_regimes))
    eval_regimes = set(parse_csv_arg(args.eval_regimes))
    regimes = train_regimes | eval_regimes
    directions = set(parse_csv_arg(args.directions))
    if not directions <= {"good_to_bad", "bad_to_good"}:
        raise SystemExit("--directions must contain only good_to_bad,bad_to_good")
    site_index = parse_site(args.site)

    rows = load_rows(Path(args.data), subtasks, regimes)
    whole_pairs, pair_skip_counts = build_pairs(rows, args.max_pairs_per_group, args.seed)
    pairs = make_directed_pairs(whole_pairs, directions, args.control, args.seed)
    if not pairs:
        raise SystemExit("No valid whole-pair directed examples after filtering.")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=not args.allow_download)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pairs, retokenize_skip_counts = retokenize_pairs(tokenizer, pairs)
    if not pairs:
        raise SystemExit("No valid whole-pair directed examples after retokenization.")
    train_pairs, eval_pairs = make_transfer_splits(
        pairs,
        train_regimes,
        eval_regimes,
        args.eval_frac,
        args.split_by,
        args.seed,
    )

    dtype = getattr(torch, args.dtype) if args.dtype != "auto" else "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, local_files_only=not args.allow_download)
    device = torch.device(args.device)
    model.to(device)
    model.eval()
    model.config.use_cache = False
    for param in model.parameters():
        param.requires_grad_(False)

    d_model = model.config.hidden_size
    if args.load_subspace:
        checkpoint = torch.load(args.load_subspace, map_location="cpu")
        raw_checkpoint = checkpoint["raw"] if isinstance(checkpoint, dict) and "raw" in checkpoint else checkpoint
        if raw_checkpoint.ndim == 1:
            raw_checkpoint = raw_checkpoint[:, None]
        if raw_checkpoint.shape[0] != d_model:
            raise SystemExit(f"Loaded subspace d_model mismatch: checkpoint={raw_checkpoint.shape[0]} model={d_model}")
        if raw_checkpoint.shape[1] != args.rank:
            raise SystemExit(f"Loaded subspace rank mismatch: checkpoint={raw_checkpoint.shape[1]} --rank={args.rank}")
        if isinstance(checkpoint, dict) and checkpoint.get("site") not in {None, args.site}:
            raise SystemExit(f"Loaded subspace site mismatch: checkpoint={checkpoint.get('site')} --site={args.site}")
        raw = raw_checkpoint.float().to(device)
    else:
        raw = torch.randn(d_model, args.rank, device=device, dtype=torch.float32) / math.sqrt(d_model)
    raw.requires_grad_(not args.load_subspace and args.control != "random_direction")
    opt = torch.optim.AdamW([raw], lr=args.lr, weight_decay=args.weight_decay) if raw.requires_grad else None

    train_groups = batch_groups(train_pairs)
    train_keys = sorted(train_groups)
    effective_epochs = 0 if args.load_subspace or args.control == "random_direction" else args.epochs
    history: list[dict[str, Any]] = []
    for epoch in range(effective_epochs):
        assert opt is not None
        train_loss, train_n = train_epoch(
            model,
            tokenizer,
            train_groups,
            train_keys,
            device,
            site_index,
            raw,
            opt,
            args.batch_size,
            args.grad_clip,
        )
        row = {"epoch": epoch + 1, "train_loss": train_loss, "train_n": train_n}
        history.append(row)
        print(json.dumps(row), flush=True)

    train_detail = evaluate(model, tokenizer, train_pairs, device, site_index, raw, args.batch_size, "train")
    eval_detail = evaluate(model, tokenizer, eval_pairs, device, site_index, raw, args.batch_size, "eval")
    injection_detail = train_detail + eval_detail
    injection_summary = summarize_injection(injection_detail)

    basis = orthonormal_basis(raw)
    projection_records = unique_projection_records({"train": train_pairs, "eval": eval_pairs})
    projection_detail = project_records(model, tokenizer, projection_records, basis, site_index, args.batch_size, device)
    projection_summary = summarize_projection(projection_detail)
    validation_summary = summarize_validation_cells(projection_detail)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_slug = args.run_name or shell_safe_slug(f"{args.model}_{args.site}_whole_pair_das")
    injection_detail_csv = out_dir / f"{run_slug}.injection_detail.csv"
    injection_summary_csv = out_dir / f"{run_slug}.injection_summary.csv"
    projection_detail_csv = out_dir / f"{run_slug}.projection_detail.csv"
    projection_summary_csv = out_dir / f"{run_slug}.projection_summary.csv"
    validation_summary_csv = out_dir / f"{run_slug}.validation_cell_summary.csv"
    history_csv = out_dir / f"{run_slug}.history.csv"
    subspace_path = out_dir / f"{run_slug}.subspace.pt"
    manifest_json = out_dir / f"{run_slug}.manifest.json"

    write_csv(injection_detail_csv, injection_detail)
    write_csv(injection_summary_csv, injection_summary)
    write_csv(projection_detail_csv, projection_detail)
    write_csv(projection_summary_csv, projection_summary)
    write_csv(validation_summary_csv, validation_summary)
    write_csv(history_csv, history)
    torch.save({"raw": raw.detach().cpu(), "site": args.site, "rank": args.rank}, subspace_path)

    manifest = {
        "model": args.model,
        "data": str(Path(args.data)),
        "site": args.site,
        "rank": args.rank,
        "control": args.control,
        "load_subspace": args.load_subspace,
        "subtasks": sorted(subtasks),
        "train_regimes": sorted(train_regimes),
        "eval_regimes": sorted(eval_regimes),
        "directions": sorted(directions),
        "rows_loaded": len(rows),
        "whole_pairs": len(whole_pairs),
        "directed_pairs_after_retokenize": len(pairs),
        "train_pairs": len(train_pairs),
        "eval_pairs": len(eval_pairs),
        "pair_skip_counts": pair_skip_counts,
        "retokenize_skip_counts": retokenize_skip_counts,
        "split_by": args.split_by,
        "eval_frac": args.eval_frac,
        "epochs": effective_epochs,
        "lr": args.lr,
        "injection_detail_csv": str(injection_detail_csv),
        "injection_summary_csv": str(injection_summary_csv),
        "projection_detail_csv": str(projection_detail_csv),
        "projection_summary_csv": str(projection_summary_csv),
        "validation_summary_csv": str(validation_summary_csv),
        "history_csv": str(history_csv),
        "subspace": str(subspace_path),
        "note": (
            "Whole-pair DAS at the verb-final residual stream. The trained injection objective is post-anchor "
            "suffix log probability because this intervention cannot change the probability assigned to the "
            "already-observed verb token. Projection summaries read the learned subspace from full-sentence "
            "verb-anchor activations without intervention."
        ),
    }
    manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--site", default="resid_post_layer_23")
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--subtasks", default=",".join(DEFAULT_SUBTASKS))
    ap.add_argument("--train-regimes", default="head")
    ap.add_argument("--eval-regimes", default="head,low")
    ap.add_argument("--directions", default="good_to_bad,bad_to_good")
    ap.add_argument("--max-pairs-per-group", type=int, default=None)
    ap.add_argument("--eval-frac", type=float, default=0.25)
    ap.add_argument("--split-by", choices=("pair", "lemma_pair"), default="lemma_pair")
    ap.add_argument("--control", choices=("none", "random_direction", "shuffled_label"), default="none")
    ap.add_argument("--load-subspace", default="")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--allow-download", action="store_true")
    ap.add_argument("--out-dir", default="results/whole_pair_das")
    ap.add_argument("--run-name", default="")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
