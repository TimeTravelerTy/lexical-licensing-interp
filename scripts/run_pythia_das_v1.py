#!/usr/bin/env python3
"""Train a small v1 DAS-style subspace on aligned lexical templates.

This is infrastructure DAS for the fixed-subject v1 scaffold. It learns a
low-rank residual-stream subspace at the prompt-final verb token, then performs
interchange interventions by moving only the clean-corrupt activation
difference projected onto that subspace.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from run_pythia_attribution_patching import (
    DEFAULT_MODEL,
    DEFAULT_REGIMES,
    DirectedPair,
    build_directed_pairs,
    load_aligned_rows,
    parse_csv_arg,
    shell_safe_slug,
    token_id,
)
from run_pythia_exact_patching import clean_forward_with_site, normalize_model_output, parse_site, replace_model_output, site_name


DEFAULT_SUBTASKS = ("causative", "inchoative")


def stable_unit(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def split_key(pair: DirectedPair, split_by: str) -> str:
    if split_by == "pair":
        return pair.pair_key
    if split_by == "lemma_pair":
        return "|".join(sorted((pair.clean_verb, pair.corrupt_verb)))
    raise ValueError(f"unknown split policy: {split_by}")


def apply_control(pair: DirectedPair, control: str, dummy_verb: str, seed: int) -> DirectedPair:
    if control in {"none", "random_direction"}:
        return pair
    if control == "shuffled_label":
        should_swap = stable_unit(pair.pair_key + "|" + pair.direction, seed) < 0.5
        if not should_swap:
            return pair
        return DirectedPair(
            **{
                **pair.__dict__,
                "clean_target": pair.corrupt_target,
                "corrupt_target": pair.clean_target,
            }
        )
    if control == "dummy_verb":
        clean_prompt = pair.clean_prompt.rsplit(" ", 1)[0] + f" {dummy_verb}"
        corrupt_prompt = pair.corrupt_prompt.rsplit(" ", 1)[0] + f" {dummy_verb}"
        return DirectedPair(
            **{
                **pair.__dict__,
                "clean_prompt": clean_prompt,
                "corrupt_prompt": corrupt_prompt,
                "clean_verb": dummy_verb,
                "corrupt_verb": dummy_verb,
            }
        )
    raise ValueError(f"unknown control: {control}")


def make_splits(pairs: list[DirectedPair], eval_frac: float, split_by: str, seed: int) -> tuple[list[DirectedPair], list[DirectedPair]]:
    train: list[DirectedPair] = []
    eval_: list[DirectedPair] = []
    for pair in pairs:
        if stable_unit(split_key(pair, split_by), seed) < eval_frac:
            eval_.append(pair)
        else:
            train.append(pair)
    if not train or not eval_:
        raise SystemExit(f"Empty split with eval_frac={eval_frac}, split_by={split_by}: train={len(train)} eval={len(eval_)}")
    return train, eval_


def make_transfer_splits(
    pairs: list[DirectedPair],
    train_regimes: set[str],
    eval_regimes: set[str],
    eval_frac: float,
    split_by: str,
    seed: int,
) -> tuple[list[DirectedPair], list[DirectedPair]]:
    train: list[DirectedPair] = []
    eval_: list[DirectedPair] = []
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
        raise SystemExit(
            "Empty transfer split: "
            f"train={len(train)} eval={len(eval_)} "
            f"train_regimes={sorted(train_regimes)} eval_regimes={sorted(eval_regimes)}"
        )
    return train, eval_


def retokenize_pairs(tokenizer: Any, pairs: list[DirectedPair]) -> list[DirectedPair]:
    checked: list[DirectedPair] = []
    for pair in pairs:
        clean_len = len(tokenizer.encode(pair.clean_prompt, add_special_tokens=False))
        corrupt_len = len(tokenizer.encode(pair.corrupt_prompt, add_special_tokens=False))
        if clean_len != corrupt_len:
            continue
        checked.append(
            DirectedPair(
                **{
                    **pair.__dict__,
                    "prompt_token_count": clean_len,
                    "anchor_token_index": clean_len - 1,
                }
            )
        )
    return checked


def batch_groups(pairs: list[DirectedPair]) -> dict[tuple[str, str, str, int, int], list[DirectedPair]]:
    grouped: dict[tuple[str, str, str, int, int], list[DirectedPair]] = defaultdict(list)
    for pair in pairs:
        grouped[(pair.regime, pair.subtask, pair.direction, pair.prompt_token_count, pair.anchor_token_index)].append(pair)
    return grouped


def metric_from_logits(logits: Any, clean_target: Any, corrupt_target: Any) -> Any:
    import torch
    import torch.nn.functional as F

    row_indices = torch.arange(logits.shape[0], device=logits.device)
    logprobs = F.log_softmax(logits[:, -1, :].float(), dim=-1)
    return logprobs[row_indices, clean_target] - logprobs[row_indices, corrupt_target]


def orthonormal_basis(raw: Any) -> Any:
    import torch

    q, _r = torch.linalg.qr(raw.float(), mode="reduced")
    return q


def patched_forward(model: Any, corrupt_inputs: Any, site_index: int, anchor: int, clean_site: Any, basis: Any) -> Any:
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
        handle = model.gpt_neox.embed_in.register_forward_hook(
            lambda _module, _inputs, output: patch_tensor(output)
        )
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


def encode_batch(tokenizer: Any, prompts: list[str], device: Any) -> Any:
    return tokenizer(
        prompts,
        return_tensors="pt",
        padding=False,
        add_special_tokens=False,
    ).to(device)


def run_batch(
    model: Any,
    tokenizer: Any,
    batch: list[DirectedPair],
    target_ids: dict[str, int],
    device: Any,
    site_index: int,
    basis: Any,
    need_clean_metric: bool,
) -> tuple[Any, Any, Any]:
    import torch

    clean_inputs = encode_batch(tokenizer, [p.clean_prompt for p in batch], device)
    corrupt_inputs = encode_batch(tokenizer, [p.corrupt_prompt for p in batch], device)
    expected_len = batch[0].prompt_token_count
    if clean_inputs.input_ids.shape[1] != expected_len or corrupt_inputs.input_ids.shape[1] != expected_len:
        raise RuntimeError("Tokenizer length mismatch; check control prompt tokenization.")

    clean_target = torch.tensor([target_ids[p.clean_target] for p in batch], device=device)
    corrupt_target = torch.tensor([target_ids[p.corrupt_target] for p in batch], device=device)
    anchor = batch[0].anchor_token_index

    with torch.no_grad():
        clean_outputs, clean_site = clean_forward_with_site(model, clean_inputs, site_index, anchor)
        clean_metric = metric_from_logits(clean_outputs.logits, clean_target, corrupt_target) if need_clean_metric else None

    patched_outputs = patched_forward(model, corrupt_inputs, site_index, anchor, clean_site, basis)
    patched_metric = metric_from_logits(patched_outputs.logits, clean_target, corrupt_target)
    return patched_metric, clean_metric, (clean_target, corrupt_target)


def evaluate(
    model: Any,
    tokenizer: Any,
    pairs: list[DirectedPair],
    target_ids: dict[str, int],
    device: Any,
    site_index: int,
    raw: Any,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    detail_rows: list[dict[str, Any]] = []
    basis = orthonormal_basis(raw)
    with torch.no_grad():
        for group_key, group_pairs in sorted(batch_groups(pairs).items()):
            for start in range(0, len(group_pairs), batch_size):
                batch = group_pairs[start : start + batch_size]
                clean_inputs = encode_batch(tokenizer, [p.clean_prompt for p in batch], device)
                corrupt_inputs = encode_batch(tokenizer, [p.corrupt_prompt for p in batch], device)
                expected_len = batch[0].prompt_token_count
                if clean_inputs.input_ids.shape[1] != expected_len or corrupt_inputs.input_ids.shape[1] != expected_len:
                    raise RuntimeError(f"Tokenizer length mismatch in eval group {group_key}")
                clean_target = torch.tensor([target_ids[p.clean_target] for p in batch], device=device)
                corrupt_target = torch.tensor([target_ids[p.corrupt_target] for p in batch], device=device)
                anchor = batch[0].anchor_token_index

                clean_outputs, clean_site = clean_forward_with_site(model, clean_inputs, site_index, anchor)
                corrupt_outputs = model(**corrupt_inputs, output_hidden_states=False, use_cache=False)
                clean_metric = metric_from_logits(clean_outputs.logits, clean_target, corrupt_target)
                corrupt_metric = metric_from_logits(corrupt_outputs.logits, clean_target, corrupt_target)
                patched_outputs = patched_forward(model, corrupt_inputs, site_index, anchor, clean_site, basis)
                patched_metric = metric_from_logits(patched_outputs.logits, clean_target, corrupt_target)
                effect = patched_metric - corrupt_metric
                denom = clean_metric - corrupt_metric

                for pair, c_m, p_m, cl_m, eff, gap in zip(
                    batch,
                    corrupt_metric.float().cpu().tolist(),
                    patched_metric.float().cpu().tolist(),
                    clean_metric.float().cpu().tolist(),
                    effect.float().cpu().tolist(),
                    denom.float().cpu().tolist(),
                ):
                    detail_rows.append(
                        {
                            "pair_key": pair.pair_key,
                            "regime": pair.regime,
                            "subtask": pair.subtask,
                            "direction": pair.direction,
                            "site": site_name(site_index),
                            "clean_verb": pair.clean_verb,
                            "corrupt_verb": pair.corrupt_verb,
                            "corrupt_metric": c_m,
                            "patched_metric": p_m,
                            "clean_metric": cl_m,
                            "effect": eff,
                            "normalized_effect": "" if abs(gap) < 1e-8 else eff / gap,
                            "patched_success": int(p_m > 0),
                        }
                    )
    if not detail_rows:
        return detail_rows, {}
    metrics = {
        "n": len(detail_rows),
        "mean_corrupt_metric": sum(float(r["corrupt_metric"]) for r in detail_rows) / len(detail_rows),
        "mean_patched_metric": sum(float(r["patched_metric"]) for r in detail_rows) / len(detail_rows),
        "mean_clean_metric": sum(float(r["clean_metric"]) for r in detail_rows) / len(detail_rows),
        "mean_effect": sum(float(r["effect"]) for r in detail_rows) / len(detail_rows),
        "patched_success_rate": sum(int(r["patched_success"]) for r in detail_rows) / len(detail_rows),
    }
    norm = [float(r["normalized_effect"]) for r in detail_rows if r["normalized_effect"] != ""]
    metrics["mean_normalized_effect"] = sum(norm) / len(norm) if norm else math.nan
    return detail_rows, metrics


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    site_index = parse_site(args.site)
    subtask_filter = set(parse_csv_arg(args.subtasks))
    train_regimes = set(parse_csv_arg(args.train_regimes or args.regimes))
    eval_regimes = set(parse_csv_arg(args.eval_regimes or args.regimes))
    regime_filter = train_regimes | eval_regimes
    directions = set(parse_csv_arg(args.directions))
    rows = load_aligned_rows(Path(args.data), subtask_filter, regime_filter)
    pairs, skip_counts = build_directed_pairs(rows, directions, args.max_pairs_per_group)
    pairs = [apply_control(pair, args.control, args.dummy_verb, args.seed) for pair in pairs]

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=not args.allow_download)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pairs = retokenize_pairs(tokenizer, pairs)
    if not pairs:
        raise SystemExit("No valid pairs after control tokenization.")
    train_pairs, eval_pairs = make_transfer_splits(
        pairs,
        train_regimes,
        eval_regimes,
        args.eval_frac,
        args.split_by,
        args.seed,
    )
    dtype = getattr(torch, args.dtype) if args.dtype != "auto" else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        local_files_only=not args.allow_download,
    )
    device = torch.device(args.device)
    model.to(device)
    model.eval()
    model.config.use_cache = False
    for param in model.parameters():
        param.requires_grad_(False)

    target_ids = {target: token_id(tokenizer, target) for p in pairs for target in (p.clean_target, p.corrupt_target)}
    d_model = model.config.hidden_size
    raw = torch.randn(d_model, args.rank, device=device, dtype=torch.float32) / math.sqrt(d_model)
    raw.requires_grad_(True)
    opt = torch.optim.AdamW([raw], lr=args.lr, weight_decay=args.weight_decay)

    train_groups = batch_groups(train_pairs)
    train_keys = sorted(train_groups)
    history: list[dict[str, Any]] = []
    effective_epochs = 0 if args.control == "random_direction" else args.epochs
    for epoch in range(effective_epochs):
        random.shuffle(train_keys)
        total_loss = 0.0
        total_n = 0
        for group_key in train_keys:
            group_pairs = train_groups[group_key][:]
            random.shuffle(group_pairs)
            for start in range(0, len(group_pairs), args.batch_size):
                batch = group_pairs[start : start + args.batch_size]
                opt.zero_grad(set_to_none=True)
                basis = orthonormal_basis(raw)
                patched_metric, _clean_metric, _targets = run_batch(
                    model,
                    tokenizer,
                    batch,
                    target_ids,
                    device,
                    site_index,
                    basis,
                    need_clean_metric=False,
                )
                loss = F.softplus(-patched_metric).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_([raw], args.grad_clip)
                opt.step()
                total_loss += float(loss.detach().cpu()) * len(batch)
                total_n += len(batch)
        row = {"epoch": epoch + 1, "train_loss": total_loss / max(total_n, 1)}
        history.append(row)
        print(json.dumps(row), flush=True)

    train_detail, train_metrics = evaluate(model, tokenizer, train_pairs, target_ids, device, site_index, raw, args.batch_size)
    eval_detail, eval_metrics = evaluate(model, tokenizer, eval_pairs, target_ids, device, site_index, raw, args.batch_size)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_slug = args.run_name or shell_safe_slug(f"{args.model}_{args.site}_{args.control}_{'-'.join(sorted(subtask_filter))}")
    write_csv(out_dir / f"{run_slug}.train_detail.csv", train_detail)
    write_csv(out_dir / f"{run_slug}.eval_detail.csv", eval_detail)
    write_csv(out_dir / f"{run_slug}.history.csv", history)
    torch.save({"raw": raw.detach().cpu(), "site": args.site, "rank": args.rank}, out_dir / f"{run_slug}.subspace.pt")
    manifest = {
        "model": args.model,
        "site": args.site,
        "rank": args.rank,
        "control": args.control,
        "subtasks": sorted(subtask_filter),
        "regimes": sorted(regime_filter),
        "train_regimes": sorted(train_regimes),
        "eval_regimes": sorted(eval_regimes),
        "directions": sorted(directions),
        "rows_loaded": len(rows),
        "directed_pairs": len(pairs),
        "train_pairs": len(train_pairs),
        "eval_pairs": len(eval_pairs),
        "skip_counts": skip_counts,
        "split_by": args.split_by,
        "eval_frac": args.eval_frac,
        "epochs": effective_epochs,
        "lr": args.lr,
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "note": (
            "V1 infrastructure DAS on fixed-subject scaffold; do not treat as final semantic causal evidence. "
            "shuffled_label retrains on permuted labels; dummy_verb tests template solvability without verb signal; "
            "random_direction evaluates an untrained random subspace of the same rank."
        ),
    }
    (out_dir / f"{run_slug}.manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/aligned_templates/lexical_licensing_aligned.jsonl")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--site", default="resid_post_layer_23")
    ap.add_argument("--subtasks", default=",".join(DEFAULT_SUBTASKS))
    ap.add_argument("--regimes", default=",".join(DEFAULT_REGIMES))
    ap.add_argument("--train-regimes", default="")
    ap.add_argument("--eval-regimes", default="")
    ap.add_argument("--directions", default="good_to_bad,bad_to_good")
    ap.add_argument("--max-pairs-per-group", type=int, default=None)
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-frac", type=float, default=0.25)
    ap.add_argument("--split-by", choices=("pair", "lemma_pair"), default="lemma_pair")
    ap.add_argument("--control", choices=("none", "shuffled_label", "dummy_verb", "random_direction"), default="none")
    ap.add_argument("--dummy-verb", default="do")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--allow-download", action="store_true")
    ap.add_argument("--out-dir", default="results/das_v1")
    ap.add_argument("--run-name", default="")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
