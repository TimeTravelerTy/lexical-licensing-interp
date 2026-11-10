#!/usr/bin/env python3
"""Gemma replication, nonce prep (tokenizers only, no model run): how the existing nonce stimuli tokenize under
Gemma-3-1B vs Pythia-1.4B.

Stimuli: data/nonce_passive/prompts.csv (round-3 C8) and data/round4/nonce_cues/prompts.csv (round-4 C8 / C9); both
use the 80 lemmas of data/nonce_passive/lemmas.csv. Forms of a lemma (the builders' own spelling rules): bare, -s
(`build_nonce_cues.s_form`), -ing (`build_nonce_cues.ing`), -ed (`build_nonce_frames.regular_past`; the passive /
perfect probe form, whose last subtoken is the measured position in `run_nonce_passive.py`).

Per lemma and form: the pieces of " <form>" (leading space, no special tokens) under each tokenizer, and the count.
Checks over the prompts (rules: reports/gemma/exploratory.md, section 8):
- probe frame: the -ed form's pieces in every probe of the lemma (probe tokens minus the tokens of its prefix,
  e.g. "The house was" / "She has"); a lemma is flagged if any frame gives pieces other than " <-ed form>"'s;
- context occurrences: every word occurrence of a lemma form in a nonce prompt, located with the fast tokenizer's
  offsets; an occurrence is flagged if a token straddles the word's start or end (e.g. merged with the following
  "." / "?"), or if its pieces differ from " <form>"'s;
- token suffix (the check `run_nonce_passive.py` applies, every prompt of both sets, all kinds): the tokens of
  " " + probe (probe itself when prompt == probe) are the last tokens of the prompt.
Flags per lemma ("splits very differently", defined before running):
- |n_Gemma - n_Pythia| >= 2 for any form;
- the -ed form's last piece is inflection-only ("ed" / "d", ignoring the word-boundary marker) in one tokenizer and
  not in the other (the measured state then sits on a suffix piece in one model and a stem-bearing piece in the other);
- any probe-frame or context-occurrence flag above.

Writes <out-dir>/{forms.csv, occurrences.csv, meta.json} and the report <report>.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_nonce_cues import ing, s_form  # noqa: E402
from build_nonce_frames import regular_past  # noqa: E402

MODELS = {"Gemma-3-1B": "google/gemma-3-1b-pt", "Pythia-1.4B": "EleutherAI/pythia-1.4b"}
SETS = {"round-3 C8 (data/nonce_passive)": "data/nonce_passive/prompts.csv",
        "round-4 C8 / C9 (data/round4/nonce_cues)": "data/round4/nonce_cues/prompts.csv"}
FORMS = ("bare", "s", "ing", "ed")


def forms(lemma):
    return {"bare": lemma, "s": s_form(lemma), "ing": ing(lemma), "ed": regular_past(lemma)}


def strip(p):
    return p.replace("Ġ", "").replace("▁", "")


def pieces(tok, text):
    return tok.convert_ids_to_tokens(tok.encode(text, add_special_tokens=False))


def main(a):
    from transformers import AutoTokenizer

    toks = {m: AutoTokenizer.from_pretrained(p) for m, p in MODELS.items()}
    lemmas = pd.read_csv(a.lemmas).lemma.tolist()
    sets = {k: pd.read_csv(v) for k, v in SETS.items()}
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # forms in isolation
    F = []
    for lem in lemmas:
        for f, w in forms(lem).items():
            r = {"lemma": lem, "form": f, "word": w}
            for m, t in toks.items():
                p = pieces(t, " " + w)
                r[f"{m}|n"], r[f"{m}|pieces"] = len(p), " ".join(p)
            F.append(r)
    F = pd.DataFrame(F)
    iso = {(r.lemma, r.form, m): r[f"{m}|pieces"] for _, r in F.iterrows() for m in toks}

    # probe frames (-ed form at the end of every nonce probe) and the token-suffix check
    frame_bad = {m: {} for m in toks}
    suffix_fail = {m: {k: 0 for k in sets} for m in toks}
    n_prompts = {k: len(P) for k, P in sets.items()}
    for k, P in sets.items():
        for m, t in toks.items():
            for r in P.itertuples():
                full = t.encode(r.prompt, add_special_tokens=False)
                probe = t.encode((" " if r.prompt != r.probe else "") + r.probe, add_special_tokens=False)
                if full[len(full) - len(probe):] != probe:
                    suffix_fail[m][k] += 1
                if r.kind != "nonce":
                    continue
                prefix, verb = r.probe.rsplit(" ", 1)
                assert verb == regular_past(r.lemma), (verb, r.lemma)
                pe = t.encode(prefix, add_special_tokens=False)
                pr = t.encode(r.probe, add_special_tokens=False)
                got = " ".join(t.convert_ids_to_tokens(pr[len(pe):])) if pr[:len(pe)] == pe else "<prefix not a token prefix>"
                if got != iso[(r.lemma, "ed", m)]:
                    frame_bad[m].setdefault(r.lemma, set()).add(f"{prefix}: {got}")

    # context occurrences of every lemma form in the nonce prompts
    O = []
    for k, P in sets.items():
        Pn = P[P.kind == "nonce"].drop_duplicates("prompt")
        for m, t in toks.items():
            for r in Pn.itertuples():
                enc = t(r.prompt, add_special_tokens=False, return_offsets_mapping=True)
                ids, offs = enc["input_ids"], enc["offset_mapping"]
                lems = {r.lemma}
                for col in ("partner", "context_lemma"):  # round-4 partner; round-3 context lemma(s), "A+B" when balanced
                    v = getattr(r, col, None)
                    if isinstance(v, str):
                        lems |= set(v.split("+"))
                words = {w: (lem, f) for lem in lems for f, w in forms(lem).items()}
                for w, (lem, f) in words.items():
                    for mt in re.finditer(rf"(?<![A-Za-z]){re.escape(w)}(?![A-Za-z])", r.prompt):
                        s, e = mt.start(), mt.end()
                        inside = [i for i, (a_, b_) in enumerate(offs) if a_ < e and b_ > s]
                        straddle = any(offs[i][1] > e for i in inside) or any(
                            offs[i][0] < s and r.prompt[offs[i][0]:s].strip() != "" for i in inside)
                        got = " ".join(t.convert_ids_to_tokens([ids[i] for i in inside]))
                        exp = iso[(lem, f, m)]
                        # sentence-initial / no preceding space: compare without the boundary marker on the first piece
                        same = got == exp or (s == 0 or r.prompt[s - 1] != " ") and strip(got) == strip(exp)
                        O.append({"set": k, "model": m, "lemma": lem, "form": f, "word": w, "straddle": straddle,
                                  "same_as_isolated": bool(same), "pieces": got, "isolated": exp})
    O = pd.DataFrame(O)
    occ_bad = O[O.straddle | ~O.same_as_isolated]

    # per-lemma flags
    G, Pm = "Gemma-3-1B", "Pythia-1.4B"
    L = []
    for lem in lemmas:
        f = F[F.lemma == lem].set_index("form")
        diff = {fm: int(f.loc[fm, f"{G}|n"] - f.loc[fm, f"{Pm}|n"]) for fm in FORMS}
        last = {m: strip(f.loc["ed", f"{m}|pieces"].split()[-1]) for m in toks}
        infl = {m: last[m] in ("ed", "d") for m in toks}
        flags = []
        if any(abs(v) >= 2 for v in diff.values()):
            flags.append("count diff >= 2 (" + ", ".join(f"{k} {v:+d}" for k, v in diff.items() if abs(v) >= 2) + ")")
        if infl[G] != infl[Pm]:
            flags.append(f"last -ed piece: Gemma '{last[G]}', Pythia '{last[Pm]}'")
        for m in toks:
            if lem in frame_bad[m]:
                flags.append(f"{m} probe-frame split")
            ob = occ_bad[(occ_bad.model == m) & (occ_bad.lemma == lem)]
            if len(ob):
                flags.append(f"{m} context split ({len(ob)} occurrences)")
        L.append({"lemma": lem, **{f"diff_{k}": v for k, v in diff.items()}, "last_ed_gemma": last[G],
                  "last_ed_pythia": last[Pm], "flag": "; ".join(flags)})
    L = pd.DataFrame(L)
    W = F.pivot(index="lemma", columns="form").sort_index()
    F.to_csv(out / "forms.csv", index=False)
    O.to_csv(out / "occurrences.csv", index=False)
    L.to_csv(out / "lemma_flags.csv", index=False)
    meta = {"models": MODELS, "tokenizer_classes": {m: type(t).__name__ for m, t in toks.items()},
            "n_lemmas": len(lemmas), "prompts": n_prompts, "suffix_check_failures": suffix_fail,
            "probe_frame_flagged_lemmas": {m: sorted(v) for m, v in frame_bad.items()},
            "context_occurrences": {m: int((O.model == m).sum()) for m in toks},
            "context_occurrences_flagged": {m: int((occ_bad.model == m).sum()) for m in toks},
            "flagged_lemmas": int((L["flag"] != "").sum())}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    # report
    R = ["# Nonce stimuli: tokenization under Gemma-3-1B vs Pythia-1.4B (EXPLORATORY prep; no model run)", "",
         "Generated by `scripts/report_nonce_tokenization.py` (definitions and flags in its docstring; rules in "
         "`exploratory.md`, section 8). Pieces of \" <form>\" without special tokens; Gemma's word-boundary marker is "
         "▁, Pythia's Ġ.", "",
         "## Summary", "",
         f"- Lemmas: {len(lemmas)}; prompts: " + ", ".join(f"{k} {v}" for k, v in n_prompts.items()) + ".",
         "- Token-suffix check (`run_nonce_passive.py`), failures: " + "; ".join(
             f"{m}: " + ", ".join(f"{v} / {n_prompts[k]}" for k, v in d.items()) for m, d in suffix_fail.items()) + ".",
         "- Probe-frame flags (-ed pieces differ between \"The N was\" / \"She has\" frames and isolation): " + "; ".join(
             f"{m}: {len(v)} lemmas" for m, v in frame_bad.items()) + ".",
         "- Context occurrences flagged (token straddles the word, or pieces differ from isolation): " + "; ".join(
             f"{m}: {meta['context_occurrences_flagged'][m]} / {meta['context_occurrences'][m]}" for m in toks) + ".",
         f"- Lemmas with any flag: {meta['flagged_lemmas']} / {len(lemmas)}.", "",
         "## Token counts per form", "",
         "| form | Gemma mean | Pythia mean | Gemma counts (n: lemmas) | Pythia counts (n: lemmas) | Gemma − Pythia: < 0 / = 0 / > 0 |",
         "|---|---:|---:|---|---|---|"]
    for fm in FORMS:
        x = F[F.form == fm]
        g, p = x[f"{G}|n"], x[f"{Pm}|n"]
        dist = lambda s: ", ".join(f"{k}: {v}" for k, v in sorted(s.value_counts().items()))
        R.append(f"| {fm} | {g.mean():.2f} | {p.mean():.2f} | {dist(g)} | {dist(p)} | "
                 f"{int((g < p).sum())} / {int((g == p).sum())} / {int((g > p).sum())} |")
    lg = L.last_ed_gemma.isin(["ed", "d"]).sum()
    lp = L.last_ed_pythia.isin(["ed", "d"]).sum()
    R += ["", f"Last piece of the -ed form inflection-only (\"ed\" / \"d\"): Gemma {lg} / {len(L)}, Pythia {lp} / {len(L)}.", "",
          "## Flagged lemmas", ""]
    fl = L[L["flag"] != ""]
    if fl.empty:
        R.append("None.")
    else:
        R += ["| lemma | -ed: Gemma | -ed: Pythia | flags |", "|---|---|---|---|"]
        for r in fl.itertuples():
            R.append(f"| {r.lemma} | `{W.loc[r.lemma, (f'{G}|pieces', 'ed')]}` | `{W.loc[r.lemma, (f'{Pm}|pieces', 'ed')]}` | {r.flag} |")
    R += ["", "## Every lemma (-ed probe form; counts for bare / -s / -ing / -ed)", "",
          "| lemma | -ed: Gemma pieces | -ed: Pythia pieces | Gemma n (bare / s / ing / ed) | Pythia n (bare / s / ing / ed) |",
          "|---|---|---|---|---|"]
    for lem in lemmas:
        cnt = lambda m: " / ".join(str(int(W.loc[lem, (f"{m}|n", fm)])) for fm in FORMS)
        R.append(f"| {lem} | `{W.loc[lem, (f'{G}|pieces', 'ed')]}` | `{W.loc[lem, (f'{Pm}|pieces', 'ed')]}` | {cnt(G)} | {cnt(Pm)} |")
    Path(a.report).write_text("\n".join(R) + "\n")
    print("\n".join(R[:20]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lemmas", default="data/nonce_passive/lemmas.csv")
    ap.add_argument("--out-dir", default="results/gemma/nonce_tokenization")
    ap.add_argument("--report", default="reports/gemma/nonce_tokenization.md")
    main(ap.parse_args())
