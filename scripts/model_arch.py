"""Per-architecture map for the scorers and DAS runners: hook points, BOS and precision defaults, scaling quirks.

Adding a model = adding an ARCH entry keyed by its `config.model_type`. Every entry gives:
- embed:       module whose output is DAS site 0 (the residual stream entering layer 0);
- layers:      decoder layer list; DAS site s >= 1 = output of layers[s - 1];
- final_norm:  module whose output feeds the unembedding;
- bos:         prepend the tokenizer's BOS id to every prompt (None = undecided: callers must pass it explicitly);
- fp32_unembed: compute logits in fp32 from the final-norm output (default for bf16 models other than Pythia;
               Pythia keeps its stored bf16 logits so its numbers do not move);
- embed_scale: where any embedding scaling happens. It must be inside `embed`, so that the hooked site-0 output and
               output_hidden_states[0] agree; check_sites() verifies this (and every layer site) on a real forward.

Known quirk (all three architectures, left unchanged): HF's last output_hidden_states entry is taken after the final
norm, while the hook on the last layer sees the pre-norm output. So site n_sites - 1 source states (from
output_hidden_states) do not match the base states the hook patches; check_sites() reports it, and it is
excluded from the sweeps' conclusions.
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import attrgetter


@dataclass(frozen=True)
class Arch:
    embed: str
    layers: str
    final_norm: str
    bos: bool | None
    fp32_unembed: bool
    embed_scale: str
    notes: str = ""


ARCH = {
    "gpt_neox": Arch(embed="gpt_neox.embed_in", layers="gpt_neox.layers", final_norm="gpt_neox.final_layer_norm",
                     bos=False, fp32_unembed=False, embed_scale="none",
                     notes="Pythia. No BOS and bf16 logits in every stored run; untied embed_out."),
    "gemma3_text": Arch(embed="model.embed_tokens", layers="model.layers", final_norm="model.norm",
                        bos=True, fp32_unembed=True,
                        embed_scale="sqrt(hidden_size), inside embed_tokens (Gemma3TextScaledWordEmbedding)",
                        notes="Gemma-3 text models (1B). Tied lm_head (fp32_unembed unties it). No final-logit softcap."),
    "olmo2": Arch(embed="model.embed_tokens", layers="model.layers", final_norm="model.norm",
                  bos=None, fp32_unembed=True, embed_scale="none",
                  notes="OLMo-2 (e.g. allenai/OLMo-2-0425-1B). Untied lm_head. The tokenizer adds no BOS and its "
                        "bos_token is <|endoftext|> (= EOS); whether to prepend it is undecided."),
}


def arch(model) -> Arch:
    mt = model.config.model_type
    if mt not in ARCH:
        raise KeyError(f"no ARCH entry for model_type {mt!r} (known: {sorted(ARCH)}); add one in model_arch.py")
    return ARCH[mt]


def embed_module(model):
    return attrgetter(arch(model).embed)(model)


def layer_modules(model):
    return attrgetter(arch(model).layers)(model)


def final_norm(model):
    return attrgetter(arch(model).final_norm)(model)


def resolve(model, bos=None, fp32_unembed=None):
    """Explicit settings win; otherwise the ARCH defaults. An undecided BOS must be set explicitly."""
    a = arch(model)
    b = a.bos if bos is None else bos
    if b is None:
        raise SystemExit(f"{model.config.model_type}: BOS undecided in model_arch.ARCH; pass --bos or --no-bos")
    return bool(b), bool(a.fp32_unembed if fp32_unembed is None else fp32_unembed)


def check_no_softcap(model):
    cap = getattr(model.config, "final_logit_softcapping", None)
    assert cap is None, f"final logit softcapping {cap}: logits are not h W^T"


def fp32_unembed(model):
    """Keep the transformer in its load dtype but compute logits in fp32 from the final-norm output.

    The unembedding is untied from the input embedding first (Gemma ties them), so the embedding keeps its dtype.
    Reason: bf16 logits of magnitude ~30 are quantized in steps of 0.125-0.25 nats.
    """
    import torch

    head = model.get_output_embeddings()
    assert getattr(head, "bias", None) is None
    head.weight = torch.nn.Parameter(head.weight.detach().float().clone(), requires_grad=False)
    head.register_forward_pre_hook(lambda m, a: (a[0].float(),) + tuple(a[1:]))
    emb = model.get_input_embeddings()
    assert emb.weight.data_ptr() != head.weight.data_ptr(), "unembedding still tied to the embedding"
    return model


def check_sites(model, input_ids, attention_mask):
    """Hooked site outputs vs output_hidden_states on one forward (exact equality expected at sites 0..L-1).

    Returns (max |diff| over sites 0..L-1, |diff| at the last site L). Site 0 = embed output, which must already
    include any embedding scaling; site L is the known post-final-norm mismatch.
    """
    import torch

    got = {}

    def grab(s):
        return lambda m, i, o: got.__setitem__(s, (o[0] if isinstance(o, tuple) else o).detach())

    hs = [embed_module(model).register_forward_hook(grab(0))]
    for s, layer in enumerate(layer_modules(model), start=1):
        hs.append(layer.register_forward_hook(grab(s)))
    try:
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
    finally:
        for h in hs:
            h.remove()
    L = len(layer_modules(model))
    m = attention_mask.bool()
    diff = [float((got[s][m].float() - out.hidden_states[s][m].float()).abs().max()) for s in range(L + 1)]
    return max(diff[:L]), diff[L]
