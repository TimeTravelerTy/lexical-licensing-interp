"""Small model-agnostic helpers for the Gemma replication (used only behind opt-in flags of the shared scorers).

- final_norm(model): the module whose output feeds the unembedding (GPT-NeoX final_layer_norm; Gemma / Llama
  model.norm).
- fp32_unembed(model): keep the transformer in its load dtype (bf16 for Gemma) but compute logits in fp32 from
  the final-norm output. The unembedding is untied from the input embedding first (Gemma ties them), so the
  embedding stays bf16. Reason: bf16 logits of magnitude ~30 are quantized in steps of 0.125-0.25 nats.
- check_no_softcap(model): the h . u - LSE reconstruction in run_frame_contrast.py assumes logits = h W^T.
"""

from __future__ import annotations


def final_norm(model):
    if hasattr(model, "gpt_neox"):
        return model.gpt_neox.final_layer_norm
    return model.model.norm


def embed_module(model):
    """Site 0 of the DAS runners: the token-embedding module (Gemma's output includes its sqrt(d) scaling)."""
    return model.gpt_neox.embed_in if hasattr(model, "gpt_neox") else model.model.embed_tokens


def layer_modules(model):
    """Decoder layers; site s >= 1 = output of layer s - 1."""
    return model.gpt_neox.layers if hasattr(model, "gpt_neox") else model.model.layers


def check_no_softcap(model):
    cap = getattr(model.config, "final_logit_softcapping", None)
    assert cap is None, f"final logit softcapping {cap}: logits are not h W^T"


def fp32_unembed(model):
    import torch

    head = model.get_output_embeddings()
    assert getattr(head, "bias", None) is None
    head.weight = torch.nn.Parameter(head.weight.detach().float().clone(), requires_grad=False)
    head.register_forward_pre_hook(lambda m, a: (a[0].float(),) + tuple(a[1:]))
    emb = model.get_input_embeddings()
    assert emb.weight.dtype != torch.float32 or model.dtype == torch.float32
    assert emb.weight.data_ptr() != head.weight.data_ptr(), "unembedding still tied to the embedding"
    return model
