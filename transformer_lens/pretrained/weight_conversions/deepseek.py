import einops
import torch

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def convert_deepseek_weights(deepseek, cfg: HookedTransformerConfig):
    print(deepseek)
    state_dict = {}

    assert cfg.n_key_value_heads is not None  # keep mypy happy
    assert cfg.d_mlp is not None  # keep mypy happy

    state_dict["embed.W_E"] = deepseek.model.embed_tokens.weight

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = deepseek.model.layers[l].input_layernorm.weight

        W_Q = deepseek.model.layers[l].self_attn.q_proj.weight
        W_K = deepseek.model.layers[l].self_attn.kv_a_proj_with_mqa.weight
        W_V = deepseek.model.layers[l].self_attn.kv_b_proj.weight
     

        W_Q = einops.rearrange(W_Q, "(n h) m->n m h", n=cfg.n_heads)
        W_K = einops.rearrange(W_K, "(n h) m->n m h", n=cfg.n_key_value_heads)
        W_V = einops.rearrange(W_V, "(n h) m->n m h", n=cfg.n_key_value_heads)

        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn._W_K"] = W_K
        state_dict[f"blocks.{l}.attn._W_V"] = W_V

        state_dict[f"blocks.{l}.attn.b_Q"] = torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype)
        state_dict[f"blocks.{l}.attn._b_K"] = torch.zeros(
            cfg.n_key_value_heads, cfg.d_head, dtype=cfg.dtype
        )
        state_dict[f"blocks.{l}.attn._b_V"] = torch.zeros(
            cfg.n_key_value_heads, cfg.d_head, dtype=cfg.dtype
        )

        W_O = deepseek.model.layers[l].self_attn.o_proj.weight
        W_O = einops.rearrange(W_O, "m (n h)->n h m", n=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O

        state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

        if l == 0:
            state_dict[f"blocks.{l}.mlp.W_in"] = deepseek.model.layers[l].mlp.up_proj.weight.T
            state_dict[f"blocks.{l}.mlp.W_gate"] = deepseek.model.layers[l].mlp.gate_proj.weight.T
            state_dict[f"blocks.{l}.mlp.b_in"] = torch.zeros(cfg.d_mlp, dtype=cfg.dtype)

            state_dict[f"blocks.{l}.mlp.W_out"] = deepseek.model.layers[l].mlp.down_proj.weight.T
            state_dict[f"blocks.{l}.mlp.b_out"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)
        else:
            for e in range(cfg.num_experts):
                state_dict[f"blocks.{l}.mlp.experts.{e}.W_in.weight"] = (
                    deepseek.model.layers[l].block_sparse_moe.experts[e].w3.weight
                )
                state_dict[f"blocks.{l}.mlp.experts.{e}.W_gate.weight"] = (
                    deepseek.model.layers[l].block_sparse_moe.experts[e].w1.weight
                )
                state_dict[f"blocks.{l}.mlp.experts.{e}.W_out.weight"] = (
                    deepseek.model.layers[l].block_sparse_moe.experts[e].w2.weight
                )
        
    # deepseekRMSNorm adds 1 to weights before multiplying by input, keep RMS calcs in float32
    state_dict["ln_final.w"] = deepseek.model.norm.weight.float() + torch.ones_like(
        deepseek.model.norm.weight, dtype=torch.float32
    )

    state_dict["unembed.W_U"] = deepseek.lm_head.weight.T
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype)

    return state_dict
