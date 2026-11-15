"""
Reference for the HalfRotation layout of RotaryEmbedding: HuggingFace Llama's rotary embedding (`x * cos + rotate_half(x) * sin`),
taken from transformers.models.llama.modeling_llama. Run ONCE; the result is versioned in tests/fixtures/rope_hf.safetensors
(sha256 in tests/fixtures/MANIFEST.md). Python is only needed to regenerate it, never at test or run time.

Needs `transformers`, `torch` and `safetensors`. Recipe used (transformers 4.47.1, torch 2.7.0, CPU, the DiffRhythm2 image):

    docker run --rm --user $(id -u):$(id -g) --entrypoint python3 \
        -v ~/Proyectos/musicgen-core:/work -e PYTHONDONTWRITEBYTECODE=1 mediaorquestator/diffrhythm2:1 \
        -I /work/tools/dump_rope_hf.py --out /work/tests/fixtures/rope_hf.safetensors

Tensors written, per case `c` in (a, b): `c.x` [B, H, S, D] fp32, `c.pos` [B, S] int64, `c.out` [B, H, S, D] fp32 (the rotated x),
and `c.params` [2] = (head_dim, rope_theta). Case a: contiguous positions; case b: a larger head and theta with non-contiguous positions.
"""
import argparse

import torch
from safetensors.torch import save_file
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb


def case(name, head_dim, theta, positions, batch=2, heads=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    cfg = LlamaConfig(hidden_size=head_dim * heads, num_attention_heads=heads, num_key_value_heads=heads,
                      rope_theta=theta, max_position_embeddings=8192)
    rope = LlamaRotaryEmbedding(config=cfg)
    pos = torch.tensor(positions, dtype=torch.long)
    assert pos.shape[0] == batch
    x = torch.randn(batch, heads, pos.shape[1], head_dim, generator=g, dtype=torch.float32) * 2
    cos, sin = rope(x, pos)
    out, _ = apply_rotary_pos_emb(x, x, cos, sin)   # q and k are rotated alike; the second result is discarded
    return {f"{name}.x": x, f"{name}.pos": pos, f"{name}.out": out.contiguous(),
            f"{name}.params": torch.tensor([float(head_dim), float(theta)])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tensors = {}
    tensors.update(case("a", 64, 10000.0, [list(range(16)), list(range(16))], seed=1))
    tensors.update(case("b", 128, 500000.0, [[5, 6, 7, 1000, 1001, 4000, 4001, 4002], [0, 3, 9, 27, 81, 243, 729, 2187]], seed=2))
    save_file(tensors, a.out)
    print(f"wrote {a.out}: {len(tensors)} tensors; transformers {__import__('transformers').__version__}, torch {torch.__version__}")


if __name__ == "__main__":
    main()
