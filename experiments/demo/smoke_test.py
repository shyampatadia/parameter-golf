"""Local smoke test for the demo model wrapper.

Loads a checkpoint + sentencepiece model, runs a prefix through, and prints
output shapes plus a top-k readout. No GPU required.

Usage (from repo root):
    python experiments/demo/smoke_test.py \
        --ckpt path/to/final_model.pt \
        --tokenizer path/to/fineweb_1024_bpe.model \
        --prompt "The quick brown fox" \
        --num-layers 9 --pos-enc partial --rope-dims 32 --num-kv-heads 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import sentencepiece as spm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import GPT


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--tokenizer", required=True, type=Path)
    p.add_argument("--prompt", default="The quick brown fox")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--num-layers", type=int, default=9)
    p.add_argument("--model-dim", type=int, default=512)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--num-kv-heads", type=int, default=4)
    p.add_argument("--mlp-mult", type=int, default=2)
    p.add_argument("--vocab-size", type=int, default=1024)
    p.add_argument("--pos-enc", choices=["rope", "partial", "none", "learned"], default="partial")
    p.add_argument("--rope-dims", type=int, default=32)
    p.add_argument("--qk-gain-init", type=float, default=1.5)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading tokenizer: {args.tokenizer}")
    sp = spm.SentencePieceProcessor()
    sp.load(str(args.tokenizer))

    print(f"Building model: {args.num_layers}L dim={args.model_dim} heads={args.num_heads} "
          f"kv={args.num_kv_heads} mlp_mult={args.mlp_mult} pos={args.pos_enc} rope_dims={args.rope_dims}")
    model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        pos_enc=args.pos_enc,
        rope_dims=args.rope_dims,
        qk_gain_init=args.qk_gain_init,
    )

    print(f"Loading checkpoint: {args.ckpt}")
    state = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    model.eval().to(args.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {n_params:,}")

    ids = sp.encode(args.prompt, out_type=int)
    print(f"\nPrompt: {args.prompt!r}")
    print(f"Token IDs ({len(ids)}): {ids}")
    print(f"Decoded pieces: {[sp.id_to_piece(i) for i in ids]}")

    input_ids = torch.tensor([ids], dtype=torch.long, device=args.device)

    with torch.no_grad():
        logits, attn_per_layer, hidden_norms = model.forward_with_intermediates(input_ids)

    print(f"\nlogits shape: {tuple(logits.shape)}")
    print(f"attn_per_layer: {len(attn_per_layer)} tensors of shape {tuple(attn_per_layer[0].shape)}")
    print(f"hidden_norms: {len(hidden_norms)} tensors of shape {tuple(hidden_norms[0].shape)}")

    next_logits = logits[0, -1]
    probs = torch.softmax(next_logits, dim=-1)
    top_p, top_i = probs.topk(args.top_k)
    print(f"\nTop-{args.top_k} next-token predictions after {args.prompt!r}:")
    for p, i in zip(top_p.tolist(), top_i.tolist()):
        piece = sp.id_to_piece(int(i)).replace("▁", "_")
        print(f"  {p:>6.2%}  id={i:>4}  '{piece}'")

    print(f"\nMean attention entropy per layer (lower = sharper):")
    for li, attn in enumerate(attn_per_layer):
        a = attn[0]
        mask = a > 0
        ent = -(a * torch.log(a.clamp_min(1e-12)) * mask).sum(dim=-1).mean().item()
        print(f"  L{li}: {ent:.3f} nats")

    print(f"\nMean hidden-state L2 norm per layer:")
    for li, h in enumerate(hidden_norms):
        print(f"  L{li}: {h.mean().item():.3f}")


if __name__ == "__main__":
    main()
