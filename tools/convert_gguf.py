#!/usr/bin/env python3
"""Convert the lchat 50M checkpoint (lmlib.GPT, Llama-style) to a GGUF file (llama arch, f32 or f16).

usage: python convert_gguf.py <model.pt> <tokenizer.json> <out.gguf> [f16|f32]
RoPE in lmlib rotates adjacent pairs (x[::2], x[1::2]) = llama.cpp NORM mode, so q/k need no permutation.
"""
import sys, json
import numpy as np, torch
import gguf

pt, tokp, out = sys.argv[1:4]
ftype = sys.argv[4] if len(sys.argv) > 4 else "f16"
ck = torch.load(pt, map_location="cpu", weights_only=True)
cfg, sd = ck["cfg"], ck["model"]
d, L, H, hid, V, ctx = cfg["d"], cfg["n_layers"], cfg["n_heads"], cfg["hidden"], cfg["vocab"], cfg["ctx"]

w = gguf.GGUFWriter(out, "llama")
w.add_name("bytenpc-50m")
w.add_context_length(ctx)
w.add_embedding_length(d)
w.add_block_count(L)
w.add_feed_forward_length(hid)
w.add_head_count(H)
w.add_head_count_kv(H)
w.add_rope_dimension_count(d // H)
w.add_rope_freq_base(cfg.get("rope_base", 10000.0))
w.add_layer_norm_rms_eps(1e-5)
w.add_vocab_size(V)
w.add_file_type(gguf.LlamaFileType.MOSTLY_F16 if ftype == "f16" else gguf.LlamaFileType.ALL_F32)

tj = json.load(open(tokp))
vocab = tj["model"]["vocab"]
added = {a["id"]: a["content"] for a in tj["added_tokens"]}
toks = [None] * V
for t, i in vocab.items():
    toks[i] = t
for i, t in added.items():
    toks[i] = t
assert all(t is not None for t in toks), "vocab has holes"
types = [gguf.TokenType.CONTROL if i in added else gguf.TokenType.NORMAL for i in range(V)]
merges = [" ".join(m) if isinstance(m, list) else m for m in tj["model"]["merges"]]
w.add_tokenizer_model("gpt2")
w.add_tokenizer_pre("gpt-2")
w.add_token_list(toks)
w.add_token_types(types)
w.add_token_merges(merges)
w.add_bos_token_id(0)
w.add_eos_token_id(0)
w.add_add_bos_token(False)
w.add_add_eos_token(False)

dt = np.float16 if ftype == "f16" else np.float32
def put(name, t, force32=False):
    a = t.float().numpy()
    w.add_tensor(name, a.astype(np.float32 if (force32 or a.ndim == 1) else dt))

put("token_embd.weight", sd["emb.weight"])
put("output_norm.weight", sd["norm.weight"])
for i in range(L):
    p = f"blocks.{i}."
    qkv = sd[p + "qkv.weight"]
    put(f"blk.{i}.attn_norm.weight", sd[p + "n1.weight"])
    put(f"blk.{i}.attn_q.weight", qkv[:d])
    put(f"blk.{i}.attn_k.weight", qkv[d:2 * d])
    put(f"blk.{i}.attn_v.weight", qkv[2 * d:])
    put(f"blk.{i}.attn_output.weight", sd[p + "o.weight"])
    put(f"blk.{i}.ffn_norm.weight", sd[p + "n2.weight"])
    put(f"blk.{i}.ffn_gate.weight", sd[p + "w1.weight"])
    put(f"blk.{i}.ffn_up.weight", sd[p + "w3.weight"])
    put(f"blk.{i}.ffn_down.weight", sd[p + "w2.weight"])

w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print("wrote", out)
