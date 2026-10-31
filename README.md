# ByteNPC

A 49.2M-parameter language model, trained from scratch, that plays fantasy NPCs. It runs entirely in the browser. **Live demo:** https://seenmttai.github.io/bytenpc/

- Pretrained on TinyStories + SimpleStories (about 1.08B tokens, one pass) with our own 8,192-token byte-level BPE tokenizer.
- Fine-tuned on LIGHT dialogues (8,307), then DPO with 2,000 pairs.
- Llama-style architecture: 14 layers, d=512, 8 heads, RoPE, RMSNorm, SwiGLU, tied embeddings, 1,024-token context.
- Served as a Q8_0 GGUF (`model/bytenpc-50m-q8_0.gguf`, about 50 MB) through [wllama](https://github.com/ngxson/wllama) 3.6.1 (llama.cpp in WebAssembly, MIT, vendored in `vendor/wllama`).

Prompt format (one character per turn, the model writes the reply after the last `<|turn|>name:`):

```
<|setting|>{place}. {kind}. {description} {background}
<|persona|>{you}: {your persona}
<|persona|>{them}: {their persona}
<|turn|>{you}: {line}
<|turn|>{them}:
```

`tools/convert_gguf.py` converts the PyTorch checkpoint to GGUF (llama arch; RoPE uses adjacent pairs, so q/k need no permutation).
