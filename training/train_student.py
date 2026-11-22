"""
Treino do ALUNO (muzi) no GitHub Actions — PyTorch, CPU.
GPT byte-level pequeno (vocab 256), treino em corpus real, validacao medida,
amostra de geracao e export ONNX. Honesto: escala que o runner aguenta.

Sucesso v1 = val loss cai + amostra coerente + ONNX exportado e recarregado.
"""
import os, math, time, urllib.request, sys
import torch, torch.nn as nn, torch.nn.functional as F

# ----------------------- config -----------------------
class Cfg:
    # enxuto para runner CPU — treino rapido (<15 min) para iterar
    n_layer = int(os.environ.get("NLAYER", "4"))
    n_head = 8
    n_embd = int(os.environ.get("NEMBD", "256"))
    block = int(os.environ.get("BLOCK", "128"))
    vocab = 256            # byte-level
    dropout = 0.0
    steps = int(os.environ.get("STEPS", "1000"))
    batch = int(os.environ.get("BATCH", "16"))
    lr = 3e-4
    eval_every = 200

# ----------------------- dados -----------------------
URLS = [
    # PT primeiro (Machado de Assis), com varias fontes; fallback tinyshakespeare
    "https://www.gutenberg.org/cache/epub/55752/pg55752.txt",   # Dom Casmurro
    "https://www.gutenberg.org/files/55752/55752-0.txt",
    "https://www.gutenberg.org/cache/epub/54829/pg54829.txt",   # Memorias Postumas
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
]
def load_corpus():
    hdr = {"User-Agent": "Mozilla/5.0 muzi-train"}
    for u in URLS:
        for attempt in range(3):
            try:
                print(f"baixando corpus (tentativa {attempt+1}): {u}", flush=True)
                req = urllib.request.Request(u, headers=hdr)
                raw = urllib.request.urlopen(req, timeout=60).read()
                if len(raw) > 100_000:
                    print(f"  ok: {len(raw)} bytes", flush=True)
                    return raw
            except Exception as e:
                print(f"  falhou: {e}", flush=True); time.sleep(3)
    raise RuntimeError("sem corpus")

# ----------------------- modelo -----------------------
class Block(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.ln1 = nn.LayerNorm(c.n_embd); s.ln2 = nn.LayerNorm(c.n_embd)
        s.qkv = nn.Linear(c.n_embd, 3*c.n_embd, bias=False)
        s.proj = nn.Linear(c.n_embd, c.n_embd, bias=False)
        s.mlp = nn.Sequential(nn.Linear(c.n_embd, 4*c.n_embd), nn.GELU(),
                              nn.Linear(4*c.n_embd, c.n_embd))
        s.nh = c.n_head; s.hd = c.n_embd // c.n_head
    def forward(s, x):
        B, T, C = x.shape
        q, k, v = s.qkv(s.ln1(x)).split(C, dim=2)
        q = q.view(B, T, s.nh, s.hd).transpose(1, 2)
        k = k.view(B, T, s.nh, s.hd).transpose(1, 2)
        v = v.view(B, T, s.nh, s.hd).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        x = x + s.proj(y)
        x = x + s.mlp(s.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.c = c
        s.tok = nn.Embedding(c.vocab, c.n_embd)
        s.pos = nn.Embedding(c.block, c.n_embd)
        s.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layer)])
        s.lnf = nn.LayerNorm(c.n_embd)
        s.head = nn.Linear(c.n_embd, c.vocab, bias=False)
        s.head.weight = s.tok.weight   # tie
    def forward(s, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = s.tok(idx) + s.pos(pos)[None]
        for b in s.blocks: x = b(x)
        return s.head(s.lnf(x))

# ----------------------- treino -----------------------
def main():
    c = Cfg()
    torch.manual_seed(0)
    data = load_corpus()
    ids = torch.tensor(list(data), dtype=torch.long)
    n = int(0.9 * len(ids)); train, val = ids[:n], ids[n:]
    print(f"corpus {len(ids)} bytes | treino {len(train)} | val {len(val)}")

    def batch(split):
        d = train if split == "train" else val
        i = torch.randint(0, len(d) - c.block - 1, (c.batch,))
        x = torch.stack([d[j:j+c.block] for j in i])
        y = torch.stack([d[j+1:j+1+c.block] for j in i])
        return x, y

    model = GPT(c)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"modelo: {nparams/1e6:.1f}M params | {c.n_layer}L {c.n_embd}d | steps {c.steps}")
    opt = torch.optim.AdamW(model.parameters(), lr=c.lr)

    @torch.no_grad()
    def evaluate():
        model.eval(); losses = []
        for _ in range(10):
            x, y = batch("val")
            logits = model(x)
            losses.append(F.cross_entropy(logits.view(-1, c.vocab), y.view(-1)).item())
        model.train(); return sum(losses)/len(losses)

    t0 = time.time()
    for step in range(c.steps):
        x, y = batch("train")
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, c.vocab), y.view(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % c.eval_every == 0 or step == c.steps-1:
            vl = evaluate()
            print(f"step {step:4d} | train {loss.item():.3f} | val {vl:.3f} | ppl {math.exp(vl):.1f} | {time.time()-t0:.0f}s", flush=True)

    # amostra de geracao
    model.eval()
    ctx = torch.tensor([list("O ".encode())], dtype=torch.long)
    for _ in range(200):
        logits = model(ctx[:, -c.block:])
        nx = torch.multinomial(F.softmax(logits[0, -1]/0.8, dim=-1), 1)
        ctx = torch.cat([ctx, nx[None]], dim=1)
    sample = bytes(ctx[0].tolist()).decode("utf-8", errors="replace")
    print("\n=== AMOSTRA ===\n" + sample[:400] + "\n===============")

    # salva checkpoint + relatorio ANTES do export (nao perder o treino)
    os.makedirs("out", exist_ok=True)
    final_val = evaluate()
    torch.save(model.state_dict(), "out/muzi_student.pt")
    with open("out/report.txt", "w") as f:
        f.write(f"params={nparams/1e6:.1f}M val_loss={final_val:.3f} ppl={math.exp(final_val):.1f}\n")
        f.write("sample:\n" + sample[:400])
    print(f"checkpoint salvo | val_loss {final_val:.3f}", flush=True)

    # export ONNX (prefill: input_ids -> logits), blindado
    try:
        dummy = torch.zeros(1, 16, dtype=torch.long)
        torch.onnx.export(model, dummy, "out/model.onnx",
                          input_names=["input_ids"], output_names=["logits"],
                          dynamic_axes={"input_ids": {0: "b", 1: "t"}, "logits": {0: "b", 1: "t"}},
                          opset_version=17)
        print("ONNX exportado em out/model.onnx", flush=True)
        import onnxruntime as ort, numpy as np
        s = ort.InferenceSession("out/model.onnx")
        o = s.run(None, {"input_ids": np.zeros((1, 8), dtype=np.int64)})
        print(f"ONNX recarregado OK, logits shape {o[0].shape}", flush=True)
    except Exception as e:
        print(f"AVISO: export/recarga ONNX falhou: {e}", flush=True)

if __name__ == "__main__":
    main()
