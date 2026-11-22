"""
Destilação com PROFESSOR REAL (Qwen2.5) -> aluno muzi (pequeno, byte-level).
Sequence-level distillation: o professor gera respostas em PT para um banco
de prompts; o aluno treina para reproduzir essas respostas (+ Wikipedia PT
para fluência). Objetivo: coerência real num modelo pequeno e rápido.

Honesto: o aluno é pequeno; ele imita o professor nos padrões vistos, não
iguala o professor. A amostra final mostra a verdade.
"""
import os, math, time, random
import torch, torch.nn.functional as F
from train_student import GPT, Cfg

TEACHER = os.environ.get("TEACHER", "Qwen/Qwen2.5-0.5B-Instruct")
N_SAMPLES = int(os.environ.get("NSAMPLES", "2"))      # variações por prompt
MAXNEW = int(os.environ.get("MAXNEW", "110"))

# ---- banco de prompts PT (diverso: QA, instrução, math, código, escrita) ----
PROMPTS = [
    "O que é o Sol?", "Explique o que é fotossíntese em uma frase.",
    "Qual é a capital do Brasil?", "O que é a água?",
    "Explique o que é gravidade de forma simples.", "O que faz um coração?",
    "Diga um sinônimo de feliz.", "O que é um computador?",
    "Explique o que é a internet.", "O que são números primos?",
    "Escreva uma frase motivacional.", "Dê uma dica de saúde.",
    "O que é reciclagem?", "Explique o que é democracia.",
    "O que é a Amazônia?", "Como se faz um bolo simples? Liste os passos.",
    "O que é inteligência artificial?", "Explique o que é o vento.",
    "Diga três frutas.", "O que é a Lua?",
    "Quanto é 12 mais 15?", "Quanto é 7 vezes 8?",
    "O que é um verbo?", "Explique o que é energia solar.",
    "O que é o oceano?", "Dê um conselho para estudar melhor.",
    "O que é um mamífero?", "Explique o que é a chuva.",
    "Escreva uma saudação amigável.", "O que é o cérebro humano?",
    "O que significa a palavra 'liberdade'?", "Explique o que é um livro.",
    "O que é a eletricidade?", "Dê uma dica para economizar dinheiro.",
    "O que é um triângulo?", "Explique o que é música.",
    "O que é o fogo?", "Diga um animal que vive na água.",
    "O que é o tempo (clima)?", "Explique o que é amizade.",
]

def build_distilled_corpus():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"carregando professor {TEACHER}...", flush=True)
    tok = AutoTokenizer.from_pretrained(TEACHER)
    teacher = AutoModelForCausalLM.from_pretrained(TEACHER, torch_dtype=torch.float32).eval()
    torch.set_num_threads(os.cpu_count() or 4)

    convos = []
    t0 = time.time()
    for i, p in enumerate(PROMPTS):
        for s in range(N_SAMPLES):
            msgs = [{"role": "user", "content": p}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = tok(text, return_tensors="pt")
            with torch.no_grad():
                out = teacher.generate(**ids, max_new_tokens=MAXNEW,
                                       do_sample=(s > 0), temperature=0.7,
                                       pad_token_id=tok.eos_token_id)
            ans = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True).strip()
            convos.append(f"Pergunta: {p}\nResposta: {ans}\n")
        if i % 5 == 0:
            print(f"  professor {i}/{len(PROMPTS)} | {time.time()-t0:.0f}s", flush=True)
    print(f"destilados {len(convos)} exemplos em {time.time()-t0:.0f}s", flush=True)
    # fluência: Wikipedia PT
    wiki = b""
    try:
        from datasets import load_dataset
        ds = load_dataset("wikimedia/wikipedia", "20231101.pt", split="train", streaming=True)
        parts = []; total = 0
        for ex in ds:
            parts.append(ex.get("text", "")); total += len(parts[-1])
            if total > 3_000_000: break
        wiki = ("\n\n".join(parts)).encode("utf-8")
        print(f"wikipedia PT: {len(wiki)} bytes", flush=True)
    except Exception as e:
        print(f"wiki falhou: {e}", flush=True)
    # repete os destilados para dar peso (instrução importa mais)
    distilled = ("\n".join(convos) + "\n") * 8
    return distilled.encode("utf-8") + b"\n\n" + wiki

def main():
    c = Cfg()
    torch.manual_seed(0)
    data = build_distilled_corpus()
    ids = torch.tensor(list(data), dtype=torch.long)
    n = int(0.95 * len(ids)); train, val = ids[:n], ids[n:]
    print(f"corpus {len(ids)} bytes | treino {len(train)} | val {len(val)}", flush=True)

    def batch(split):
        d = train if split == "train" else val
        i = torch.randint(0, len(d) - c.block - 1, (c.batch,))
        x = torch.stack([d[j:j+c.block] for j in i])
        y = torch.stack([d[j+1:j+1+c.block] for j in i])
        return x, y

    model = GPT(c)
    print(f"aluno: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params | steps {c.steps}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=c.lr)

    @torch.no_grad()
    def evaluate():
        model.eval(); L = []
        for _ in range(10):
            x, y = batch("val"); L.append(F.cross_entropy(model(x).view(-1, c.vocab), y.view(-1)).item())
        model.train(); return sum(L)/len(L)

    t0 = time.time()
    for step in range(c.steps):
        x, y = batch("train")
        loss = F.cross_entropy(model(x).view(-1, c.vocab), y.view(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 300 == 0 or step == c.steps-1:
            print(f"step {step:4d} | train {loss.item():.3f} | val {evaluate():.3f} | {time.time()-t0:.0f}s", flush=True)

    # amostra: pergunta -> resposta (testa a destilação)
    model.eval()
    prompt = "Pergunta: O que é o Sol?\nResposta:"
    ctx = torch.tensor([list(prompt.encode())], dtype=torch.long)
    for _ in range(160):
        logits = model(ctx[:, -c.block:])
        nx = torch.multinomial(F.softmax(logits[0, -1]/0.7, dim=-1), 1)
        ctx = torch.cat([ctx, nx[None]], dim=1)
    sample = bytes(ctx[0].tolist()).decode("utf-8", errors="replace")
    print("\n=== AMOSTRA ===\n" + sample[:400] + "\n===============", flush=True)

    os.makedirs("out", exist_ok=True)
    torch.save(model.state_dict(), "out/muzi_student.pt")
    with open("out/report.txt", "w") as f:
        f.write(f"val_loss={evaluate():.3f}\nsample:\n{sample[:400]}")
    try:
        dummy = torch.zeros(1, 16, dtype=torch.long)
        torch.onnx.export(model, dummy, "out/model.onnx",
                          input_names=["input_ids"], output_names=["logits"],
                          dynamic_axes={"input_ids": {0: "b", 1: "t"}, "logits": {0: "b", 1: "t"}},
                          opset_version=18)
        import onnxruntime as ort, numpy as np
        ort.InferenceSession("out/model.onnx").run(None, {"input_ids": np.zeros((1, 8), dtype=np.int64)})
        print("ONNX OK", flush=True)
    except Exception as e:
        print(f"AVISO ONNX: {e}", flush=True)

if __name__ == "__main__":
    main()
