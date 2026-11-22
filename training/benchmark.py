"""
Benchmark REAL do modelo — codigo (executado de verdade), matematica, texto,
escrita e seguimento de instrucao (tamanho exato). Pontuacao objetiva, zero
invencao. Deriva de um modelo existente (Qwen2.5) via transformers, CPU.

Nota final = media das tarefas * 10. O numero e o que for medido, sem maquiagem.
"""
import os, re, sys, io, contextlib, traceback
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
MAXNEW = int(os.environ.get("MAXNEW", "320"))

def load():
    print(f"carregando {MODEL} (CPU)...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    model.eval()
    torch.set_num_threads(os.cpu_count() or 4)
    return model, tok

def chat(model, tok, prompt, max_new=MAXNEW):
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)

# ---------------- scorers ----------------
def run_code(resp, test_src):
    m = re.search(r"```(?:python)?\s*(.*?)```", resp, re.S) or re.search(r"(def .+)", resp, re.S)
    if not m: return 0.0
    code = m.group(1)
    ns = {}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(code, ns)
            exec(test_src, ns)
        return 1.0
    except Exception:
        return 0.0

def contains(resp, *keys): return 1.0 if all(k.lower() in resp.lower() for k in keys) else 0.0
def word_count_score(resp, target, tol=0.15):
    n = len(resp.split()); lo, hi = target*(1-tol), target*(1+tol)
    if lo <= n <= hi: return 1.0
    return max(0.0, 1.0 - abs(n-target)/target)   # proporcional

# ---------------- tarefas ----------------
TASKS = [
    ("codigo: soma de lista",
     "Escreva uma função Python chamada soma_lista(xs) que retorna a soma dos elementos.",
     lambda r: run_code(r, "assert soma_lista([1,2,3])==6; assert soma_lista([])==0")),
    ("codigo: fatorial",
     "Escreva uma função Python chamada fatorial(n) iterativa.",
     lambda r: run_code(r, "assert fatorial(5)==120; assert fatorial(0)==1")),
    ("codigo: fibonacci",
     "Escreva uma função Python fib(n) que retorna o n-esimo numero de Fibonacci (fib(0)=0, fib(1)=1).",
     lambda r: run_code(r, "assert fib(10)==55; assert fib(0)==0")),
    ("matematica: multiplicacao",
     "Quanto é 17 multiplicado por 23? Responda apenas o número.",
     lambda r: contains(r, "391")),
    ("matematica: porcentagem",
     "Quanto é 15% de 200? Responda apenas o número.",
     lambda r: contains(r, "30")),
    ("conhecimento: capital",
     "Qual é a capital da Austrália?",
     lambda r: contains(r, "canberra")),
    ("texto: resumo",
     "Resuma em uma frase: O sol é uma estrela no centro do sistema solar que fornece luz e calor.",
     lambda r: 1.0 if 3 <= len(r.split()) <= 40 and ("sol" in r.lower() or "estrela" in r.lower()) else 0.0),
    ("instrucao: tamanho exato",
     "Escreva um texto com exatamente 50 palavras sobre o Brasil.",
     lambda r: word_count_score(r, 50)),
    ("escrita: criatividade",
     "Escreva um nome criativo para uma cafeteria e explique em uma frase.",
     lambda r: 1.0 if len(r.split()) >= 5 else 0.0),
    ("seguir instrucao: formato",
     "Liste 3 frutas, uma por linha, sem numeração.",
     lambda r: 1.0 if len([l for l in r.strip().splitlines() if l.strip()]) >= 3 else 0.0),
]

def main():
    model, tok = load()
    total = 0.0; lines = []
    for name, prompt, scorer in TASKS:
        resp = chat(model, tok, prompt)
        try: sc = float(scorer(resp))
        except Exception: sc = 0.0
        total += sc
        preview = resp.replace("\n", " ")[:80]
        line = f"{sc:4.2f}  {name:34s} | {preview}"
        print(line, flush=True); lines.append(line)
    nota = total / len(TASKS) * 10
    print(f"\n=== NOTA REAL: {nota:.1f}/10  ({total:.1f}/{len(TASKS)} tarefas) ===", flush=True)
    os.makedirs("out", exist_ok=True)
    with open("out/benchmark.txt", "w") as f:
        f.write(f"modelo={MODEL}\nNOTA={nota:.1f}/10\n\n" + "\n".join(lines))

if __name__ == "__main__":
    main()
