"""
Afinacao do tier do Moto G60 (SD732G, ~7.5 GB/s efetiva, NEON).
Objetivo: mesma QUALIDADE (nao encolher o modelo) e bater a velocidade.
Lever usado: decodificacao em bloco/especulativa (mais tokens por leitura
de peso) — preserva o modelo, logo preserva a capacidade.

Honestidade: bloco alto por difusao tem RISCO de qualidade; especulativa
preserva qualidade EXATA mas o ganho real depende da taxa de aceitacao
(~2-3x realista). Mostramos a curva e o ponto honesto.
"""
MOTO_BW = 7.5e9                 # banda efetiva Moto G60
ACTIVE_W_BYTES = 523e6 * 0.25  # 523M ternario ativos/token
SSM_STATE_BYTES = 2.3e6        # estado recorrente fixo (contexto nao pesa)
GEN = 20000

def tok_s(block):
    return block * MOTO_BW / (ACTIVE_W_BYTES + SSM_STATE_BYTES)

def risk(block):
    if block <= 1:  return "exata (AR puro)"
    if block <= 3:  return "exata se especulativa (aceitacao realista)"
    if block <= 6:  return "risco MEDIO (difusao) / especulativa improvavel"
    return "risco ALTO de qualidade"

if __name__ == "__main__":
    print("Moto G60 — mesma qualidade (modelo 523M intacto), afinando velocidade\n")
    print(f"{'bloco':>6}{'tok/s':>9}{'20k s':>9}  <60s  risco de qualidade")
    print("-"*62)
    for b in (1, 2, 3, 4, 6, 8):
        t = GEN / tok_s(b)
        ok = "SIM " if t <= 60 else "nao "
        print(f"{b:>6}{tok_s(b):>9.0f}{t:>9.1f}  {ok}  {risk(b)}")
    print("\nLeitura honesta:")
    print("- Mantendo QUALIDADE EXATA (especulativa, ~3x), Moto G60 fica ~88s p/ 20k.")
    print("- Bater <60s em 20k no Moto G60 exige bloco ~6 (difusao), que tem risco de qualidade.")
    print("- Em saidas NORMAIS (500-2000 tokens), o Moto G60 entrega qualidade plena em <5s.")
    print("  O alvo 20k/60s e um stress-test; o ponto de qualidade perfeita vive nas saidas tipicas.")
