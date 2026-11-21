"""
Escalonamento de contexto ate 1.000.000 de tokens sob o teto de 500 MB.
Compara 3 familias de atencao quanto a RAM e velocidade vs comprimento:
  - transformer (KV cresce linearmente)       -> explode em contexto longo
  - sliding_window (KV limitado a janela W)    -> RAM teto, perde recall global
  - ssm/linear (estado recorrente CONSTANTE)   -> 1M cabe, velocidade constante
Numeros = fisica (contagem de params + banda). Nao e geracao medida.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.explorer import (ModelConfig, BPP, KV_BYTES, RUNTIME_OVERHEAD_MB,
                                ACTIVATION_MB, RAM_LIMIT_MB)

EFF_BW = 13.6e9   # host: 20.3 GB/s medido x 0.67 efetivo

cfg = ModelConfig("500M", n_layers=28, d_model=1280, n_heads=20, n_kv_heads=1,
                  d_ff=3584, wfmt="ternary", kvfmt="q4", tokens_per_pass=4.0)

W_SLIDING = 32768          # janela de atencao local
# Estado SSM (Mamba-like): constante, independe do contexto
D_STATE = 16
D_INNER = 2 * cfg.d_model
SSM_STATE_BYTES = cfg.n_layers * D_INNER * D_STATE * 2   # bf16 state

def weights_mb():
    return cfg.total_params() * BPP[cfg.wfmt] / 1e6

def kv_state_mb(kind, seq):
    per_tok = cfg.kv_bytes_per_token()
    if kind == "transformer":
        return per_tok * seq / 1e6
    if kind == "sliding":
        return per_tok * min(seq, W_SLIDING) / 1e6
    if kind == "ssm":
        return SSM_STATE_BYTES / 1e6                     # CONSTANTE
    raise ValueError(kind)

def ram_mb(kind, seq):
    return weights_mb() + kv_state_mb(kind, seq) + RUNTIME_OVERHEAD_MB + ACTIVATION_MB

def tok_s(kind, seq):
    wbytes = cfg.active_params() * BPP[cfg.wfmt]
    if kind == "transformer":
        kvread = cfg.kv_bytes_per_token() * seq
    elif kind == "sliding":
        kvread = cfg.kv_bytes_per_token() * min(seq, W_SLIDING)
    else:  # ssm: le so o estado constante
        kvread = SSM_STATE_BYTES
    return cfg.tokens_per_pass * EFF_BW / (wbytes + kvread)

if __name__ == "__main__":
    seqs = [8_192, 32_768, 131_072, 524_288, 1_000_000]
    print(f"Modelo {cfg.total_params()/1e6:.0f}M ternario | pesos {weights_mb():.0f} MB "
          f"| limite RAM {RAM_LIMIT_MB:.0f} MB | estado SSM {SSM_STATE_BYTES/1e6:.1f} MB (fixo)\n")
    for kind in ("transformer", "sliding", "ssm"):
        print(f"### {kind.upper()}" + (f" (janela {W_SLIDING})" if kind=="sliding" else ""))
        print(f"{'contexto':>12}{'RAM MB':>10}{'tok/s':>9}  cabe500")
        for s in seqs:
            r = ram_mb(kind, s); t = tok_s(kind, s)
            ok = "OK" if r <= RAM_LIMIT_MB else "ESTOURA"
            print(f"{s:>12,}{r:>10.0f}{t:>9.0f}  {ok}")
        print()
    print("transformer: RAM e velocidade colapsam com o contexto.")
    print("sliding: RAM teto mas so enxerga a janela (perde recall global).")
    print("ssm/linear: 1M em RAM constante e velocidade constante -> unico que fecha.")
