"""
Explorador do espaco de design — LLM Distillation Lab
Fisico-engenheiro: calcula RAM residente real e teto de tok/s pela banda
de memoria medida, para cada arquitetura. Guardrails GR-RAM/GR-VEL embutidos.

NAO inventa resultados: todo numero sai de contagem de parametros + fisica
de banda. tok/s e TETO (ceiling), nao medicao de geracao real (essa exige
o modelo baixado). Marcado como tal.
"""
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# CONSTANTES DE HARDWARE (medidas nesta maquina)
# ---------------------------------------------------------------------------
BANDWIDTH_GBS      = 21.0   # leitura sequencial medida (COPY @512MB)
DECODE_EFFICIENCY  = 0.65   # fracao realista alcancavel em decode (2 nucleos, dequant)
RAM_LIMIT_MB       = 500.0  # GUARDRAIL duro
RUNTIME_OVERHEAD_MB= 55.0   # runtime classe llama.cpp (alocador, grafo, threads)
ACTIVATION_MB      = 12.0   # buffers temporarios de ativacao

# bytes por parametro por formato (inclui overhead de empacotamento)
BPP = {
    "fp16":    2.0,
    "int8":    1.0,
    "q8":      1.06,
    "q4":      0.5625,   # Q4_K_M ~4.5 bits
    "ternary": 0.25,     # BitNet b1.58 ~1.58 bits empacotado + overhead
}
KV_BYTES = {"fp16": 2.0, "q8": 1.0, "q4": 0.5}

# ---------------------------------------------------------------------------
# CONFIGURACAO DO MODELO ALUNO (~500M classe)
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    name: str
    n_layers: int = 24
    d_model: int = 1024
    n_heads: int = 16
    n_kv_heads: int = 4          # GQA
    d_ff: int = 2816             # SwiGLU
    vocab: int = 32000
    # MoE
    n_experts: int = 1           # 1 = denso
    top_k: int = 1
    # formato
    wfmt: str = "q4"
    kvfmt: str = "q8"
    # decodificacao
    tokens_per_pass: float = 1.0 # AR=1 ; bloco/difusao > 1 (efetivo)

    @property
    def head_dim(self):
        return self.d_model // self.n_heads

    def _attn_params(self):
        kv_dim = self.n_kv_heads * self.head_dim
        q = self.d_model * self.d_model
        k = self.d_model * kv_dim
        v = self.d_model * kv_dim
        o = self.d_model * self.d_model
        return q + k + v + o

    def _mlp_params_per_expert(self):
        # SwiGLU: gate + up (d_model->d_ff) + down (d_ff->d_model)
        return 3 * self.d_model * self.d_ff

    def total_params(self):
        emb = self.vocab * self.d_model  # tied embedding
        per_layer = self._attn_params() + self.n_experts * self._mlp_params_per_expert()
        router = self.n_layers * self.d_model * self.n_experts if self.n_experts > 1 else 0
        return emb + self.n_layers * per_layer + router

    def active_params(self):
        """Parametros efetivamente lidos por token (MoE ativa so top_k)."""
        emb = self.vocab * self.d_model
        per_layer = self._attn_params() + self.top_k * self._mlp_params_per_expert()
        return emb + self.n_layers * per_layer

    def kv_bytes_per_token(self):
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * KV_BYTES[self.kvfmt]

# ---------------------------------------------------------------------------
# FISICA
# ---------------------------------------------------------------------------
def resident_ram_mb(cfg, seq_len):
    weights = cfg.total_params() * BPP[cfg.wfmt]           # TODOS residentes (MoE inclui frios)
    kv = cfg.kv_bytes_per_token() * seq_len
    return (weights + kv) / 1e6 + RUNTIME_OVERHEAD_MB + ACTIVATION_MB

def tokens_per_second(cfg, avg_seq):
    bytes_weights = cfg.active_params() * BPP[cfg.wfmt]    # so ativos sao lidos
    bytes_kv      = cfg.kv_bytes_per_token() * avg_seq     # le todo o KV por token
    bytes_per_pass = bytes_weights + bytes_kv
    eff_bw = BANDWIDTH_GBS * 1e9 * DECODE_EFFICIENCY
    return cfg.tokens_per_pass * eff_bw / bytes_per_pass

# ---------------------------------------------------------------------------
# GUARDRAILS
# ---------------------------------------------------------------------------
def evaluate(cfg, prompt=512, gen=20000):
    max_seq = prompt + gen
    avg_seq = prompt + gen / 2
    peak_ram = resident_ram_mb(cfg, max_seq)
    tps = tokens_per_second(cfg, avg_seq)
    time_20k = gen / tps
    gr_ram = peak_ram <= RAM_LIMIT_MB                       # GR-RAM-04
    gr_vel = time_20k <= 60.0                               # alvo prompt
    return {
        "name": cfg.name,
        "total_M": cfg.total_params() / 1e6,
        "active_M": cfg.active_params() / 1e6,
        "wfmt": cfg.wfmt,
        "peak_ram_mb": peak_ram,
        "tok_s": tps,
        "time_20k_s": time_20k,
        "GR_RAM": gr_ram,
        "GR_VEL": gr_vel,
    }

# ---------------------------------------------------------------------------
# ARQUITETURAS A COMPARAR
# ---------------------------------------------------------------------------
def build_architectures():
    A = []
    # 1. Denso Q8 (referencia "sem truque")
    A.append(ModelConfig("1. Denso INT8 (ref)", wfmt="q8"))
    # 2. Denso Q4 (estado da arte comum)
    A.append(ModelConfig("2. Denso Q4", wfmt="q4"))
    # 3. Denso Ternario (BitNet b1.58) — quebra bytes/peso
    A.append(ModelConfig("3. Denso Ternario b1.58", wfmt="ternary"))
    # 4. MoE Q4 residente (8 experts, top-1) — menos ativos/token
    A.append(ModelConfig("4. MoE Q4 (8e, top1)", n_experts=8, top_k=1, wfmt="q4"))
    # 5. MoE Ternario (8e, top1) — combina os dois vetores
    A.append(ModelConfig("5. MoE Ternario (8e, top1)", n_experts=8, top_k=1, wfmt="ternary"))
    # 6. INOVACAO: MoE Ternario + decode em bloco (difusao, ~4 tok/pass efetivo)
    A.append(ModelConfig("6. MoE Ternario + bloco", n_experts=8, top_k=1,
                         wfmt="ternary", tokens_per_pass=4.0))
    # 7. Denso Ternario + MQA + KV Q4 (ataca o KV cache, que domina em 20k)
    A.append(ModelConfig("7. Denso Tern + MQA + KVq4", wfmt="ternary",
                         n_kv_heads=1, kvfmt="q4"))
    # 8. STACK FINAL: Denso Ternario + MQA + KV Q4 + decode em bloco
    A.append(ModelConfig("8. FINAL Tern+MQA+KVq4+bloco", wfmt="ternary",
                         n_kv_heads=1, kvfmt="q4", tokens_per_pass=4.0))
    # 9. STACK FINAL na ESCALA REAL de ~500M parametros
    A.append(ModelConfig("9. FINAL @500M (stack full)", n_layers=28, d_model=1280,
                         n_heads=20, n_kv_heads=1, d_ff=3584, wfmt="ternary",
                         kvfmt="q4", tokens_per_pass=4.0))
    return A

if __name__ == "__main__":
    print(f"Banda medida: {BANDWIDTH_GBS} GB/s | eficiencia decode: {DECODE_EFFICIENCY} "
          f"| limite RAM: {RAM_LIMIT_MB:.0f} MB\n")
    hdr = f"{'Arquitetura':<28}{'tot M':>7}{'ativ M':>8}{'RAM MB':>9}{'tok/s':>8}{'20k s':>9}  RAM VEL"
    print(hdr); print("-"*len(hdr))
    for cfg in build_architectures():
        r = evaluate(cfg)
        ram = "OK " if r["GR_RAM"] else "X  "
        vel = "OK" if r["GR_VEL"] else "X "
        print(f"{r['name']:<28}{r['total_M']:>7.0f}{r['active_M']:>8.0f}"
              f"{r['peak_ram_mb']:>9.0f}{r['tok_s']:>8.1f}{r['time_20k_s']:>9.1f}  {ram} {vel}")
    print("\nGR_RAM = cabe em 500 MB no pico | GR_VEL = 20k tokens em <60s")
    print("tok/s = TETO pela banda (ceiling), nao medicao de geracao real.")
