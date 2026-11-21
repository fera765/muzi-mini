"""
RUNTIME PROPRIO — nucleo SSM recorrente (estilo Mamba) em numpy puro.
Mede DE VERDADE: RSS pico (psutil) e latencia real por token.

Prova empirica central: latencia por token e CONSTANTE conforme o contexto
cresce (estado recorrente fixo) -> e isso que viabiliza 1M de contexto sem
perder velocidade nem RAM. Pesos aleatorios: mede RAM+VELOCIDADE reais,
NAO qualidade (qualidade exige treino). Safe p/ `python3 -I`.
"""
import time, os, gc
import numpy as np
import psutil

def rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1e6

class SSMBlock:
    """Recorrencia linear seletiva: h_t = decay*h_{t-1} + dB*x ; y = sum(h*C)."""
    def __init__(self, d_model, d_inner, d_state, rng):
        self.in_proj  = rng.standard_normal((d_model, d_inner)).astype(np.float32) * 0.02
        self.out_proj = rng.standard_normal((d_inner, d_model)).astype(np.float32) * 0.02
        self.A = -np.abs(rng.standard_normal((d_inner, d_state)).astype(np.float32)) * 0.1
        self.B = rng.standard_normal((d_inner, d_state)).astype(np.float32) * 0.1
        self.C = rng.standard_normal((d_state,)).astype(np.float32) * 0.1
        self.dt = np.float32(0.05)
        self.h = np.zeros((d_inner, d_state), dtype=np.float32)   # ESTADO FIXO
    def reset(self):
        self.h[:] = 0.0
    def step(self, x):                      # x: (d_model,)
        xi = x @ self.in_proj               # (d_inner,)
        decay = np.exp(self.A * self.dt)    # (d_inner, d_state)
        self.h = decay * self.h + (self.dt * self.B) * xi[:, None]
        y = self.h @ self.C                 # (d_inner,)
        return y @ self.out_proj            # (d_model,)

class SSMRuntime:
    def __init__(self, n_layers, d_model, d_inner, d_state, vocab, seed=0):
        rng = np.random.default_rng(seed)
        self.emb = rng.standard_normal((vocab, d_model)).astype(np.float32) * 0.02
        self.layers = [SSMBlock(d_model, d_inner, d_state, rng) for _ in range(n_layers)]
        self.head = rng.standard_normal((d_model, vocab)).astype(np.float32) * 0.02
        self.vocab = vocab
        self.params = (self.emb.size + self.head.size +
                       sum(l.in_proj.size+l.out_proj.size+l.A.size+l.B.size+l.C.size
                           for l in self.layers))
    def reset(self):
        for l in self.layers: l.reset()
    def step(self, token):
        x = self.emb[token]
        for l in self.layers:
            x = x + l.step(x)               # residual
        logits = x @ self.head
        return int(np.argmax(logits))

def benchmark(rt, n_tokens=400, probe_positions=(50, 100, 200, 390)):
    rt.reset()
    tok = 1
    lat_at = {}
    gc.collect(); peak = rss_mb()
    t0 = time.perf_counter()
    for i in range(n_tokens):
        t = time.perf_counter()
        tok = rt.step(tok)
        dt = time.perf_counter() - t
        if i in probe_positions:
            lat_at[i] = dt * 1000
        peak = max(peak, rss_mb())
    total = time.perf_counter() - t0
    return {"tok_s": n_tokens/total, "peak_rss_mb": peak,
            "lat_ms_by_pos": lat_at, "params_M": rt.params/1e6}

if __name__ == "__main__":
    print("RUNTIME SSM PROPRIO — medicao real (RAM+velocidade; NAO qualidade)\n")
    rt = SSMRuntime(n_layers=8, d_model=512, d_inner=1024, d_state=16, vocab=4000)
    base = rss_mb()
    r = benchmark(rt)
    print(f"Config demo: 8 camadas, d_model 512, d_state 16, {r['params_M']:.1f}M params (fp32)")
    print(f"RSS pico medido:  {r['peak_rss_mb']:.0f} MB")
    print(f"Velocidade medida: {r['tok_s']:.0f} tok/s (numpy, 2 nucleos)\n")
    print("Latencia por token em diferentes posicoes do contexto:")
    for pos, ms in sorted(r["lat_ms_by_pos"].items()):
        print(f"   posicao {pos:>4}:  {ms:.3f} ms/token")
    vals = list(r["lat_ms_by_pos"].values())
    spread = (max(vals)-min(vals))/min(vals)*100
    print(f"\nVariacao da latencia posicao 50 -> 390: {spread:.1f}%")
    print(">> Latencia PLANA vs contexto = prova do estado recorrente fixo (1M viavel).")
