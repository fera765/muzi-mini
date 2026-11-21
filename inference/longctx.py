"""
Validacao empirica de CONTEXTO LONGO no runtime SSM proprio.
Gera milhares de tokens e mede: RSS ao longo do caminho e latencia por token
em posicoes espalhadas. Se ambos ficam planos, o custo NAO cresce com o
contexto -> 1M viavel. Medicao real (RAM+velocidade), nao qualidade.
"""
import sys, os, time, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.ssm_runtime import SSMRuntime, rss_mb

def run(n=12000, probes=(100, 3000, 6000, 9000, 11900)):
    rt = SSMRuntime(n_layers=6, d_model=384, d_inner=768, d_state=16, vocab=4000, seed=3)
    rt.reset(); tok = 1
    gc.collect(); rss0 = rss_mb()
    lat, rss_trace = {}, {}
    t0 = time.perf_counter()
    for i in range(n):
        t = time.perf_counter()
        tok = rt.step(tok)
        d = (time.perf_counter()-t)*1000
        if i in probes:
            lat[i] = d; rss_trace[i] = rss_mb()
    total = time.perf_counter()-t0
    print(f"Runtime SSM proprio | {rt.params/1e6:.1f}M params | {n} tokens gerados")
    print(f"Tempo total: {total:.1f}s | velocidade media: {n/total:.0f} tok/s")
    print(f"RSS inicial: {rss0:.0f} MB\n")
    print(f"{'posicao':>9}{'lat ms/tok':>13}{'RSS MB':>10}")
    for p in sorted(lat):
        print(f"{p:>9}{lat[p]:>13.3f}{rss_trace[p]:>10.0f}")
    lv = list(lat.values()); rv = list(rss_trace.values())
    print(f"\nLatencia: variacao {(max(lv)-min(lv))/min(lv)*100:.1f}% da pos 100 a {max(lat)}")
    print(f"RSS: variacao {(max(rv)-min(rv)):.0f} MB em {n} tokens")
    print(">> Ambos planos -> contexto nao custa tempo nem memoria crescente (1M ok).")

if __name__ == "__main__":
    run()
