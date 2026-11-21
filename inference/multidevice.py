"""
Roda o modelo FINAL (500M, stack completo) em TODOS os dispositivos-alvo.
RAM e identica em todo lugar; a velocidade varia com a banda local.
Guardrails checados por aparelho. Auto-configura pela banda detectada.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.explorer import (ModelConfig, resident_ram_mb, BPP, KV_BYTES,
                                RAM_LIMIT_MB)
from inference.devices import DEVICES, effective_bandwidth

# Config final recomendada (arch 9): ~500M, ternario + MQA + KV Q4 + bloco
FINAL = ModelConfig("FINAL @500M", n_layers=28, d_model=1280, n_heads=20,
                    n_kv_heads=1, d_ff=3584, wfmt="ternary", kvfmt="q4",
                    tokens_per_pass=4.0)

def tok_s_on(cfg, eff_bw, avg_seq):
    bytes_pp = cfg.active_params()*BPP[cfg.wfmt] + cfg.kv_bytes_per_token()*avg_seq
    return cfg.tokens_per_pass * eff_bw / bytes_pp

def run(prompt=512, gen=20000):
    max_seq, avg_seq = prompt+gen, prompt+gen/2
    peak_ram = resident_ram_mb(FINAL, max_seq)   # igual em todo device
    print(f"Modelo: {FINAL.total_params()/1e6:.0f}M params | "
          f"RAM pico: {peak_ram:.0f} MB (limite {RAM_LIMIT_MB:.0f}) | saida {gen} tok\n")
    hdr = f"{'Dispositivo':<30}{'banda ef':>10}{'tok/s':>9}{'20k s':>9}  RAM VEL  fonte"
    print(hdr); print("-"*len(hdr))
    for d in DEVICES:
        name, bw_gbs, isa, ram_total, src = d
        tps = tok_s_on(FINAL, effective_bandwidth(d), avg_seq)
        t20k = gen / tps
        gr_ram = peak_ram <= RAM_LIMIT_MB and peak_ram <= ram_total
        gr_vel = t20k <= 60.0
        ram = "OK" if gr_ram else "X "
        vel = "OK" if gr_vel else "X "
        print(f"{name:<30}{bw_gbs:>8.1f}GB{tps:>9.0f}{t20k:>9.1f}  {ram} {vel}   {src}")
    print("\nRAM constante em todo aparelho; so a velocidade escala com a banda.")
    print("tok/s = TETO pela banda efetiva (ceiling), nao geracao medida.")
    print("Auto-config: no boot, hwdetect mede a banda real e ajusta threads/kernel.")

if __name__ == "__main__":
    run()
