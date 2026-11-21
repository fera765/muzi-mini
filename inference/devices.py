"""
Perfis representativos de dispositivos-alvo. A banda e EFETIVA de decode
(estimada por specs de catalogo + eficiencia de ISA), exceto o host, que
e MEDIDO por hwdetect. Marcado 'medido' vs 'estimado' para honestidade.
"""
# bandwidth_gbs = banda efetiva de leitura; eff ja embutida nela aqui p/ device fixo
DEVICES = [
    # nome, banda_efetiva_GBs, isa, ram_total_MB, fonte
    ("Desktop DDR5 8c (AVX-512)", 55.0, "avx512_vnni", 16000, "estimado"),
    ("Notebook DDR4 (AVX2)",      22.0, "avx2",          8000, "estimado"),
    ("Host Xeon 2c (este)",       13.6, "avx512_vnni",   8000, "MEDIDO*"),
    ("Moto G60 (SD732G, NEON)",    7.5, "neon",          6000, "estimado"),
    ("Raspberry Pi 4 (NEON)",      3.5, "neon",          4000, "estimado"),
]
# *Host: banda bruta medida 21 GB/s x eficiencia decode 0.65 = 13.6 GB/s efetiva.
# Os demais ja estao em GB/s EFETIVA (banda bruta de catalogo x eff da ISA).

def effective_bandwidth(device):
    return device[1] * 1e9  # ja efetiva
