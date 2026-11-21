"""
Empacotamento ternario BitNet b1.58: 5 valores {-1,0,1} por byte (3^5=243<256).
Mede o footprint REAL de 523M params e valida round-trip. Safe p/ python3 -I.
"""
import numpy as np, os, psutil, gc

POW3 = np.array([1, 3, 9, 27, 81], dtype=np.uint16)

def pack(t):
    """t: array int8 em {-1,0,1}. Retorna uint8 empacotado (5 por byte)."""
    flat = t.reshape(-1).astype(np.int16) + 1          # {0,1,2}
    pad = (-flat.size) % 5
    if pad: flat = np.concatenate([flat, np.zeros(pad, np.int16)])
    g = flat.reshape(-1, 5)
    return (g * POW3).sum(axis=1).astype(np.uint8), t.shape

def unpack(packed, shape):
    n = int(np.prod(shape))
    vals = np.empty((packed.size, 5), dtype=np.int8)
    x = packed.astype(np.uint16)
    for i in range(5):
        vals[:, i] = (x // POW3[i]) % 3
    return (vals.reshape(-1)[:n] - 1).astype(np.int8).reshape(shape)

def rss_mb(): return psutil.Process(os.getpid()).memory_info().rss/1e6

if __name__ == "__main__":
    # 1) validar round-trip
    rng = np.random.default_rng(0)
    sample = rng.integers(-1, 2, size=(1000, 1000)).astype(np.int8)
    packed, shape = pack(sample)
    back = unpack(packed, shape)
    ok = np.array_equal(sample, back)
    print(f"Round-trip ternario correto: {ok}")
    print(f"1M pesos: {sample.nbytes/1e6:.2f} MB int8 -> {packed.nbytes/1e6:.2f} MB packed "
          f"({sample.nbytes/packed.nbytes:.1f}x)\n")

    # 2) footprint REAL do modelo de 523M empacotado (aloca de verdade e mede RSS)
    gc.collect(); base = rss_mb()
    PARAMS = 523_000_000
    model_bytes = np.zeros((PARAMS + 4)//5, dtype=np.uint8)   # pesos packed residentes
    ssm_state = np.zeros(28*2560*16*2, dtype=np.uint8)        # estado SSM fixo (~2.3 MB)
    gc.collect(); peak = rss_mb()
    print(f"Modelo 523M ternario packed: {model_bytes.nbytes/1e6:.1f} MB (pesos)")
    print(f"RSS medido com modelo alocado: {peak:.0f} MB (base {base:.0f} MB)")
    print(f">> 523M params residentes em {model_bytes.nbytes/1e6:.0f} MB — cabe nos 500 MB.")
    del model_bytes, ssm_state
