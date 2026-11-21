"""
Auto-deteccao de hardware — portavel x86 (AVX/AVX-512) e ARM (NEON/SVE).
Roda no boot do runtime e escolhe banda/ISA/threads reais do aparelho.
Mede a banda localmente (nao confia em specs de catalogo).
Safe para `python3 -I`.
"""
import os, time, platform, re
import numpy as np

X86_KERNEL_FLAGS = ["avx512f", "avx512_vnni", "avx512_bf16", "amx_int8",
                    "avx2", "avx_vnni", "fma", "f16c", "sse4_2"]
ARM_KERNEL_FLAGS = ["sve2", "sve", "i8mm", "bf16", "dotprod", "asimd", "neon"]

def read_cpuinfo():
    try:
        with open("/proc/cpuinfo") as f:
            return f.read().lower()
    except OSError:
        return ""

def detect_isa():
    info = read_cpuinfo()
    arch = platform.machine().lower()
    if arch in ("x86_64", "amd64", "i686"):
        flags = set(re.findall(r"[a-z0-9_]+", info))
        present = [f for f in X86_KERNEL_FLAGS if f in flags]
        best = ("amx_int8" if "amx_int8" in flags else
                "avx512_vnni" if "avx512_vnni" in flags else
                "avx512f" if "avx512f" in flags else
                "avx2" if "avx2" in flags else "sse4_2")
        return {"arch": "x86", "best_kernel": best, "present": present}
    if arch.startswith(("aarch64", "arm")):
        # ARM expoe recursos via 'Features' em alguns kernels
        feats = set(re.findall(r"[a-z0-9_]+", info))
        present = [f for f in ARM_KERNEL_FLAGS if f in feats]
        best = ("sve2" if "sve2" in feats else "sve" if "sve" in feats else
                "i8mm" if "i8mm" in feats else "dotprod" if "dotprod" in feats
                else "neon")
        return {"arch": "arm", "best_kernel": best, "present": present or ["neon"]}
    return {"arch": arch, "best_kernel": "scalar", "present": []}

def detect_cores():
    try:
        import psutil
        logical = psutil.cpu_count(logical=True)
        physical = psutil.cpu_count(logical=False) or logical
    except Exception:
        logical = os.cpu_count() or 1
        physical = logical
    return {"physical": physical, "logical": logical}

def measure_bandwidth_gbs(mb=256, reps=20):
    n = (mb * 1024 * 1024) // 8
    a = np.ones(n, dtype=np.float64); c = np.empty(n, dtype=np.float64)
    np.copyto(c, a)
    t = time.perf_counter()
    for _ in range(reps):
        np.copyto(c, a)
    dt = (time.perf_counter() - t) / reps
    return (2 * a.nbytes) / dt / 1e9

def recommend_threads(cores):
    # decode escala com nucleos de alto desempenho; evita oversubscription
    return max(1, cores["physical"])

def profile():
    isa = detect_isa()
    cores = detect_cores()
    bw = measure_bandwidth_gbs()
    # eficiencia de decode por familia de ISA (realista, nao teorica)
    eff = {"amx_int8": 0.70, "avx512_vnni": 0.68, "avx512f": 0.65, "avx2": 0.62,
           "sve2": 0.58, "sve": 0.55, "i8mm": 0.52, "dotprod": 0.50,
           "neon": 0.45, "sse4_2": 0.50, "scalar": 0.30}.get(isa["best_kernel"], 0.45)
    return {
        "machine": platform.machine(),
        "system": platform.system(),
        "isa": isa,
        "cores": cores,
        "bandwidth_gbs": round(bw, 1),
        "decode_efficiency": eff,
        "threads": recommend_threads(cores),
    }

if __name__ == "__main__":
    import json
    print(json.dumps(profile(), indent=2, ensure_ascii=False))
