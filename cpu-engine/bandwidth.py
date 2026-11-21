"""Benchmark STREAM-like de banda de memoria real (numpy), -I safe."""
import time, numpy as np

def measure_bandwidth(mb=256, reps=30):
    n = (mb * 1024 * 1024) // 8          # float64 elements
    a = np.ones(n, dtype=np.float64)
    b = np.ones(n, dtype=np.float64)
    c = np.empty(n, dtype=np.float64)
    bytes_moved_copy = 2 * a.nbytes       # read a + write c
    bytes_moved_triad = 3 * a.nbytes       # read a,b + write c
    # warm
    np.copyto(c, a)
    # COPY: c = a
    t = time.perf_counter()
    for _ in range(reps):
        np.copyto(c, a)
    tcopy = (time.perf_counter() - t) / reps
    # TRIAD: c = a + 3*b
    t = time.perf_counter()
    for _ in range(reps):
        np.add(a, b, out=c); np.multiply(b, 3.0, out=c); np.add(a, c, out=c)
    ttriad = (time.perf_counter() - t) / reps
    copy_gbs  = bytes_moved_copy  / tcopy  / 1e9
    triad_gbs = bytes_moved_triad / ttriad / 1e9
    return copy_gbs, triad_gbs

if __name__ == "__main__":
    for size in (64, 256, 512):
        copy_gbs, triad_gbs = measure_bandwidth(size)
        print(f"[{size:>4} MB] COPY={copy_gbs:6.1f} GB/s   TRIAD={triad_gbs:6.1f} GB/s")
