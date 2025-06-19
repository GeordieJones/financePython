import multiprocessing as mp
import numpy as np
from time import time
import matplotlib.pyplot as plt

def simulate_geo_brownian_motion(p):
    M, I = p
    S0 = 100; r = 0.05; sigma = 0.2; T = 1.0
    dt = T / M
    paths = np.zeros((M + 1, I))
    paths[0] = S0
    for t in range(1, M + 1):
        paths[t] = paths[t - 1] * np.exp((r - 0.5 * sigma**2) * dt +
                                         sigma * np.sqrt(dt) * np.random.standard_normal(I))
    return paths

def run_multiprocessing():
    M, I, t = 10000, 100, 100
    timings = []
    for w in range(1, 17):
        t0 = time()
        with mp.Pool(processes=w) as pool:
            result = pool.map(simulate_geo_brownian_motion, [(M, I)] * t)
        elapsed = time() - t0
        timings.append((w, elapsed))
        print(f"Workers: {w}, Time: {elapsed:.2f} seconds")
    return timings

if __name__ == "__main__":
    timings = run_multiprocessing()
    workers, times = zip(*timings)

    plt.figure(figsize=(8, 5))
    plt.plot(workers, times, label="Execution Time")
    plt.plot(workers, times, 'ro')  # Red dots
    plt.grid(True)
    plt.xlabel('Number of Processes')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time vs Number of Processes in Monte Carlo Simulation')
    plt.show()

'''
Output on my Mac:
Workers: 1, Time: 10.33 seconds
Workers: 2, Time: 6.59 seconds
Workers: 3, Time: 5.17 seconds
Workers: 4, Time: 4.94 seconds
Workers: 5, Time: 4.71 seconds
Workers: 6, Time: 4.60 seconds
Workers: 7, Time: 4.80 seconds
Workers: 8, Time: 4.90 seconds
Workers: 9, Time: 5.16 seconds
Workers: 10, Time: 5.16 seconds
Workers: 11, Time: 5.16 seconds
Workers: 12, Time: 5.39 seconds
Workers: 13, Time: 5.06 seconds
Workers: 14, Time: 5.24 seconds
Workers: 15, Time: 5.57 seconds
Workers: 16, Time: 5.74 seconds
'''