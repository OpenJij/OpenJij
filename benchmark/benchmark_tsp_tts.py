import openjij as oj
import neal
import dimod
import numpy as np
import itertools
import time
import matplotlib.pyplot as plt


tsp_instance = {
    5: {"x": [0.1, 0.2, 0.4, 0.7, 0.2], "y": [0.9, 0.3, 0.2, 0.4, 0.6], 'known_lowest': 2.3634102},
    10: {"x": [0.335, 0.947, 0.868, 0.108, 0.636, 0.368, 0.844, 0.467, 0.444, 0.112],
         "y": [0.405, 0.728, 0.798, 0.696, 0.579, 0.938, 0.522, 0.637, 0.496, 0.205], 'known_lowest': 3.2472424},
    30: {"x": [0.446, 0.440, 0.272, 0.870, 0.288, 0.496, 0.608, 0.156, 0.589, 0.754, 0.654, 0.879, 0.581, 0.461, 0.318, 0.939, 0.101, 0.699, 0.730, 0.543, 0.0212, 0.345, 0.365, 0.0798, 0.636, 0.595 , 0.985 , 0.248, 0.207, 0.694],
         "y": [0.129, 0.895, 0.136, 0.762, 0.856, 0.687, 0.266, 0.653, 0.098, 0.964, 0.256, 0.649, 0.209, 0.947, 0.386, 0.939, 0.989, 0.443, 0.500, 0.968, 0.03551971, 0.59497137, 0.15832582, 0.73873134, 0.52305714, 0.23057308, 0.65426788, 0.41555848, 0.44112897, 0.26007979],
         "known_lowest": 9.994}
}

N = 30
city = tsp_instance[N]

XX, XX_T = np.meshgrid(city['x'], city['x'])
YY, YY_T = np.meshgrid(city['y'], city['y'])
distance = np.sqrt((XX - XX_T)**2 + (YY - YY_T)**2)

d_max = np.max(distance)
lam_time, lam_city = 2*d_max, 2*d_max
qubo = {}
# Objective function sum_{i, j} sum_t d_ij x_it x_{j, (t+1) mod N}
for i, j in itertools.combinations(range(N), 2):
    for t in range(N):
        qubo[(i, t), (j, (t + 1) % N)] = distance[i, j]
        qubo[(j, t), (i, (t + 1) % N)] = distance[j, i]

# One-hot constraint sum_t(sum_i x_it - 1)^2
for t in range(N):
    for i in range(N):
        for j in range(N):
            qubo[(i, t), (j, t)] = 2*lam_city
        qubo[(i, t), (i, t)] = -lam_city

# One-hot constraint sum_i(sum_t x_it - 1)^2
for i in range(N):
    for t in range(N):
        for k in range(N):
            if ((i, t), (i, k)) in qubo:
                qubo[(i, t), (i, k)] += 2*lam_city
            else:
                qubo[(i, t), (i, k)] = 2*lam_city
        qubo[(i, t), (i, t)] = -lam_city



def convert_to_tours(response):
    vars = response.variables
    tours = []
    for r in response.record:
        binaries = r[0]
        ones = np.where(binaries == 1)
        one_it = np.array(vars)[ones].T
        tour = np.zeros(N, dtype=int)
        tour[one_it[1]] = one_it[0]
        tour = np.append(tour, [tour[0]])
        tours.append(tour)
    return tours

def is_feasible_tour(tour, N):
    _t = tour[:-1]
    if len(_t) != N:
        return False
    if len(np.unique(_t)) != N:
        return False
    return True

def tsp_objective(tour, distance):
    n = len(distance)
    obj = 0
    for t, i in enumerate(tour):
        obj += distance[i, tour[(t+1) % n]]
    return obj


def default_benchmark(sampler, num_reads: int, threshold_rate: float, **kwargs):
    response = sampler.sample_qubo(qubo, num_reads=1, **kwargs)
    sampling_time = []
    for _ in range(num_reads-1):
        start = time.time()
        _res = sampler.sample_qubo(qubo, num_reads=1, **kwargs)
        sampling_time.append(time.time() - start)
        response = dimod.concatenate([response, _res])

    tours = convert_to_tours(response)
    feas_num = 0
    success_num = 0
    objective = []
    for tour in tours:
        if is_feasible_tour(tour, N):
            feas_num += 1
            objective.append(tsp_objective(tour, distance))
            if objective[-1] <= city['known_lowest'] * threshold_rate:
                success_num += 1
    p_success = success_num / num_reads
    p_r = 0.99
    tts = np.mean(sampling_time) * np.log(1-p_r)/np.log(1-p_success)

    print("  Benchmark Results:")
    print("   # of feasible:     ", f"{feas_num}/{num_reads}")
    print("   # of success:      ", f"{success_num}/{num_reads}")
    print("   TTS[sec]:          ", tts)
    print("   sampling time[sec]:", np.mean(sampling_time))
    print("   min obj:           ", np.min(objective))


num_reads = 10
threshold_rate = 1.2
sampler = neal.SimulatedAnnealingSampler()
print("neal Results -------")
default_benchmark(sampler, num_reads, threshold_rate)

sampler = oj.SASampler()
print("OpenJij (sparse) Results -------")
default_benchmark(sampler, num_reads, threshold_rate, sparse=True)

print("OpenJij (Dense) Results -------")
default_benchmark(sampler, num_reads, threshold_rate, sparse=None)
