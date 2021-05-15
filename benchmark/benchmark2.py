# openjij の名前空間ではなく 
# cxxjij の名前空間から直接呼ぶベンチマーク
# (OpenJijと矛盾がないかチェックするために)


import cxxjij
import openjij as oj
import numpy as np
import time

N= 1000
h, J = {}, {}

for i in range(N-1):
    for j in range(i+1, N):
        J[i, j] = -1 # np.random.uniform(-1, 1)

BETA_MAX = 14000000
BETA_MIN = 0.0015
ONE_MCSTEP = 1
NUM_CALL_UPDATER = 1000


bqm = oj.BinaryQuadraticModel(h, J, vartype='SPIN')

graph = bqm.get_cxxjij_ising_graph(sparse=False)

init_spin = np.random.choice([-1, 1], N)
system = cxxjij.system.make_classical_ising(init_spin.tolist(), graph)
algorithm = cxxjij.algorithm.Algorithm_SingleSpinFlip_run
schedule = cxxjij.utility.make_classical_schedule_list(
    beta_min=BETA_MIN, beta_max=BETA_MAX,
    one_mc_step=ONE_MCSTEP,
    num_call_updater=NUM_CALL_UPDATER
)

start = time.time()
algorithm(system, schedule)
elapsed_time = time.time() - start
print("\telapsed_time:{0}".format(elapsed_time) + "[sec]")