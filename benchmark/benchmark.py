# openjij-dense, openjij-sparse, neal
# でのベンチマーク用のスクリプト
# sparse は遅すぎるのでベンチマーク測定できない

import openjij as oj
import neal
import numpy as np
import time

n = 1000
h, J = {}, {}

for i in range(n-1):
    for j in range(i+1, n):
        if np.random.random() <= 0.3:
            J[i, j] = np.random.uniform(-1, 1)

NUM_READS = 10

BETA_MAX = 14000000
BETA_MIN = 0.0015

# Benchmark OpenJij Dense
print('OpenJij Dense')
sampler = oj.SASampler()
steps_openjij = [1000, 3000, 6000, 12000]
openjij_time = []
openjij_energy = []
for step in steps_openjij:
    print('num_sweeps:', step)
    start = time.time()
    response = sampler.sample_ising(h, J, num_sweeps=step, num_reads=NUM_READS, beta_max=BETA_MAX, beta_min=BETA_MIN)
    elapsed_time = time.time() - start

    openjij_time.append(elapsed_time)
    openjij_energy.append(np.mean(response.energies))

    print("\telapsed_time:{0}".format(elapsed_time) + "[sec]")

# Benchmark OpenJij Sparse
print('OpenJij Sparse')
sampler = oj.SASampler()
openjij_sparse_time = []
openjij_sparse_energy = []
for step in steps_openjij:
    print('num_sweeps:', step)
    start = time.time()
    response = sampler.sample_ising(h, J, num_sweeps=step, num_reads=NUM_READS, beta_max=BETA_MAX, beta_min=BETA_MIN, sparse=True)
    elapsed_time = time.time() - start

    openjij_sparse_time.append(elapsed_time)
    openjij_sparse_energy.append(np.mean(response.energies))

    print("\telapsed_time:{0}".format(elapsed_time) + "[sec]")

# Benchmark OpenJij SQA
print('OpenJij SQA Dense')
sampler = oj.SQASampler(beta=10, trotter=8)
openjij_sql_time = []
openjij_sql_energy = []
for step in steps_openjij:
    print('num_sweeps:', step)
    start = time.time()
    response = sampler.sample_ising(h, J, num_sweeps=step, num_reads=NUM_READS)
    elapsed_time = time.time() - start

    openjij_sql_time.append(elapsed_time)
    openjij_sql_energy.append(np.mean(response.energies))

    print("\telapsed_time:{0}".format(elapsed_time) + "[sec]")

print('OpenJij SQA Sparse')
sampler = oj.SQASampler(beta=10, trotter=8)
openjij_sqs_time = []
openjij_sqs_energy = []
for step in steps_openjij:
    print('num_sweeps:', step)
    start = time.time()
    response = sampler.sample_ising(h, J, num_sweeps=step, num_reads=NUM_READS, sparse=True)
    elapsed_time = time.time() - start

    openjij_sqs_time.append(elapsed_time)
    openjij_sqs_energy.append(np.mean(response.energies))

    print("\telapsed_time:{0}".format(elapsed_time) + "[sec]")

# Benchmark neal
print('dwave-neal')
sampler = neal.SimulatedAnnealingSampler()
steps_openjij = [1000, 3000, 6000, 12000]
neal_time = []
neal_energy = []
for step in steps_openjij:
    print('num_sweeps:', step)
    start = time.time()
    response = sampler.sample_ising(h, J, num_sweeps=step, num_reads=NUM_READS, beta_range=(BETA_MIN, BETA_MAX))
    elapsed_time = time.time() - start

    neal_time.append(elapsed_time)
    neal_energy.append(np.mean([r[1] for r in response.record]))

    print("\telapsed_time:{0}".format(elapsed_time) + "[sec]")


import matplotlib.pyplot as plt
plt.plot(openjij_time, openjij_energy, '-o', label='openjij-dense')
plt.plot(openjij_sparse_time, openjij_sparse_energy, '-o', label='openjij-sparse')
plt.plot(openjij_sql_time, openjij_sql_energy, '-o', label='openjij-sqa-dense')
plt.plot(openjij_sqs_time, openjij_sqs_energy, '-o', label='openjij-sqa-sparse')
plt.plot(neal_time, neal_energy, '-o', label='neal')
plt.xlabel('time [sec]')
plt.ylabel('Energy')
plt.legend()
plt.show()
