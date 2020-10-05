# for antiferromagnetic one-dimensional Ising model
import openjij as oj
import numpy as np

N = 30
afih = {0: -10}
afiJ = {(i, i+1): 1+(0.1*np.sqrt(i)) for i in range(N-1)}
afiground = {i:(-1)**i for i in range(N)}
init_spin = [1]*N


sampler = oj.SASampler(num_sweeps=51, num_reads=100)
res = sampler.sample_ising(afih, afiJ, initial_state=init_spin, seed=5)
print(res)
