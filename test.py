import openjij as oj
import numpy as np


h, J = {}, {}
# n = 10
# for i in range(n-1):
#     for j in range(i+1, n):
#         J[i, j] = np.random.uniform(0, 1)

J['x[0][0]', 'x[1][0]'] = 0
J['x[0][1]', 'x[0][2]'] = -1

h['x[0][0]'] = -1
h['x[0][2]'] = 2

sampler = oj.SQASampler()
res = sampler.sample_qubo(J, num_reads=10)
print(res)
