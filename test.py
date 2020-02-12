import openjij as oj
import cxxjij as cj
import numpy as np


h, J = {}, {}
# n = 10
# for i in range(n-1):
#     for j in range(i+1, n):
#         J[i, j] = np.random.uniform(0, 1)

J['x[0][0]', 'x[1][0]'] = 1
J['x[0][1]', 'x[0][2]'] = -1

h['x[0][0]'] = -1
h['x[0][2]'] = 2

bqm = oj.BinaryQuadraticModel(h, J)

spin_config = [[(0.1, 1), (0.6, -1)], [(0.1, 1), (0.6, -1)],
               [(0.1, 1), (0.6, -1)], [(0.1, 1), (0.6, -1)]]

# ising_graph = bqm.get_cxxjij_ising_graph()
# print(bqm.indices)
# sqa_system = cj.system.ContinuousTimeIsing_Dense(spin_config, ising_graph, 1.0)
# schedule = cj.utility.make_transverse_field_schedule_list(
#     beta=10.0, one_mc_step=1, num_call_updater=100
# )
# cj.algorithm.Algorithm_ContinuousTimeSwendsenWang_run(sqa_system, schedule)
# state = cj.result.get_solution(sqa_system)
# print(sqa_system.spin_config)
# print(state)

print(dir(oj.sampler))
sampler = oj.sampler.csqa_sampler.CSQASampler()
res = sampler.sample_qubo(J, num_reads=10)
print(res)
