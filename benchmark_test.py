import openjij as oj
import numpy as np


n = 2000
qubo_matrix = np.random.uniform(-1, 1, (n, n))
qubo_matrix = qubo_matrix + qubo_matrix.T
__row, __col = np.where(qubo_matrix != 0.0)
qubo_value = qubo_matrix.flatten()
qubo_row = __row
qubo_col = __col

qubo = {}
for r, c, v in zip(qubo_row, qubo_col, qubo_value):
    qubo[(int(r), int(c))] = v

num_sweeps = 500

sampler = oj.SASampler()
response = sampler.sample_qubo(qubo, num_sweeps=num_sweeps)

beta_scheudle = []
for _beta_obj in sampler._params["schedule"]:
    text = str(_beta_obj)
    beta_str = text.split(':')[1].split(')')[0].strip()
    beta = float(beta_str)
    beta_scheudle.append(beta)

_row = qubo_row
_col = qubo_col
_value = qubo_value
# for (r, c), v in qubo.items():
#     if (c, r) in qubo:
#         _r, _c = sorted([r, c])
#         if (_r, _c) == (r, c):
#             _row.append(_r)
#             _col.append(_c)
#             _value.append(v + qubo[(_c, _r)])
#     else:
#         _row.append(r)
#         _col.append(c)
#         _value.append(v)

_row = np.array(_row, dtype=np.int32)
_col = np.array(_col, dtype=np.int32)
_value = np.array(_value, dtype=np.float64)
state = np.random.choice([0.0, 1.0], n)
print(response.info)
# beta_scheudle = beta_scheudle[:10]
import time
start = time.time()
oj.cxxjij.sa_qubo_ssf_coo(
    _row, _col, _value, state, np.array(beta_scheudle, dtype=np.float64)
)
print("time: ", time.time() - start)


qubo_matrix = np.zeros((n, n))
for (r, c), v in qubo.items():
    qubo_matrix[r, c] = v

result = np.zeros(n)
for i, v in list(response.samples())[0].items():
    result[i] = v

print("openjij energy: ", response.record.energy[0])
openjij_energy = result @ qubo_matrix @ result
print("openjij energy: ", openjij_energy)

sa_result = state @ qubo_matrix @ state
print("sa energy: ", sa_result)
