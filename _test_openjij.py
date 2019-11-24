import openjij as oj
import numpy as np


sampler = oj.SASampler()
h = {0: -1}
J = {(0, 1): -1}
res = sampler.sample_ising(h, J)
print(res.info)
