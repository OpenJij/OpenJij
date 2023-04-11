# Copyright (c) 2023, Jij Inc.
# All rights reserved.

# %%
import math
import numpy as np
import jijbench as jb
import openjij as oj
# %%
!pip show openjij
# %%
sampler = oj.SASampler()

# HUBO Instance
number_variables = 100
order = 8
sparse_rate = [1, 0.9, 0.5, 0.1, 0.1, 0.5, 0., 1] 
hubo = {}
for _order in range(1, order+1):
    fully_connected_edges = math.comb(number_variables, _order)
    number_of_edges = int(fully_connected_edges * sparse_rate[_order-1])
    keys = np.random.choice(number_variables, size=(number_of_edges, _order)).tolist()
    key_set = set(map(tuple, keys))
    values = np.random.uniform(-1, 1, size=len(key_set))
    hubo.update(dict(zip(key_set, values)))


# %%
