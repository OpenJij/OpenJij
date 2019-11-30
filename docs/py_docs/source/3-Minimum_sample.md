# Minimum sample code

## Simulated annealing (SA)

### for the Ising model

to get a sample that executed SA 100 times
```python
import openjij as oj

n = 10
h, J = {}, {}
for i in range(n-1):
    for j in range(i+1, n):
        J[i, j] = -1

sampler = oj.SASampler()
response = sampler.sample_ising(h, J, num_reads=100)

# minimum energy state
response.min_samples['states']
# array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
#       [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1]])

# indices (labels) of state (spins)
response.indices
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```