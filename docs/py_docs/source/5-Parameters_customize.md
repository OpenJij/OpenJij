# Parameters customize

## Customize annealing schedule


```python
import openjij as oj

n = 10
h, J = {}, {}
for i in range(n-1):
    for j in range(i+1, n):
        J[i, j] = -1

# customized annealing schedule
# list of [beta, monte carlo steps in beta]
schedule = [
    [10, 3],
    [ 5, 5],
    [0.5, 10]
]

sampler = SASampler()
response = sampler.sample_ising(h, J, schedule=schedule)
```