# Benchmark

``solver_benchmark`` provides benchmarking features.


## Example: anti-ferromagnetic model

make instance
```python
import random
import numpy as np
import matplotlib.pyplot as plt
import openjij as oj

# 1-D Anti-ferromagnetic model
N = 30
h = {0: -10}
J = {(i, i+1): 1 for i in range(N-1)}
```

call ``solver_benchmark``
```python
# solution
correct_state = [(-1)**i for i in range(N)]

# number of steps and iteration for benchmark
step_num_list = list(range(10, 101, 10))  # [10, 20,,,, 100]
iteration = 100

# calculate average and standard error of time to solution (TTS), residual energy, success probability 
sampler = oj.SASampler(num_reads=iteration)
result = oj.solver_benchmark(
                      solver=lambda time, **args: sampler.sample_ising(h,J,num_sweeps=time),
                      time_list=step_num_list, solutions=[correct_state], p_r=0.99
            )
```

## Benchmark D-Wave sampler

If you want to take benchmark of D-Wave Sampler,
you can convert response to openjij.Response from dimod.SampleSet

```python
sampler = DWaveSampler()
solver = lambda time, **args: oj.convert_response(sampler.sample_ising(h, J, num_sweeps=time))
result = oj.solver_benchmark(
                      solver=solver),
                      time_list=step_num_list, solutions=[correct_state], p_r=0.99
            )
```