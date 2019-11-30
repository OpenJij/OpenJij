# Higher order model

If you want to handle higher order model as follows:

$$H = \sum_{i}h_i\sigma_i + \sum{i < j} J_{ij} \sigma_i \sigma_j + \sum{i, j, k} K_{i,j,k} \sigma_i\sigma_j \sigma_k \cdots$$

use ``.sample_hubo``

> HUBO: Higher order unconstraint binary optimization

## Sample code
```python
import openjij as oj

# Only SASampler can handle HUBO.
sampler = oj.SASampler()

# make HUBO
h = {0: -1}
J = {(0, 1): -1}
K = {(0, 1, 2): 1}

response = sampler.sample_hubo([h, J, K], var_type="SPIN")
response.states[0]
> [1, 1, -1]
```

## Note

``.sample_hubo``
- The first argument (``interactions``) must be in ascending order.
- Need ``var_type``