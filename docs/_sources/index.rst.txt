=========================
OpenJij -- A unified framework for QUBO.
=========================

`OpenJij <https://github.com/OpenJij/OpenJij>`_ is a unified annealing platform for research and development.

Minimum tutorial
----------------

Let's solve QUBO

.. math::
    E = - 2q_0 q_1 - 2q_1 q_2 + 1q_0q_2 + q_0 = \sum_{i \leq j} Q_{ij} q_i q_j

which

.. math::
    Q_{01} = -2,~ Q_{12} = -2,~ Q_{02} = 1,~ Q_{00} = 1.

We represent this QUBO matrix using a python dictionary.

.. code-block::

    Q = {(0,1): -1, (1,2): -2, (0,2): 1, (0,0): 2}

Next, we choose a sampler to solve Q.
Then use the sampler's `sample_qubo` method to perform sampling and get a solution.

.. code-block::

    >>> import openjij as oj
    >>> sampler = oj.SASampler()
    >>> response = sampler.sample_qubo(Q)

    >>> response
    ... iteration : 1, minimum energy : -2.0, var_type : BINARY
    ... indices: [0, 1, 2] 
    ... minmum energy state sample : [0, 1, 1]

Here we chose a sampler to perform simulated annealing. We can adjust the parameters of SA when instantiation of SASampler.
Let's change the SA temperature schedule and sample multiple times.
The inverse temperature schedule can be controlled by the ``schedule_info`` argument. If we use simple geometric cooling, we can specify only ``beta_min`` and ``beta_max``.
Let's do annealing at high temperature (ie reverse temperature decreases).
We can specify how many times to sample with the ``iteration`` argument.

.. code-block::

    >>> import openjij as oj
    >>> beta_sch = [(0.1, 10),(0.5,10)]
    >>> sampler = oj.SASampler(schedule=beta_sch, iteration=100)
    >>> response = sampler.sample_qubo(Q)

    >>> response
    ... iteration : 100, minimum energy : -2.0, var_type : BINARY
    ... indices: [0, 1, 2] 
    ... minmum energy state sample : [0, 1, 1]

    >>> response.min_samples
    ... {'min_states': array([[0, 1, 1]]), 'num_occurrences': array([39]), 'min_energy': -2.0}


``beta_sch = [(0.1, 10),(0.5,10)]`` represents the inverse temperature and the number of MCMC sampling at that temperature as a tuple.  
With this temperature schedule, the maximum reverse temperature is 0.5, so annealing is finished at a high temperature. 
Therefore, as you can see in ``response.min_samples``, the minimum energy solution is only obtained 39 times out of 100 times.

The feature of OpenJij is that you can easily switch the algorithm and hardware by changing the `sampler`.
For example, ``SQASampler`` is provided for simulating transverse field quantum annealing on a CPU. There are other hardware options available, so please try it.

.. toctree::
   :maxdepth: 2
   :caption: Contents:




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
