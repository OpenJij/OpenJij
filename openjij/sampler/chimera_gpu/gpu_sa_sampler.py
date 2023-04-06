# Copyright 2023 Jij Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import openjij
import openjij as oj
import openjij.cxxjij as cxxjij

from openjij.sampler.chimera_gpu.base_gpu_chimera import BaseGPUChimeraSampler
from openjij.sampler.sa_sampler import SASampler


class GPUChimeraSASampler(SASampler, BaseGPUChimeraSampler):
    """Sampler with Simulated Annealing (SA) on GPU.

    Inherits from :class:`openjij.sampler.sampler.BaseSampler`.

    Args:
        beta_min (float): Minimum inverse temperature.
        beta_max (float): Maximum inverse temperature.
        num_sweeps (int): Length of Monte Carlo step.
        schedule_info (dict): Information about a annealing schedule.
        num_reads (int): Number of iterations.
        unit_num_L (int): Length of one side of two-dimensional lattice in which chimera unit cells are arranged.

    Raises:
        ValueError: If variables violate as below.
        - trotter number is odd.
        - no input "unit_num_L" to an argument or this constructor.
        - given problem graph is incompatible with chimera graph.

        AttributeError: If GPU doesn't work.
    """

    def __init__(
        self,
        beta_min=None,
        beta_max=None,
        num_sweeps=1000,
        schedule=None,
        num_reads=1,
        unit_num_L=None,
    ):

        super().__init__(
            beta_min=beta_min,
            beta_max=beta_max,
            num_reads=num_reads,
            num_sweeps=num_sweeps,
            schedule=schedule,
        )

        self.unit_num_L = unit_num_L

        self._make_system = {"singlespinflip": cxxjij.system.make_chimera_classical_gpu}
        self._algorithm = {"singlespinflip": cxxjij.algorithm.Algorithm_GPU_run}

    def sample_ising(
        self,
        h,
        J,
        beta_min=None,
        beta_max=None,
        num_sweeps=None,
        num_reads=1,
        schedule=None,
        initial_state=None,
        updater=None,
        reinitialize_state=True,
        seed=None,
        unit_num_L=None,
    ):
        """Sample with Ising model.

        Args:
            h (dict): linear biases
            J (dict): quadratic biases
            beta_min (float): minimal value of inverse temperature
            beta_max (float): maximum value of inverse temperature
            num_sweeps (int): number of sweeps
            num_reads (int): number of reads
            schedule (list): list of inverse temperature
            initial_state (dict): initial state
            updater(str): updater algorithm
            reinitialize_state (bool): if true reinitialize state for each run
            seed (int): seed for Monte Carlo algorithm
            unit_num_L (int): number of chimera units

        Returns:
            :class:`openjij.sampler.response.Response`: results

        Examples::

            >>> sampler = openjij.sampler.chimera_gpu.gpu_sa_sampler.GPUChimeraSASampler(unit_num_L=2)
            >>> h = {0: -1, 1: -1, 2: 1, 3: 1},
            >>> J = {(0, 4): -1, (2, 5): -1}
            >>> res = sampler.sample_ising(h, J)
        """

        if updater is None:
            updater = "single spin flip"

        self.unit_num_L = unit_num_L if unit_num_L else self.unit_num_L

        model = oj.model.chimera_model.ChimeraModel(
            linear=h, quadratic=J, vartype="SPIN", unit_num_L=self.unit_num_L, gpu=True
        )

        # define Chimera structure
        structure = {}
        structure["size"] = 8 * self.unit_num_L * self.unit_num_L
        structure["dict"] = {}
        if isinstance(model.indices[0], int):
            # identity dict
            for ind in model.indices:
                structure["dict"][ind] = ind
        elif isinstance(model.indices[0], tuple):
            # map chimera coordinate to index
            for ind in model.indices:
                structure["dict"][ind] = model.to_index(*ind, self.unit_num_L)

        return self._sampling(
            model,
            beta_min,
            beta_max,
            num_sweeps,
            num_reads,
            schedule,
            initial_state,
            updater,
            reinitialize_state,
            seed,
            structure,
        )
