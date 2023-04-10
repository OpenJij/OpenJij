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
from openjij.sampler.sqa_sampler import SQASampler


class GPUChimeraSQASampler(SQASampler, BaseGPUChimeraSampler):
    """Sampler with Simulated Quantum Annealing (SQA) on GPU.

    Inherits from :class:`openjij.sampler.sqa_sampler.SQASampler`.

    Args:
        beta (float): Inverse temperature.
        gamma (float): Amplitude of quantum fluctuation.
        trotter (int): Trotter number.
        num_sweeps (int): number of sweeps
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
        beta=10.0,
        gamma=1.0,
        trotter=4,
        num_sweeps=100,
        schedule=None,
        num_reads=1,
        unit_num_L=None,
    ):
        # GPU Sampler allows only even trotter number
        if trotter % 2 != 0:
            raise ValueError("GPU Sampler allows only even trotter number")
        self.trotter = trotter
        self.unit_num_L = unit_num_L

        super().__init__(
            beta=beta,
            gamma=gamma,
            trotter=trotter,
            num_reads=num_reads,
            num_sweeps=num_sweeps,
            schedule=schedule,
        )

        self._make_system = {
            "singlespinflip": cxxjij.system.make_chimera_transverse_gpu
        }
        self._algorithm = {"singlespinflip": cxxjij.algorithm.Algorithm_GPU_run}

    def _get_result(self, system, model):
        result = cxxjij.result.get_solution(system)
        sys_info = {}
        return result, sys_info

    def sample_ising(
        self,
        h,
        J,
        beta=None,
        gamma=None,
        num_sweeps=None,
        schedule=None,
        num_reads=None,
        unit_num_L=None,
        initial_state=None,
        updater=None,
        reinitialize_state=True,
        seed=None,
    ):
        """Sampling from the Ising model.

        Args:
            h (dict): Linear term of the target Ising model.
            J (dict): Quadratic term of the target Ising model.
            beta (float, optional): inverse tempareture.
            gamma (float, optional): strangth of transverse field. Defaults to None.
            num_sweeps (int, optional): number of sweeps. Defaults to None.
            schedule (list[list[float, int]], optional): List of annealing parameter. Defaults to None.
            num_reads (int, optional): number of sampling. Defaults to 1.
            initial_state (list[int], optional): Initial state. Defaults to None.
            updater (str, optional): update method. Defaults to 'single spin flip'.
            reinitialize_state (bool, optional): Re-initilization at each sampling. Defaults to True.
            seed (int, optional): Sampling seed. Defaults to None.

        Returns:
            :class:`openjij.sampler.response.Response`: results

        Examples::

            >>> sampler = openjij.sampler.chimera_gpu.gpu_sqa_sampler.GPUChimeraSQASampler(unit_num_L=2)
            >>> h = {0: -1, 1: -1, 2: 1, 3: 1},
            >>> J = {(0, 4): -1, (2, 5): -1}
            >>> res = sampler.sample_ising(h, J)
        """

        # Set default updater
        if updater is None:
            updater = "single spin flip"

        if num_reads is None:
            num_reads = 1

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
            beta=beta,
            gamma=gamma,
            num_sweeps=num_sweeps,
            schedule=schedule,
            num_reads=num_reads,
            initial_state=initial_state,
            updater=updater,
            reinitialize_state=reinitialize_state,
            seed=seed,
            structure=structure,
        )
