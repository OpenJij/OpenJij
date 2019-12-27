
import cxxjij
from openjij.model import BinaryQuadraticModel
from openjij.model import ChimeraModel
import dimod


class GPUBaseSampler():

    def _create_chimera_matrix(self, model, **kwargs):
        # convert to ChimeraModel from normal BQM
        if isinstance(model, dimod.BinaryQuadraticModel):
            model = BinaryQuadraticModel(
                linear=model.linear, quadratic=model.quadratic,
                offset=model.offset, var_type=model.vartype
            )

        if isinstance(model, BinaryQuadraticModel):
            if 'unit_num_L' in kwargs:
                self.unit_num_L = kwargs['unit_num_L']
            elif not self.unit_num_L:
                raise ValueError(
                    'Input "unit_num_L" to the argument or the constructor of GPUSQASampler.')
            chimera_model = ChimeraModel(
                model=model, unit_num_L=self.unit_num_L, gpu=True)
        elif isinstance(model, ChimeraModel):
            chimera_model = model
            chimera_model.gpu = True

        if chimera_model.unit_num_L % 2 != 0:
            raise ValueError('unit_num_L should be even number.')

        self.unit_num_L = chimera_model.unit_num_L
        self._set_model(chimera_model)

        self.model = chimera_model

        chimera = chimera_model.get_chimera_graph()

        return chimera

    def _gpu_sampling(self, system_name: str, init_generator, system_maker, **kwargs):

        # Check the system for GPU is compiled
        try:
            if system_name == "quantum":
                self.system_class = cxxjij.system.ChimeraTransverseGPU
            elif system_name == "classical":
                self.system_class = cxxjij.system.ChimeraClassicalGPU
        except AttributeError:
            raise AttributeError(
                'Does the computer you are running have a GPU? Compilation for the GPU has not been done. Please reinstall or compile.')

        # use all spins ?
        self._use_all = len(self.model.indices) == (
            self.unit_num_L * self.unit_num_L * 8)

        system = system_maker(init_generator())

        response = self._sampling(
            self.model, init_generator,
            algorithm, system, reinitialize_state, seed
        )

        return response

    def _set_model(self, model):
        self.model = model
        self.indices = model.indices
        self.energy_bias = model.energy_bias
        self.var_type = model.var_type
