from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

from openjij.sampler.chimera_gpu.gpu_sa_sampler import GPUChimeraSASampler
from openjij.sampler.chimera_gpu.gpu_sqa_sampler import GPUChimeraSQASampler

__all__ = ["GPUChimeraSASampler", "GPUChimeraSQASampler"]
