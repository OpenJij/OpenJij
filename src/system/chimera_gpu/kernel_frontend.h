#pragma once
#include <cuda_runtime.h>
#include <curand.h>

namespace openjij{
	namespace system{
		namespace chimera_gpu{

			void cuda_init_spin(int32_t*& dev_spin, float*& dev_random, uint32_t num_trot, uint32_t num_row, uint32_t num_col, curandGenerator_t& rng, dim3& grid, dim3& block);

			void cuda_run(float beta, float gamma, float s,
					int32_t*& dev_spin,
					float*& dev_random,
					float*& dev_J_out_p,
					float*& dev_J_out_n,
					float*& dev_J_in_0,
					float*& dev_J_in_1,
					float*& dev_J_in_2,
					float*& dev_J_in_3,
					float*& dev_H,
					uint32_t num_trot, uint32_t num_row, uint32_t num_col,
					curandGenerator_t& rng, dim3& grid, dim3& block
					);

		}
	}
}
