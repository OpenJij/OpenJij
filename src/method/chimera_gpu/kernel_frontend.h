#pragma once

namespace openjij{
	namespace method{
		namespace chimera_gpu{

			//functions
			void cuda_set_device(int device);

			void cuda_init(
					uint32_t arg_num_trot,
					uint32_t arg_num_row,
					uint32_t arg_num_col
					);

			void cuda_init_interactions(
					const float* J_out_p,
					const float* J_out_n,
					const float* J_in_0,
					const float* J_in_1,
					const float* J_in_2,
					const float* J_in_3,
					const float* H
					);

			void cuda_init_spin();

			void cuda_run(float beta, float gamma);

			void copy_spins();

			int32_t get_spin(uint32_t t, uint32_t i, uint32_t j, uint32_t ind);

			void cuda_free();

		}
	}
}
