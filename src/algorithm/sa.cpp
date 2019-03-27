#include "sa.h"
#include <cmath>

namespace openjij {
	namespace algorithm {

		void SA::exec(updater::ClassicalUpdater& updater, const std::string& algo){
			double r_beta = pow(beta_max/beta_min, 1.0/step_num);
			double beta = beta_min;
			//anneal
			while(beta <= beta_max){
				//step length -> number of MCS for each step
				for(size_t i=0; i<step_length; i++){
					//update
					updater.update(beta, algo);
				}
				beta *= r_beta;
			}
		}
	} // namespace algorithm
} // namespace openjij
