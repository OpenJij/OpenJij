#include "sqa.h"
#include <cmath>

namespace openjij {
	namespace algorithm {

		void SQA::exec(updater::QuantumUpdater& updater, const std::string& algo){
			double r_g = pow(gamma_min/gamma_max, 1.0/step_num);
			double gamma = gamma_max;
			//anneal
			while(gamma >= gamma_min){
				//step length -> number of MCS for each step
				for(size_t i=0; i<step_length; i++){
					//update
					updater.update(beta, gamma, algo);
				}
				gamma *= r_g;
			}
		}
	} // namespace algorithm
} // namespace openjij
