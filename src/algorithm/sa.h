#pragma once
#include <string>
#include "algorithm.h"
#include "../updater/classical_updater.h"

namespace openjij{
	namespace algorithm{
		using namespace openjij::updater;

		//simulated annealing
		class SA : public Algorithm{
			private:
				double beta_min;
				double beta_max;
				double step_length;
				double step_num;
			public:
				//TODO: add annealing schedule option
				SA(double beta_min, double beta_max, double step_length, size_t step_num)
					:beta_min(beta_min), beta_max(beta_max), step_length(step_length), step_num(step_num){}

				//do SA protocol
				void exec(ClassicalUpdater& updater, const std::string& algo = "");
		};
	}
}
