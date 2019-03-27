#pragma once
#include <string>
#include "algorithm.h"
#include "../updater/quantum_updater.h"

namespace openjij{
	namespace algorithm{
		using namespace openjij::updater;

		//simulated quantum annealing
		class SQA : public Algorithm{
			private:
				double beta;
				double gamma_min;
				double gamma_max;
				double step_length;
				double step_num;
			public:
				SQA(double beta, double gamma_min, double gamma_max, double step_length, size_t step_num)
					:beta(beta), gamma_min(gamma_min), gamma_max(gamma_max), step_length(step_length), step_num(step_num){}

				//do SQA protocol
				void exec(QuantumUpdater& updater, const std::string& algo = "");
		};
	}
}
