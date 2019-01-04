#include "../openjij/sampler/sampler.h"

#include <random>
#include <iostream>
#include <fstream>

void show_spins(openjij::Spins& spins){
	for(int s: spins)
		std::cout << s << " ";
	std::cout << std::endl;
}

int main(){

	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<double> dist(1.0, -1.0);

	int N = 10;

	openjij::SquareMatrix<double> int_mat(N, 0.0);
	// fully connected random ising model
	for(int i=0; i < N-1; i++){
		for(int j=i+1; j < N; j++){
			// double Jij = dist(mt); // if you want to calc random ising model
			int_mat(i, j) = -1.0;
			int_mat(j, i) = -1.0;
		}
	}

	openjij::sampler::Results resu;
	openjij::sampler::Sampler samp(int_mat);

	#ifdef SA
	// simulated annealing
	std::cout << "simulated annealing" << std::endl;
	double beta_min = 0.5;  // initial beta
	double beta_max = 2.0;  // final beta.
	int step_length = 10;   // Number of update at a certain temperature.
	int step_num    = 100;  // Number of temperature update.
	int iter 	    = 1;    // Iteration counts of SA process.
	samp.simulated_annealing(beta_min, beta_max, step_length, step_num, iter, resu);

	std::ofstream outputfile("sa_energy.dat");
	for(double e: resu.energies){
		outputfile << e << std::endl;
	}
	std::cout << "result:" << std::endl;
	show_spins(resu.states[0]);
	#endif // SA

    #ifdef SQA
	// simulated quantum annealing
	std::cout << "simulated quantum annealing" << std::endl;
	int trotter     = 5;
	double beta 	= 0.3;
	double g_min    = 0.5;
	double g_max    = 5.0;
	int step_length = 10;
	int step_num    = 100;
	int iter        = 1;
	samp.simulated_quantum_annealing(beta, g_min, g_max, trotter, step_length, step_num, iter, resu);
	std::cout << "result:" << std::endl;
	// trotterized spin is added trotter dimension.
	// show First trotter number spin configuration of first SQA result.
	show_spins(resu.quantum_states[0][0]);
    #endif

	return 0;
}



