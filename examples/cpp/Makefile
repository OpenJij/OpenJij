CPP = c++
OPENJIJ = ../openjij
UPDATER = $(OPENJIJ)/updater
SAMPLER = $(OPENJIJ)/sampler
CPPVER = c++11
sqa_example: example_optimization.cpp updater.o sampler.o
	$(CPP) -O3 -o sqa_example example_optimization.cpp sampler.o updater.o -DSQA -std=$(CPPVER)
sa_example: example_optimization.cpp updater.o sampler.o
	$(CPP) -O3 -o sa_example example_optimization.cpp sampler.o updater.o -DSA -std=$(CPPVER)
sampler.o: $(SAMPLER)/sampler.h $(SAMPLER)/sampler.cpp updater.o
	$(CPP) -c -O3 -o sampler.o $(SAMPLER)/sampler.cpp updater.o -std=$(CPPVER)
updater.o: $(UPDATER)/updater.h $(UPDATER)/single_spin_flip.h $(UPDATER)/single_spin_flip.cpp $(OPENJIJ)/model.h 
	$(CPP) -c -O3 -o updater.o $(UPDATER)/single_spin_flip.cpp -std=$(CPPVER)

