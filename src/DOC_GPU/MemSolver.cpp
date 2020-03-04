#include "MemSolver.h"
#include <iostream>
#include <OsiClpSolverInterface.hpp>
#include <CelModel.h>
#include <CelNumVar.h>




Memory_sizes MemSolver::computeForAllocations(unsigned int dim, unsigned int number_of_points, unsigned int number_of_centroids, unsigned int m, unsigned int sample_size, size_t freeMem){


	OsiClpSolverInterface solver;
	rehearse::CelModel model(solver);

	rehearse::CelNumVar x1;
	rehearse::CelNumVar x2;

	model.setObjective (       7 * x1 + 9 * x2 );

	model.addConstraint(       1 * x1 +     x2 == 18  );
	model.addConstraint(                    x2 <= 14  );
	model.addConstraint(       2 * x1 + 3 * x2 <= 50  );

	solver.setObjSense(-1.0);
	model.builderToSolver();
	solver.setLogLevel(0);
	solver.initialSolve();

	printf("Solution for x1 : %g\n", model.getSolutionValue(x1));
	printf("Solution for x2 : %g\n", model.getSolutionValue(x2));
	printf("Solution objective value = : %g\n", solver.getObjValue());
	
} 
