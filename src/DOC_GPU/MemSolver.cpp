#include "MemSolver.h"
#include <iostream>
#include <curand.h>
#include "DOCGPU_Kernels.h"
#include <OsiClpSolverInterface.hpp>
//#include <OsiCbcSolverInterface.hpp>

#include <CelModel.h>
#include <CelIntVar.h>
#include <CelNumVar.h>
#include <CelVariableArray.h>
#include <CelExpression.h>




Memory_sizes MemSolver::computeForAllocations(unsigned int dim, unsigned int number_of_points, unsigned int number_of_centroids_max, unsigned int m, unsigned int sample_size, unsigned k, size_t freeMem){
	
	OsiClpSolverInterface solver;
	rehearse::CelModel model(solver);

	// The number of centroids to use in each run
	rehearse::CelVariableArray<rehearse::CelNumVar> number_of_centroids;
	number_of_centroids.multiDimensionResize(1,k);


	// Create variables for all the sizes, along with one to find the maximal one
	rehearse::CelNumVar size_of_data_max;
	rehearse::CelNumVar size_of_samples_max;
	rehearse::CelNumVar size_of_centroids_max;
	rehearse::CelNumVar size_of_findDim_max;
	rehearse::CelNumVar size_of_findDim_count_max;
	rehearse::CelNumVar size_of_pointsContained_max;
	rehearse::CelNumVar size_of_pointsContained_count_max;
	rehearse::CelNumVar size_of_score_max;
	rehearse::CelNumVar size_of_index_max;
	rehearse::CelNumVar size_of_randomStates_max;
	rehearse::CelNumVar size_of_bestDims_max;

	std::vector<unsigned int> data_points;
	data_points.push_back(number_of_points);
	for(int i = 0; i < k; i++){
		data_points.push_back(data_points.at(i)/(2));	
	}


	// The objective is to maximize the number of centroids used.
	rehearse::CelExpression objective;
	for(int i = 0; i < k; i++){
		objective += number_of_centroids[i];
	}
	model.setObjective ( objective );
	solver.setObjSense(-1.0); // maximize

	
	// Define the constraints
	for(int i = 0; i < k; i++){

		model.addConstraint(data_points.at(i) * dim * sizeof(float) <= size_of_data_max);

		model.addConstraint(number_of_centroids[i] * m * sample_size * sizeof(float) <= size_of_samples_max);

		model.addConstraint((number_of_centroids[i]+1)*sizeof(unsigned int) <= size_of_centroids_max); // +1 for ceilf

		model.addConstraint(number_of_centroids[i] * m * dim * sizeof(bool) <= size_of_findDim_max);

		model.addConstraint(number_of_centroids[i] * m * dim * sizeof(bool) <= size_of_findDim_count_max);
		
		model.addConstraint((number_of_centroids[i] * m * data_points.at(i) +1) * sizeof(bool) <= size_of_pointsContained_max);

		model.addConstraint(number_of_centroids[i] * m * sizeof(unsigned int) <= size_of_pointsContained_count_max);

		model.addConstraint(number_of_centroids[i] * m * sizeof(float) <= size_of_score_max);
				
		model.addConstraint(number_of_centroids[i] * m * sizeof(unsigned int) <= size_of_index_max);

		model.addConstraint(1024*10 * sizeof(curandState) <= size_of_randomStates_max); // OBS: hardcoded

		model.addConstraint((number_of_points+1)*sizeof(bool) <= size_of_bestDims_max);
		
		model.addConstraint(number_of_centroids[i] <= number_of_centroids_max);
		
	}

	// Constraint that the maximum allocation for each of the arrays are smaller than the total amount of free memory.
	model.addConstraint(size_of_data_max +
						size_of_samples_max +
						size_of_centroids_max +
						size_of_findDim_max +
						size_of_findDim_count_max +
						size_of_pointsContained_max +
						size_of_pointsContained_count_max +
						size_of_score_max +
						size_of_index_max +
						size_of_randomStates_max +
						size_of_bestDims_max
						<= freeMem
						);

	

	model.builderToSolver();
	solver.setLogLevel(0); // don't print stuff
	solver.initialSolve(); // solve

	// create the output
	Memory_sizes result;

	result.size_of_data =                  model.getSolutionValue(size_of_data_max);
	result.size_of_samples =               model.getSolutionValue(size_of_samples_max);
	result.size_of_centroids =             model.getSolutionValue(size_of_centroids_max);
	result.size_of_findDim =               model.getSolutionValue(size_of_findDim_max);
	result.size_of_findDim_count =         model.getSolutionValue(size_of_findDim_count_max);
	result.size_of_pointsContained =       model.getSolutionValue(size_of_pointsContained_max);
	result.size_of_pointsContained_count = model.getSolutionValue(size_of_pointsContained_count_max);
	result.size_of_score =                 model.getSolutionValue(size_of_score_max);
	result.size_of_index =                 model.getSolutionValue(size_of_index_max);
	result.size_of_randomStates =          model.getSolutionValue(size_of_randomStates_max);
	result.size_of_bestDims =              model.getSolutionValue(size_of_bestDims_max);

	return result;
	
} 
