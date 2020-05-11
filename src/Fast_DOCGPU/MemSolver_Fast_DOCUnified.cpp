#include "MemSolver_Fast_DOCUnified.h"
#include <iostream>
#include <curand.h>
#include "../DOC_GPU/DOCGPU_Kernels.h"
#include <OsiClpSolverInterface.hpp>
//#include <OsiCbcSolverInterface.hpp>

#include <CelModel.h>
#include <CelIntVar.h>
#include <CelNumVar.h>
#include <CelVariableArray.h>
#include <CelExpression.h>




Memory_sizes MemSolver_Fast_DOCUnified::computeForAllocations(unsigned int dim, unsigned int number_of_points, unsigned int number_of_centroids_max, unsigned int m, unsigned int sample_size, unsigned k, size_t freeMem){
	bool optimal = false;
	Memory_sizes result;
	while(!optimal){
		OsiClpSolverInterface solver;
		rehearse::CelModel model(solver);

		// The number of centroids to use in each run
		rehearse::CelVariableArray<rehearse::CelNumVar> number_of_centroids;
		number_of_centroids.multiDimensionResize(1,k);

		// Create variables for all the sizes, along with one to find the maximal one
		rehearse::CelIntVar size_of_memory;
	
		rehearse::CelIntVar size_of_data_max;
		rehearse::CelIntVar size_of_samples_max;
		rehearse::CelIntVar size_of_centroids_max;
		rehearse::CelIntVar size_of_findDim_max;
		rehearse::CelIntVar size_of_findDim_count_max;
		rehearse::CelIntVar size_of_pointsContained_max;
		rehearse::CelIntVar size_of_pointsContained_count_max;
		rehearse::CelIntVar size_of_index_max;
		rehearse::CelIntVar size_of_randomStates_max;
		rehearse::CelIntVar size_of_bestDims_max;
		rehearse::CelIntVar lower;

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
		model.addConstraint(lower == 0.1);
	
		// Define the constraints
		for(int i = 0; i < k; i++){

			model.addConstraint( data_points.at(i) * dim * sizeof(float)
								 <= size_of_data_max);

			model.addConstraint( number_of_centroids[i] * m * sample_size * sizeof(unsigned int)
								 <= size_of_samples_max);

			model.addConstraint( (number_of_centroids[i]+1) * sizeof(unsigned int)
								 <= size_of_centroids_max); // +1 for ceilf

			model.addConstraint( number_of_centroids[i] * dim * m * sizeof(bool)
								 <= size_of_findDim_max);

			model.addConstraint( number_of_centroids[i] * dim * m * sizeof(unsigned int)
								 <= size_of_findDim_count_max);

				
			model.addConstraint( number_of_centroids[i] * m * sizeof(unsigned int)
								 <= size_of_index_max);
		
			model.addConstraint( number_of_centroids[i]
								 <= number_of_centroids_max);
		
			model.addConstraint( lower <=  number_of_centroids[i]);
		
		}
	
		model.addConstraint( (number_of_points+1)*sizeof(bool)
							 == size_of_bestDims_max);
		
		model.addConstraint( sizeof(unsigned int)
							 == size_of_pointsContained_count_max);
			
		model.addConstraint( 1024*2 * sizeof(curandState)
							 == size_of_randomStates_max); // OBS: hardcoded
	
		model.addConstraint( ((data_points.at(0)+1) * sizeof(bool))
							 == size_of_pointsContained_max);

	

		// Constraint that the maximum allocation for each of the arrays are smaller than the total amount of free memory.
		model.addConstraint(2*size_of_data_max + // times 2 to allow for the return array
							size_of_samples_max +
							size_of_centroids_max +
							size_of_findDim_max +
							size_of_findDim_count_max +
							size_of_pointsContained_max +
							size_of_pointsContained_count_max +
							size_of_index_max +
							size_of_randomStates_max +
							size_of_bestDims_max
							== size_of_memory);
	
		model.addConstraint(size_of_memory
							<= (size_t)(freeMem*0.94) // only utilise some fraction of memory, due to allocation overheads...

							);


		model.builderToSolver();
		solver.setLogLevel(0); // don't print stuff
		solver.initialSolve(); // solve


		// std::cout << solver.isAbandoned() << solver.isProvenOptimal() << solver.isProvenPrimalInfeasible() << solver.isProvenDualInfeasible() << solver.isPrimalObjectiveLimitReached() << solver.isDualObjectiveLimitReached() << solver.isIterationLimitReached() << std::endl;


		if(solver.isProvenPrimalInfeasible()){
			freeMem *= 1.5;
			std::cout << "increasing memory to " << freeMem << std::endl;
			continue;
				}else{
			optimal=true;
		}
		// create the output
   		bool allMedioids =true;
		for(int i = 0; i < k; i++){
			allMedioids &= model.getSolutionValue(number_of_centroids[i]) == number_of_centroids_max;
		}
		if(allMedioids){
			result.size_of_data =                  number_of_points*dim*sizeof(float);
			result.size_of_samples =               number_of_centroids_max*m*sample_size*sizeof(unsigned int);
			result.size_of_centroids =             number_of_centroids_max*sizeof(unsigned int);
			result.size_of_findDim =               number_of_centroids_max * dim * m * sizeof(bool);
			result.size_of_findDim_count =         number_of_centroids_max * dim * m * sizeof(unsigned int);
			result.size_of_pointsContained =       (number_of_points+1)*sizeof(bool);//(model.getSolutionValue(size_of_pointsContained_max));
			result.size_of_pointsContained_count = (model.getSolutionValue(size_of_pointsContained_count_max));
			result.size_of_index =                 number_of_centroids_max * m * sizeof(unsigned int);
			result.size_of_randomStates =          1024*2 * sizeof(curandState);
			result.size_of_bestDims =              (number_of_points+1)*sizeof(bool);//(model.getSolutionValue(size_of_bestDims_max));
			result.first_number_of_centroids =     (model.getSolutionValue(number_of_centroids[0]));
			result.size_of_score =                 0;
		}
		else{
			result.size_of_data =                  (size_t)(model.getSolutionValue(size_of_data_max));
			result.size_of_samples =               (size_t)(model.getSolutionValue(size_of_samples_max));
			result.size_of_centroids =             (size_t)ceilf(model.getSolutionValue(size_of_centroids_max)/4)*4;
			result.size_of_findDim =               (size_t)(model.getSolutionValue(size_of_findDim_max));
			result.size_of_findDim_count =         (size_t)(model.getSolutionValue(size_of_findDim_count_max));
			result.size_of_pointsContained =       (size_t)(number_of_points+1)*sizeof(bool);//(size_t)(model.getSolutionValue(size_of_pointsContained_max));
			result.size_of_pointsContained_count = (size_t)(model.getSolutionValue(size_of_pointsContained_count_max));
			result.size_of_index =                 (size_t)ceilf(model.getSolutionValue(size_of_index_max));
			result.size_of_randomStates =          (size_t)1024*2 * sizeof(curandState);
			result.size_of_bestDims =              (size_t)(number_of_points+1)*sizeof(bool);//(size_t)(model.getSolutionValue(size_of_bestDims_max));
			result.first_number_of_centroids =     (size_t)(model.getSolutionValue(number_of_centroids[0]));
			result.size_of_score =                 0;
		};

	

	
		// std::cout << model.getSolutionValue(size_of_index_max) << ", " << (model.getSolutionValue(size_of_index_max)) << ", " << ceilf(model.getSolutionValue(size_of_index_max)) <<  (size_t) ceilf(model.getSolutionValue(size_of_index_max)) << std::endl;
		// std::cout <<
		// 	result.size_of_data
		// 	+result.size_of_samples
		// 	+result.size_of_centroids
		// 	+result.size_of_findDim
		// 	+result.size_of_findDim_count
		// 	+result.size_of_pointsContained
		// 	+result.size_of_pointsContained_count
		// 	+result.size_of_index
		// 	+result.size_of_randomStates
		// 	+result.size_of_bestDims
		// 		  <<  " free mem: " << (size_t)(freeMem*0.94) << std::endl;

		// std::cout << " result.size_of_data: " << result.size_of_data << std::endl;
		// std::cout << " result.size_of_samples: " << result.size_of_samples << std::endl;
		// std::cout << " result.size_of_centroids: " << result.size_of_centroids << std::endl;
		// std::cout << " result.size_of_findDim: " << result.size_of_centroids << std::endl;
		// std::cout << " result.size_of_findDim_count: " << result.size_of_findDim_count << std::endl;
		// std::cout << " result.size_of_pointsContained: " << result.size_of_pointsContained << std::endl;
		// std::cout << " result.size_of_pointsContained_count: " << result.size_of_pointsContained_count << std::endl;
		// std::cout << " result.size_of_index: " << result.size_of_index << std::endl;
		// std::cout << " result.size_of_randomStates: " << result.size_of_randomStates << std::endl;
		// std::cout << " result.size_of_bestDims: " << result.size_of_bestDims << std::endl;
		// std::cout << " result.first_number_of_centroids: " << result.first_number_of_centroids << std::endl;

	
		assert(result.size_of_data                  == (size_t)number_of_points * dim * sizeof(float));
		assert(result.size_of_samples               >= (size_t)(result.first_number_of_centroids*m)
			   *(size_t)sample_size*sizeof(unsigned int));
		assert(result.size_of_centroids             >= (size_t)(result.first_number_of_centroids*sizeof(unsigned int)));
		assert(result.size_of_findDim               >= (size_t)(result.first_number_of_centroids*m)*dim*sizeof(bool));
		assert(result.size_of_findDim_count         >= result.first_number_of_centroids*m*sizeof(unsigned int));
		assert(result.size_of_pointsContained_count >= (size_t)sizeof(unsigned int));
		assert(result.size_of_index                 >= ((size_t)(result.first_number_of_centroids * m)) * sizeof(unsigned int));
		assert(result.size_of_bestDims              >= (size_t)number_of_points*sizeof(bool)+sizeof(bool));
	


		assert(result.size_of_data == number_of_points*dim*sizeof(float));
		return result;
	}
}



double MemSolver_Fast_DOCUnified::computeCentroidSizeForAllocation(Memory_sizes allocations, unsigned int dim, unsigned int number_of_points, unsigned int number_of_centroids_max, unsigned int m, unsigned int sample_size){
	
	OsiClpSolverInterface solver;
	rehearse::CelModel model(solver);
	
	rehearse::CelNumVar number_of_centroids;


	model.setObjective ( number_of_centroids );
	solver.setObjSense(-1.0); // maximize
	

	model.addConstraint(number_of_centroids * m * sample_size * sizeof(float)
						<= allocations.size_of_samples);

	model.addConstraint((number_of_centroids+1)*sizeof(unsigned int)
						<= allocations.size_of_centroids); // +1 for ceilf

	model.addConstraint(number_of_centroids * m * dim * sizeof(bool)
						<= allocations.size_of_findDim);

	model.addConstraint(number_of_centroids * m * dim * sizeof(bool)
						<= allocations.size_of_findDim_count);
				
	model.addConstraint(number_of_centroids * m * sizeof(unsigned int)
						<= allocations.size_of_index);
	
	model.addConstraint(number_of_centroids
						<= number_of_centroids_max);

	model.builderToSolver();
	solver.setLogLevel(0); // don't print stuff
	solver.initialSolve(); // solve
	
	if(solver.isProvenPrimalInfeasible()){
		throw std::runtime_error("Not enough free memory");
	}

	return model.getSolutionValue(number_of_centroids);
   
};



Array_sizes MemSolver_Fast_DOCUnified::computeArraySizes(double number_of_centroids,
												  unsigned int number_of_points,
												  unsigned int m,
												  unsigned int sample_size){
	
	Array_sizes res;
	res.number_of_centroids_f = number_of_centroids;
	res.number_of_centroids = ceilf(number_of_centroids);
	res.number_of_samples = (size_t)(number_of_centroids*m);
	res.number_of_values_in_samples = (size_t)(number_of_centroids*m)*sample_size;
	res.number_of_values_in_pointsContained = (size_t)number_of_points;
	
	return res;
}
