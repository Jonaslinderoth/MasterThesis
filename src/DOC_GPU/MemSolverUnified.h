#ifndef MEMSOLVERUNIFIED_H
#define MEMSOLVERUNIFIED_H
#include <stdlib.h>
#include "MemSolver.h"


class MemSolverUnified{
 public:
	static Memory_sizes computeForAllocations(unsigned int dim,
											  unsigned int number_of_points,
											  unsigned int number_of_centroids,
											  unsigned int m,
											  unsigned int sample_size,
											  unsigned int k,
											  size_t freeMem);
	
	static double computeCentroidSizeForAllocation(Memory_sizes allocations,
										unsigned int dim,
										unsigned int number_of_points,
										unsigned int number_of_centroids,
										unsigned int m,
										unsigned int sample_size);

	static Array_sizes computeArraySizes(double number_of_centroids,
										 unsigned int number_of_points,
										 unsigned int m,
										 unsigned int sample_size);
};

#endif
