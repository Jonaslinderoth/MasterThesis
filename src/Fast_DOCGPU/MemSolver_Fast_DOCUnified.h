#ifndef MEMSOLVER_FAST_DOC_H
#define MEMSOLVER_FAST_DOC_H
#include <stdlib.h>
#include "../DOC_GPU/MemSolver.h"

class MemSolver_Fast_DOCUnified{
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
