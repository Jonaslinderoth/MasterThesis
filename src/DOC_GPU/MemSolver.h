#ifndef MEMSOLVER_H
#define MEMSOLVER_H
#include <stdlib.h>

struct Memory_sizes{
	size_t size_of_data;
	size_t size_of_samples;
	size_t size_of_centroids;
	size_t size_of_findDim;
	size_t size_of_findDim_count;
	size_t size_of_pointsContained;
	size_t size_of_pointsContained_count;
	size_t size_of_score;
	size_t size_of_index;
	size_t size_of_randomStates;
	size_t size_of_bestDims;
};

struct Array_sizes{
	size_t number_of_samples;
	size_t number_of_values_in_samples;
	size_t number_of_bools_for_findDim;
	size_t number_of_values_in_pointsContained;
	size_t number_of_centroids;
	float number_of_centroids_f;
};

class MemSolver{
 public:
	static Memory_sizes computeForAllocations(unsigned int dim, unsigned int number_of_points, unsigned int number_of_centroids, unsigned int m, unsigned int sample_size, unsigned int k,  size_t freeMem);
	
	static Array_sizes computeForArrays(Memory_sizes allocations, unsigned int dim, unsigned int number_of_points, unsigned int number_of_centroids, unsigned int m, unsigned int sample_size, size_t freeMem);
};

#endif
