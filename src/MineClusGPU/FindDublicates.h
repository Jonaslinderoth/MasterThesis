#ifndef FINDDUBLICATES_H
#define FINDDUBLICATES_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <vector>
#include "../MineClusGPU/HashTable.h"


enum dublicatesType { Naive, Breaking, MoreBreaking, Hash };

void findDublicatesWrapper(unsigned int dimGrid,
						   unsigned int dimBlock,
						   cudaStream_t stream,
						   unsigned int* candidates,
						   unsigned int numberOfCandidates,
						   unsigned int dim,
						   bool* alreadyDeleted,
						   bool* output,
						   dublicatesType version = dublicatesType::Naive
						   );

// ONLY FOR TESTING
std::vector<bool> findDublicatesTester(std::vector<std::vector<bool>> candidates, dublicatesType version = dublicatesType::Naive);
#endif
