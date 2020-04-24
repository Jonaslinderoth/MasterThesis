#ifndef COUNTSUPPORT_H
#define COUNTSUPPORT_H
#include <vector>
#include <tuple>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "../randomCudaScripts/Utils.h"

enum countSupportType {NaiveCount, SmemCount};

void countSupportWrapper(unsigned int dimGrid,
						 unsigned int dimBlock,
						 cudaStream_t stream,
						 unsigned int* candidates,
						 unsigned int* itemSet,
						 unsigned int dim,
						 unsigned int numberOfItems,
						 unsigned int numberOfCandidates,
						 unsigned int minSupp,
						 float beta,
						 unsigned int* outSupp,
						 float* outScore,
						 bool* outToBeDeleted
						 );


std::tuple<std::vector<unsigned int>,std::vector<float>, std::vector<bool>> countSupportTester(std::vector<std::vector<bool>> candidates, std::vector<std::vector<bool>> itemSet, unsigned int minSupp, float beta, countSupportType version = NaiveCount);

#endif
