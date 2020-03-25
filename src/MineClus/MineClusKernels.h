#ifndef MINECLUSKERNELS_H
#define MINECLUSKERNELS_H
#include <vector>
#include <tuple>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

enum dublicatesType { Naive, Breaking, MoreBreaking };

void createItemSetWrapper(unsigned int dimGrid,
						  unsigned int dimBlock,
						  cudaStream_t stream,
						  float* data,
						  unsigned int dim,
						  unsigned int numberOfPoints,
						  unsigned int centroidId,
						  float width,
						  unsigned int* output);

void createInitialCandidatesWrapper(unsigned int dimGrid,
									unsigned int dimBlock,
									cudaStream_t stream,
									unsigned int dim,
									unsigned int* output
									);


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


void mergeCandidatesWrapper(unsigned int dimGrid,
							unsigned int dimBlock,
							cudaStream_t stream,
							unsigned int* candidates,
							unsigned int numberOfCandidates,
							unsigned int dim,
							unsigned int* output
							);


void findDublicatesWrapper(unsigned int dimGrid,
						   unsigned int dimBlock,
						   cudaStream_t stream,
						   unsigned int* candidates,
						   unsigned int numberOfCandidates,
						   unsigned int dim,
						   bool* output,
						   dublicatesType version = Naive
						   );
// Testing functions
// ONLY FOR TESTING THE KERNELS
std::vector<unsigned int> createItemSetTester(std::vector<std::vector<float>*>* data, unsigned int centroid, float width);
std::vector<unsigned int> createInitialCandidatesTester(unsigned int dim);
std::tuple<std::vector<unsigned int>,std::vector<float>, std::vector<bool>> countSupportTester(std::vector<std::vector<bool>> candidates, std::vector<std::vector<bool>> itemSet, unsigned int minSupp, float beta);
std::vector<unsigned int> mergeCandidatesTester(std::vector<std::vector<bool>> candidates);
std::vector<bool> findDublicatesTester(std::vector<std::vector<bool>> candidates, dublicatesType version = Naive);
#endif
