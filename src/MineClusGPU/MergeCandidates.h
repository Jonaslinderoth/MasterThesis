#ifndef MERGECANDIDATES_H
#define MERGECANDIDATES_H
#include <algorithm>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
enum mergeCandidatesType {NaiveMerge, SharedMemoryMerge, EarlyStoppingMerge};

void mergeCandidatesWrapper(unsigned int dimGrid,
							unsigned int dimBlock,
							cudaStream_t stream,
							unsigned int* candidates,
							unsigned int numberOfCandidates,
							unsigned int dim,
							unsigned int itrNr,
							unsigned int* output,
							bool* toBeDeleted,
							mergeCandidatesType version = NaiveMerge
							);

std::pair<std::vector<unsigned int>,std::vector<bool>>
	mergeCandidatesTester(std::vector<std::vector<bool>> candidates,
						  unsigned int itrNr = 2,
						  mergeCandidatesType version = NaiveMerge,
						  unsigned int chunkSize = 1024);


#endif
