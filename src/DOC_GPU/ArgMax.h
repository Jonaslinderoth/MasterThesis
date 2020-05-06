/*
 * ArgMax.h
 *
 *  Created on: Apr 13, 2020
 *      Author: mikkel
 */

#ifndef ARGMAX_H_
#define ARGMAX_H_
#include "../randomCudaScripts/Utils.h"
#include <vector>



int argMax(std::vector<float>* scores);

int argMaxBound(std::vector<float>* scores, unsigned int bound);


void argMaxKernel(unsigned int dimGrid,
				  unsigned int dimBlock,
				  unsigned int sharedMemorySize,
				  cudaStream_t stream,
				  float* scores,
				  unsigned int* scores_index,
				  unsigned int input_size);


void argMaxKernel(unsigned int dimGrid,
				  unsigned int dimBlock,
				  unsigned int sharedMemorySize,
				  cudaStream_t stream,
				  unsigned int* scores,
				  unsigned int* scores_index,
				  unsigned int input_size);


void argMaxKernelWidthUpperBound(unsigned int dimGrid,
								 unsigned int dimBlock,
								 unsigned int sharedMemorySize,
								 cudaStream_t stream,
								 unsigned int* scores,
								 unsigned int* scores_index,
								 unsigned int input_size,
								 unsigned int bound);

void argMaxKernelWidthUpperBound(unsigned int dimGrid,
								 unsigned int dimBlock,
								 unsigned int sharedMemorySize,
								 cudaStream_t stream,
								 float* scores,
								 unsigned int* scores_index,
								 unsigned int input_size,
								 unsigned int bound);


void createIndicesKernel(unsigned int dimGrid,
						 unsigned int dimBlock,
						 cudaStream_t stream,
						 unsigned int* index,
						 unsigned int length);


#endif /* ARGMAX_H_ */
