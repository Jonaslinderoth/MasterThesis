/*
 * ArgMax.h
 *
 *  Created on: Apr 13, 2020
 *      Author: mikkel
 */

#ifndef ARGMAX_H_
#define ARGMAX_H_

#include <vector>



int argMax(std::vector<float>* scores);


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

void createIndicesKernel(unsigned int dimGrid,
						 unsigned int dimBlock,
						 cudaStream_t stream,
						 unsigned int* index,
						 unsigned int length);


#endif /* ARGMAX_H_ */
