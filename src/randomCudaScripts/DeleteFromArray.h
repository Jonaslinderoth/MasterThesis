/*
 * DeleteFromArray.h
 *
 *  Created on: Feb 18, 2020
 *      Author: mikkel
 */

#ifndef DELETEFROMARRAY_H_
#define DELETEFROMARRAY_H_


//this is from https://github.com/mark-poscablo/gpu-prefix-sum

#include "src/randomCudaScripts/Utils.h"

#define MAX_BLOCK_SZ 1024
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5

#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(n) \
	((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif

void sum_scan_blelloch(cudaStream_t stream, unsigned int* const d_out,const unsigned int* const d_in,const size_t numElems);
void sum_scan_blelloch(cudaStream_t stream, unsigned int* const d_out, bool* d_in,const size_t numElems, bool inverted = false);
void sum_scan_blelloch_managed(cudaStream_t stream, cudaStream_t stream_preprocess, unsigned int* const d_out,
							   const unsigned int* d_in,
							   const size_t numElems);
void sum_scan_blelloch_managed(cudaStream_t stream, cudaStream_t stream_preprocess, unsigned int* const d_out,
							   bool* d_in,
							   const size_t numElems,
							   bool inverted);


void cpu_sum_scan(unsigned int* const h_out, const bool* const h_in, const size_t numElems);
void cpu_sum_scan(unsigned int* const h_out, const unsigned int* const h_in,const size_t numElems);

void cpuDeleteFromArray(float* const d_outData, const bool* delete_array, const float* data,
						const size_t numElements, unsigned int dimension = 1);


void deleteFromArrayWrapper(cudaStream_t stream,
							float* data,
							unsigned int* prefixSum,
							unsigned int numberOfPoints,
							unsigned int dim,
							float* output);

void deleteFromArraySpecialWrapper(cudaStream_t stream,
								   float* data,
								   unsigned int* prefixSum,
								   unsigned int numberOfPoints,
								   unsigned int dim,
								   float* output);

void deleteFromArray(cudaStream_t stream,
					 float* d_outData,
					 bool* delete_array,
					 const float* data,
					 const unsigned long numElements,
					 unsigned int dimension = 1,
					 bool inverted = false,
					 float* time = nullptr);

void deleteFromArraySpecial(cudaStream_t stream,
					 	    float* d_outData,
					        bool* delete_array,
					        const float* data,
					        const unsigned long numElements,
					        unsigned int dimension = 1,
					        bool inverted = false,
					        float* time = nullptr);


void deleteFromArray(float* d_outData,
					 bool* delete_array,
					 const float* data,
					 const unsigned long numElements,
					 unsigned int dimension = 1,
					 bool inverted = false,
					 float* time = nullptr);
void deleteFromArray_managed(cudaStream_t stream,
					 float* d_outData,
					 bool* d_delete_array,
					 const float* d_data,
					 const unsigned long numElements,
					 const unsigned int dimension,
					 const bool inverted = false,
							 float* time = nullptr);

void deleteFromArrayTrasformedData(cudaStream_t stream,
					 			   float* d_outData,
					 			   bool* delete_array,
					 			   const float* data,
					 			   const unsigned long numElements,
					 			   unsigned int dimension = 1,
					 			   bool inverted = false,
					 			   float* time = nullptr);


void deleteFromArrayOld(cudaStream_t stream,
						float* d_outData,
						bool* delete_array,
						const float* data,
						const unsigned long numElements,
						unsigned int dimension = 1,
						const bool inverted = false,
						float* time = nullptr);


void deleteFromArrayTransfomedDataWrapper(unsigned int dimGrid, unsigned int dimBlock, cudaStream_t stream,
										  float* data, unsigned int* prefixSum, unsigned int numberOfElements,
										  unsigned int dim, float* output);

void deleteFromArrayTransfomedDataWrapper(unsigned int dimGrid, unsigned int dimBlock, cudaStream_t stream,
										  unsigned int* data, unsigned int* prefixSum, unsigned int numberOfElements,
										  unsigned int dim, unsigned int* output);
void deleteFromArrayWrapper(unsigned int dimGrid,
							unsigned int dimBlock,
							cudaStream_t stream,
							float* data,
							unsigned int* prefixSum,
							unsigned int numberOfElements,
							unsigned int dim,
							float* output);


void deleteFromArrayWrapper(unsigned int dimGrid,
							unsigned int dimBlock,
							cudaStream_t stream,
							unsigned int* data,
							unsigned int* prefixSum,
							unsigned int numberOfElements,
							unsigned int dim,
							unsigned int* output);
#endif /* DELETEFROMARRAY_H_ */
