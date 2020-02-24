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

void sum_scan_blelloch(unsigned int* const d_out,const unsigned int* const d_in,const size_t numElems);
void sum_scan_blelloch(unsigned int* const d_out,const bool* const d_in,const size_t numElems);
void cpu_sum_scan(unsigned int* const h_out,
	const bool* const h_in,
	const size_t numElems);
void cpu_sum_scan(unsigned int* const h_out,
	const unsigned int* const h_in,
	const size_t numElems);
void cpuDeleteFromArray(float* const d_outData, const bool* delete_array, const float* data, const size_t numElements, unsigned int dimension = 1);
void deleteFromArray(float* d_outData, const bool* delete_array, const float* data, const unsigned long numElements, unsigned int dimension = 1);

#endif /* DELETEFROMARRAY_H_ */