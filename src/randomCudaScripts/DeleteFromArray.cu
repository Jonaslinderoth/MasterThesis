
#include "src/randomCudaScripts/DeleteFromArray.h"
#include <cmath>
#include <cuda.h>
#include <iostream>
#include "device_launch_parameters.h"




__global__ void gpu_add_block_sums(unsigned int* const d_out,
	const unsigned int* const d_in,
	unsigned int* const d_block_sums,
	const size_t numElems)
{
	//unsigned int glbl_t_idx = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int d_block_sum_val = d_block_sums[blockIdx.x];

	//unsigned int d_in_val_0 = 0;
	//unsigned int d_in_val_1 = 0;

	// Simple implementation's performance is not significantly (if at all)
	//  better than previous verbose implementation
	unsigned int cpy_idx = 2 * blockIdx.x * blockDim.x + threadIdx.x;
	if (cpy_idx < numElems)
	{
		d_out[cpy_idx] = d_in[cpy_idx] + d_block_sum_val;
		if (cpy_idx + blockDim.x < numElems)
			d_out[cpy_idx + blockDim.x] = d_in[cpy_idx + blockDim.x] + d_block_sum_val;
	}

	//if (2 * glbl_t_idx < numElems)
	//{
	//	d_out[2 * glbl_t_idx] = d_in[2 * glbl_t_idx] + d_block_sum_val;
	//	if (2 * glbl_t_idx + 1 < numElems)
	//		d_out[2 * glbl_t_idx + 1] = d_in[2 * glbl_t_idx + 1] + d_block_sum_val;
	//}

	//if (2 * glbl_t_idx < numElems)
	//{
	//	d_in_val_0 = d_in[2 * glbl_t_idx];
	//	if (2 * glbl_t_idx + 1 < numElems)
	//		d_in_val_1 = d_in[2 * glbl_t_idx + 1];
	//}
	//else
	//	return;
	//__syncthreads();

	//d_out[2 * glbl_t_idx] = d_in_val_0 + d_block_sum_val;
	//if (2 * glbl_t_idx + 1 < numElems)
	//	d_out[2 * glbl_t_idx + 1] = d_in_val_1 + d_block_sum_val;
}


// Modified version of Mark Harris' implementation of the Blelloch scan
//  according to https://www.mimuw.edu.pl/~ps209291/kgkp/slides/scan.pdf
__global__ void gpu_prescan(unsigned int* const d_out,
	const unsigned int* const d_in,
	unsigned int* const d_block_sums,
	const unsigned int len,
	const unsigned int shmem_sz,
	const unsigned int max_elems_per_block)
{
	// Allocated on invocation
	extern __shared__ unsigned int s_out[];

	int thid = threadIdx.x;
	int ai = thid;
	int bi = thid + blockDim.x;

	// Zero out the shared memory
	// Helpful especially when input size is not power of two
	s_out[thid] = 0;
	s_out[thid + blockDim.x] = 0;
	// If CONFLICT_FREE_OFFSET is used, shared memory
	//  must be a few more than 2 * blockDim.x
	if (thid + max_elems_per_block < shmem_sz)
		s_out[thid + max_elems_per_block] = 0;


	__syncthreads();

	// Copy d_in to shared memory
	// Note that d_in's elements are scattered into shared memory
	//  in light of avoiding bank conflicts
	unsigned int cpy_idx = max_elems_per_block * blockIdx.x + threadIdx.x;
	if (cpy_idx < len)
	{
		s_out[ai + CONFLICT_FREE_OFFSET(ai)] = d_in[cpy_idx];
		if (cpy_idx + blockDim.x < len)
			s_out[bi + CONFLICT_FREE_OFFSET(bi)] = d_in[cpy_idx + blockDim.x];
	}

	// For both upsweep and downsweep:
	// Sequential indices with conflict free padding
	//  Amount of padding = target index / num banks
	//  This "shifts" the target indices by one every multiple
	//   of the num banks
	// offset controls the stride and starting index of
	//  target elems at every iteration
	// d just controls which threads are active
	// Sweeps are pivoted on the last element of shared memory

	// Upsweep/Reduce step
	int offset = 1;
	for (int d = max_elems_per_block >> 1; d > 0; d >>= 1)
	{
		__syncthreads();

		if (thid < d)
		{
			int ai = offset * ((thid << 1) + 1) - 1;
			int bi = offset * ((thid << 1) + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			s_out[bi] += s_out[ai];
		}
		offset <<= 1;
	}

	// Save the total sum on the global block sums array
	// Then clear the last element on the shared memory
	if (thid == 0)
	{
		d_block_sums[blockIdx.x] = s_out[max_elems_per_block - 1
			+ CONFLICT_FREE_OFFSET(max_elems_per_block - 1)];
		s_out[max_elems_per_block - 1
			+ CONFLICT_FREE_OFFSET(max_elems_per_block - 1)] = 0;
	}

	// Downsweep step
	for (int d = 1; d < max_elems_per_block; d <<= 1)
	{
		offset >>= 1;
		__syncthreads();

		if (thid < d)
		{
			int ai = offset * ((thid << 1) + 1) - 1;
			int bi = offset * ((thid << 1) + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			unsigned int temp = s_out[ai];
			s_out[ai] = s_out[bi];
			s_out[bi] += temp;
		}
	}
	__syncthreads();

	// Copy contents of shared memory to global memory
	if (cpy_idx < len)
	{
		d_out[cpy_idx] = s_out[ai + CONFLICT_FREE_OFFSET(ai)];
		if (cpy_idx + blockDim.x < len)
			d_out[cpy_idx + blockDim.x] = s_out[bi + CONFLICT_FREE_OFFSET(bi)];
	}
}

// Modified version of Mark Harris' implementation of the Blelloch scan
//  according to https://www.mimuw.edu.pl/~ps209291/kgkp/slides/scan.pdf
__global__ void gpu_prescan(unsigned int* const d_out,
	const bool* const d_in,
	unsigned int* const d_block_sums,
	const unsigned int len,
	const unsigned int shmem_sz,
	const unsigned int max_elems_per_block)
{
	// Allocated on invocation
	extern __shared__ unsigned int s_out[];

	int thid = threadIdx.x;
	int ai = thid;
	int bi = thid + blockDim.x;

	// Zero out the shared memory
	// Helpful especially when input size is not power of two
	s_out[thid] = 0;
	s_out[thid + blockDim.x] = 0;
	// If CONFLICT_FREE_OFFSET is used, shared memory
	//  must be a few more than 2 * blockDim.x
	if (thid + max_elems_per_block < shmem_sz)
		s_out[thid + max_elems_per_block] = 0;


	__syncthreads();

	// Copy d_in to shared memory
	// Note that d_in's elements are scattered into shared memory
	//  in light of avoiding bank conflicts
	unsigned int cpy_idx = max_elems_per_block * blockIdx.x + threadIdx.x;
	if (cpy_idx < len)
	{
		s_out[ai + CONFLICT_FREE_OFFSET(ai)] = d_in[cpy_idx];
		if (cpy_idx + blockDim.x < len)
			s_out[bi + CONFLICT_FREE_OFFSET(bi)] = d_in[cpy_idx + blockDim.x];
	}

	// For both upsweep and downsweep:
	// Sequential indices with conflict free padding
	//  Amount of padding = target index / num banks
	//  This "shifts" the target indices by one every multiple
	//   of the num banks
	// offset controls the stride and starting index of
	//  target elems at every iteration
	// d just controls which threads are active
	// Sweeps are pivoted on the last element of shared memory

	// Upsweep/Reduce step
	int offset = 1;
	for (int d = max_elems_per_block >> 1; d > 0; d >>= 1)
	{
		__syncthreads();

		if (thid < d)
		{
			int ai = offset * ((thid << 1) + 1) - 1;
			int bi = offset * ((thid << 1) + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			s_out[bi] += s_out[ai];
		}
		offset <<= 1;
	}

	// Save the total sum on the global block sums array
	// Then clear the last element on the shared memory
	if (thid == 0)
	{
		d_block_sums[blockIdx.x] = s_out[max_elems_per_block - 1
			+ CONFLICT_FREE_OFFSET(max_elems_per_block - 1)];
		s_out[max_elems_per_block - 1
			+ CONFLICT_FREE_OFFSET(max_elems_per_block - 1)] = 0;
	}

	// Downsweep step
	for (int d = 1; d < max_elems_per_block; d <<= 1)
	{
		offset >>= 1;
		__syncthreads();

		if (thid < d)
		{
			int ai = offset * ((thid << 1) + 1) - 1;
			int bi = offset * ((thid << 1) + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			unsigned int temp = s_out[ai];
			s_out[ai] = s_out[bi];
			s_out[bi] += temp;
		}
	}
	__syncthreads();

	// Copy contents of shared memory to global memory
	if (cpy_idx < len)
	{
		d_out[cpy_idx] = s_out[ai + CONFLICT_FREE_OFFSET(ai)];
		if (cpy_idx + blockDim.x < len)
			d_out[cpy_idx + blockDim.x] = s_out[bi + CONFLICT_FREE_OFFSET(bi)];
	}
}


void sum_scan_blelloch(unsigned int* const d_out,
	const unsigned int* d_in,
	const size_t numElems)
{
	// Zero out d_out
	checkCudaErrors(cudaMemset(d_out, 0, numElems * sizeof(unsigned int)));

	// Set up number of threads and blocks

	unsigned int block_sz = MAX_BLOCK_SZ / 2;
	unsigned int max_elems_per_block = 2 * block_sz; // due to binary tree nature of algorithm

	// If input size is not power of two, the remainder will still need a whole block
	// Thus, number of blocks must be the ceiling of input size / max elems that a block can handle
	//unsigned int grid_sz = (unsigned int) std::ceil((double) numElems / (double) max_elems_per_block);
	// UPDATE: Instead of using ceiling and risking miscalculation due to precision, just automatically
	//  add 1 to the grid size when the input size cannot be divided cleanly by the block's capacity
	unsigned int grid_sz = numElems / max_elems_per_block;
	// Take advantage of the fact that integer division drops the decimals
	if (numElems % max_elems_per_block != 0)
		grid_sz += 1;

	// Conflict free padding requires that shared memory be more than 2 * block_sz
	unsigned int shmem_sz = max_elems_per_block + ((max_elems_per_block - 1) >> LOG_NUM_BANKS);

	// Allocate memory for array of total sums produced by each block
	// Array length must be the same as number of blocks
	unsigned int* d_block_sums;
	checkCudaErrors(cudaMalloc(&d_block_sums, sizeof(unsigned int) * grid_sz));
	checkCudaErrors(cudaMemset(d_block_sums, 0, sizeof(unsigned int) * grid_sz));

	// Sum scan data allocated to each block
	//gpu_sum_scan_blelloch<<<grid_sz, block_sz, sizeof(unsigned int) * max_elems_per_block >>>(d_out, d_in, d_block_sums, numElems);
	gpu_prescan<<<grid_sz, block_sz, sizeof(unsigned int) * shmem_sz>>>(d_out,
																	d_in,
																	d_block_sums,
																	numElems,
																	shmem_sz,
																	max_elems_per_block);

	// Sum scan total sums produced by each block
	// Use basic implementation if number of total sums is <= 2 * block_sz
	//  (This requires only one block to do the scan)
	if (grid_sz <= max_elems_per_block)
	{
		unsigned int* d_dummy_blocks_sums;
		checkCudaErrors(cudaMalloc(&d_dummy_blocks_sums, sizeof(unsigned int)));
		checkCudaErrors(cudaMemset(d_dummy_blocks_sums, 0, sizeof(unsigned int)));
		//gpu_sum_scan_blelloch<<<1, block_sz, sizeof(unsigned int) * max_elems_per_block>>>(d_block_sums, d_block_sums, d_dummy_blocks_sums, grid_sz);
		gpu_prescan<<<1, block_sz, sizeof(unsigned int) * shmem_sz>>>(d_block_sums,
																	d_block_sums,
																	d_dummy_blocks_sums,
																	grid_sz,
																	shmem_sz,
																	max_elems_per_block);
		checkCudaErrors(cudaFree(d_dummy_blocks_sums));
	}
	// Else, recurse on this same function as you'll need the full-blown scan
	//  for the block sums
	else
	{
		unsigned int* d_in_block_sums;
		checkCudaErrors(cudaMalloc(&d_in_block_sums, sizeof(unsigned int) * grid_sz));
		checkCudaErrors(cudaMemcpy(d_in_block_sums, d_block_sums, sizeof(unsigned int) * grid_sz, cudaMemcpyDeviceToDevice));
		sum_scan_blelloch(d_block_sums, d_in_block_sums, grid_sz);
		checkCudaErrors(cudaFree(d_in_block_sums));
	}

	//// Uncomment to examine block sums
	//unsigned int* h_block_sums = new unsigned int[grid_sz];
	//checkCudaErrors(cudaMemcpy(h_block_sums, d_block_sums, sizeof(unsigned int) * grid_sz, cudaMemcpyDeviceToHost));
	//std::cout << "Block sums: ";
	//for (int i = 0; i < grid_sz; ++i)
	//{
	//	std::cout << h_block_sums[i] << ", ";
	//}
	//std::cout << std::endl;
	//std::cout << "Block sums length: " << grid_sz << std::endl;
	//delete[] h_block_sums;

	// Add each block's total sum to its scan output
	// in order to get the final, global scanned array
	gpu_add_block_sums<<<grid_sz, block_sz>>>(d_out, d_out, d_block_sums, numElems);

	checkCudaErrors(cudaFree(d_block_sums));
}

//this is mikkel and jonas work.
__global__ void gpuDeleteFromArray(float* d_outData,
								   const unsigned int* d_delete_array,
								   const float* d_data,
								   const size_t numElements,
								   const unsigned int dimensions){
	unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;

	if(i < numElements){
		unsigned int offset = d_delete_array[i];
		unsigned int nextPrefix = d_delete_array[i+1];
		if(offset == nextPrefix){
			offset = i-offset;
			unsigned int offsetWithDim = offset*dimensions;
			unsigned int iWithDim = i*dimensions;
			for(int dimIndex = 0 ; dimIndex < dimensions ; ++dimIndex){
				d_outData[offsetWithDim+dimIndex] = d_data[iWithDim+dimIndex];
			}
		}

	}
}


void sum_scan_blelloch(unsigned int* const d_out,
	const bool* d_in,
	const size_t numElems)
{
	// Zero out d_out
	checkCudaErrors(cudaMemset(d_out, 0, numElems * sizeof(unsigned int)));

	// Set up number of threads and blocks

	unsigned int block_sz = MAX_BLOCK_SZ / 2;
	unsigned int max_elems_per_block = 2 * block_sz; // due to binary tree nature of algorithm

	// If input size is not power of two, the remainder will still need a whole block
	// Thus, number of blocks must be the ceiling of input size / max elems that a block can handle
	//unsigned int grid_sz = (unsigned int) std::ceil((double) numElems / (double) max_elems_per_block);
	// UPDATE: Instead of using ceiling and risking miscalculation due to precision, just automatically
	//  add 1 to the grid size when the input size cannot be divided cleanly by the block's capacity
	unsigned int grid_sz = numElems / max_elems_per_block;
	// Take advantage of the fact that integer division drops the decimals
	if (numElems % max_elems_per_block != 0)
		grid_sz += 1;

	// Conflict free padding requires that shared memory be more than 2 * block_sz
	unsigned int shmem_sz = max_elems_per_block + ((max_elems_per_block - 1) >> LOG_NUM_BANKS);

	// Allocate memory for array of total sums produced by each block
	// Array length must be the same as number of blocks
	unsigned int* d_block_sums;
	checkCudaErrors(cudaMalloc(&d_block_sums, sizeof(unsigned int) * grid_sz));
	checkCudaErrors(cudaMemset(d_block_sums, 0, sizeof(unsigned int) * grid_sz));

	// Sum scan data allocated to each block
	//gpu_sum_scan_blelloch<<<grid_sz, block_sz, sizeof(unsigned int) * max_elems_per_block >>>(d_out, d_in, d_block_sums, numElems);
	gpu_prescan<<<grid_sz, block_sz, sizeof(unsigned int) * shmem_sz>>>(d_out,
																	d_in,
																	d_block_sums,
																	numElems,
																	shmem_sz,
																	max_elems_per_block);

	// Sum scan total sums produced by each block
	// Use basic implementation if number of total sums is <= 2 * block_sz
	//  (This requires only one block to do the scan)
	if (grid_sz <= max_elems_per_block)
	{
		unsigned int* d_dummy_blocks_sums;
		checkCudaErrors(cudaMalloc(&d_dummy_blocks_sums, sizeof(unsigned int)));
		checkCudaErrors(cudaMemset(d_dummy_blocks_sums, 0, sizeof(unsigned int)));
		//gpu_sum_scan_blelloch<<<1, block_sz, sizeof(unsigned int) * max_elems_per_block>>>(d_block_sums, d_block_sums, d_dummy_blocks_sums, grid_sz);
		gpu_prescan<<<1, block_sz, sizeof(unsigned int) * shmem_sz>>>(d_block_sums,
																	d_block_sums,
																	d_dummy_blocks_sums,
																	grid_sz,
																	shmem_sz,
																	max_elems_per_block);
		checkCudaErrors(cudaFree(d_dummy_blocks_sums));
	}
	// Else, recurse on this same function as you'll need the full-blown scan
	//  for the block sums
	else
	{
		unsigned int* d_in_block_sums;
		checkCudaErrors(cudaMalloc(&d_in_block_sums, sizeof(unsigned int) * grid_sz));
		checkCudaErrors(cudaMemcpy(d_in_block_sums, d_block_sums, sizeof(unsigned int) * grid_sz, cudaMemcpyDeviceToDevice));
		sum_scan_blelloch(d_block_sums, d_in_block_sums, grid_sz);
		checkCudaErrors(cudaFree(d_in_block_sums));
	}

	//// Uncomment to examine block sums
	//unsigned int* h_block_sums = new unsigned int[grid_sz];
	//checkCudaErrors(cudaMemcpy(h_block_sums, d_block_sums, sizeof(unsigned int) * grid_sz, cudaMemcpyDeviceToHost));
	//std::cout << "Block sums: ";
	//for (int i = 0; i < grid_sz; ++i)
	//{
	//	std::cout << h_block_sums[i] << ", ";
	//}
	//std::cout << std::endl;
	//std::cout << "Block sums length: " << grid_sz << std::endl;
	//delete[] h_block_sums;

	// Add each block's total sum to its scan output
	// in order to get the final, global scanned array
	gpu_add_block_sums<<<grid_sz, block_sz>>>(d_out, d_out, d_block_sums, numElems);

	checkCudaErrors(cudaFree(d_block_sums));
}

void cpu_sum_scan(unsigned int* const h_out,
	const bool* const h_in,
	const size_t numElems)
{
	unsigned int run_sum = 0;
	for (int i = 0; i < numElems; ++i)
	{
		h_out[i] = run_sum;
		run_sum = run_sum + h_in[i];
	}
}


void cpu_sum_scan(unsigned int* const h_out,
	const unsigned int* const h_in,
	const size_t numElems)
{
	unsigned int run_sum = 0;
	for (int i = 0; i < numElems; ++i)
	{
		h_out[i] = run_sum;
		run_sum = run_sum + h_in[i];
	}
}



void cpuDeleteFromArray(float* const h_outData,
		const bool* h_delete_array,
		const float* data,
		const size_t numElements,
		unsigned int dimension){
	unsigned int ammountNotDeleted = 0;
	for(unsigned int i = 0 ; i < numElements ; ++i){
		if(!h_delete_array[i]){
			for(unsigned int dimIndex = 0 ; dimIndex < dimension ; ++dimIndex){
				h_outData[ammountNotDeleted+dimIndex] = data[i*dimension+dimIndex];
			}
			ammountNotDeleted += dimension;
		}
	}

}


/*
 * this fuction takes the data , and an array of bools that is one longer than the data where the last bool is not relevant.
 * return the list where entry i in the data is deleted if entry i in the bools array is 0.
 * it also deletes from the indexes keeping track of what is what.
 * but it does not resize the indexs , meaning that some if the indexs array can be garbage.
 */
void deleteFromArray(float* d_outData,
		const bool* d_delete_array,
		const float* d_data,
		const unsigned long numElements,
		const unsigned int dimension){

	// Set up device-side memory for output
	unsigned int* d_out_blelloch;
	checkCudaErrors(cudaMalloc(&d_out_blelloch, sizeof(unsigned int) * (numElements+1)));



	sum_scan_blelloch(d_out_blelloch,d_delete_array,(numElements+1));

	/*
	unsigned int* h_prefixSum = new unsigned int[numElements+1];
	std::cout << "prefix sum: ";
	checkCudaErrors(cudaMemcpy(h_prefixSum, d_out_blelloch, sizeof(unsigned int) * (numElements+1), cudaMemcpyDeviceToHost));

	for(unsigned int i = 0 ; i < (numElements+1) ; ++i){
		std::cout << h_prefixSum[i] << " " ;
	}
	std::cout << std::endl;
	*/


	//unsigned int* const d_outData, const unsigned int* delete_array, const float* data,unsigned int* indexes , const size_t numElements
	const unsigned int threadsUsed = 1024;
	unsigned int blocksNeccesary = numElements/threadsUsed;
	if(numElements%1024 != 0){
		blocksNeccesary++;
	}
	gpuDeleteFromArray<<<blocksNeccesary,threadsUsed>>>(d_outData,d_out_blelloch,d_data,numElements,dimension);

	cudaFree(d_out_blelloch);

}

