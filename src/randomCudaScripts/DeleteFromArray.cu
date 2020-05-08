
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
							bool*  d_in,
							unsigned int* const d_block_sums,
							const unsigned int len,
							const unsigned int shmem_sz,
							const unsigned int max_elems_per_block, bool inverted)
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
		bool a = d_in[cpy_idx] ^ inverted;
		s_out[ai + CONFLICT_FREE_OFFSET(ai)] = a;
		if (cpy_idx + blockDim.x < len)
			s_out[bi + CONFLICT_FREE_OFFSET(bi)] = d_in[cpy_idx + blockDim.x] ^ inverted;
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


void sum_scan_blelloch(cudaStream_t stream, 
					   unsigned int* const d_out,
					   const unsigned int* d_in,
					   const size_t numElems)
{
	// Zero out d_out
	checkCudaErrors(cudaMemsetAsync(d_out, 0, numElems * sizeof(unsigned int), stream));

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
	checkCudaErrors(cudaMemsetAsync(d_block_sums, 0, sizeof(unsigned int) * grid_sz, stream));

	// Sum scan data allocated to each block
	//gpu_sum_scan_blelloch<<<grid_sz, block_sz, sizeof(unsigned int) * max_elems_per_block >>>(d_out, d_in, d_block_sums, numElems);
	gpu_prescan<<<grid_sz, block_sz, sizeof(unsigned int) * shmem_sz, stream>>>(d_out,
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
		checkCudaErrors(cudaMemsetAsync(d_dummy_blocks_sums, 0, sizeof(unsigned int), stream));
		//gpu_sum_scan_blelloch<<<1, block_sz, sizeof(unsigned int) * max_elems_per_block>>>(d_block_sums, d_block_sums, d_dummy_blocks_sums, grid_sz);
		gpu_prescan<<<1, block_sz, sizeof(unsigned int) * shmem_sz, stream>>>(d_block_sums,
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
		checkCudaErrors(cudaMemcpyAsync(d_in_block_sums, d_block_sums, sizeof(unsigned int) * grid_sz, cudaMemcpyDeviceToDevice, stream));
		sum_scan_blelloch(stream, d_block_sums, d_in_block_sums, grid_sz);
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
	gpu_add_block_sums<<<grid_sz, block_sz, 0, stream>>>(d_out, d_out, d_block_sums, numElems);

	checkCudaErrors(cudaFree(d_block_sums));
}




void sum_scan_blelloch_managed(cudaStream_t stream, cudaStream_t stream_preprocess, 
					   unsigned int* const d_out,
					   const unsigned int* d_in,
					   const size_t numElems)
{
	// Zero out d_out
	checkCudaErrors(cudaMemPrefetchAsync(d_in, numElems * sizeof(unsigned int),0, stream_preprocess));
	checkCudaErrors(cudaMemPrefetchAsync(d_out, numElems * sizeof(unsigned int),0, stream_preprocess));
	checkCudaErrors(cudaMemsetAsync(d_out, 0, numElems * sizeof(unsigned int), stream_preprocess));

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
	checkCudaErrors(cudaMallocManaged(&d_block_sums, sizeof(unsigned int) * grid_sz));
	checkCudaErrors(cudaMemPrefetchAsync(d_block_sums, sizeof(unsigned int) * grid_sz,0, stream_preprocess));
	checkCudaErrors(cudaMemsetAsync(d_block_sums, 0, sizeof(unsigned int) * grid_sz, stream_preprocess));
	//checkCudaErrors(cudaStreamSynchronize(stream_preprocess)); 

	// Sum scan data allocated to each block
	//gpu_sum_scan_blelloch<<<grid_sz, block_sz, sizeof(unsigned int) * max_elems_per_block >>>(d_out, d_in, d_block_sums, numElems);
	gpu_prescan<<<grid_sz, block_sz, sizeof(unsigned int) * shmem_sz, stream>>>(d_out,
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
		checkCudaErrors(cudaMallocManaged(&d_dummy_blocks_sums, sizeof(unsigned int)));
		checkCudaErrors(cudaMemPrefetchAsync(d_dummy_blocks_sums, sizeof(unsigned int), 0, stream_preprocess));
		checkCudaErrors(cudaMemsetAsync(d_dummy_blocks_sums, 0, sizeof(unsigned int), stream_preprocess));
		//checkCudaErrors(cudaStreamSynchronize(stream_preprocess)); 
		gpu_prescan<<<1, block_sz, sizeof(unsigned int) * shmem_sz, stream>>>(d_block_sums,
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
		checkCudaErrors(cudaMallocManaged(&d_in_block_sums, sizeof(unsigned int) * grid_sz));
		checkCudaErrors(cudaMemPrefetchAsync(d_in_block_sums, sizeof(unsigned int) * grid_sz, 0, stream_preprocess));
		checkCudaErrors(cudaMemcpyAsync(d_in_block_sums, d_block_sums, sizeof(unsigned int) * grid_sz, cudaMemcpyDeviceToDevice, stream));
		sum_scan_blelloch_managed(stream, stream_preprocess, d_block_sums, d_in_block_sums, grid_sz);
		checkCudaErrors(cudaFree(d_in_block_sums));
	}

	// Add each block's total sum to its scan output
	// in order to get the final, global scanned array
	gpu_add_block_sums<<<grid_sz, block_sz, 0, stream>>>(d_out, d_out, d_block_sums, numElems);

	checkCudaErrors(cudaFree(d_block_sums));
}






//this is mikkel and jonas work.



__global__ void gpuDeleteFromArrayOld(float* d_outData,
								      const unsigned int* d_delete_array,
								      const float* d_data,
								      const size_t numElements,
								      const unsigned int dimensions,
								      const unsigned int numberOfThreadsPrPoint){
	extern __shared__ float temp[];
	unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int point = i/numberOfThreadsPrPoint;
	unsigned int pointOffset = i%numberOfThreadsPrPoint;
	unsigned int dim = ceilf((float)dimensions/(float)numberOfThreadsPrPoint); // the amount of dims i am responsible for
	unsigned int dimOffset = pointOffset*dim;
	unsigned int dim2 = dim;
	if(dimensions-dimOffset < dim){
		dim2 = dimensions-dimOffset;
	}
	
	/*if(i < 12){
	  printf("i %u, pointNr; %u, pointOffset %u, dim %u, dimOffset %u, dim2 %u \n", i, point, pointOffset, dim, dimOffset, dim2);
	  }*/


	if(point < numElements){
		unsigned int offset = d_delete_array[point];
		unsigned int nextPrefix = d_delete_array[point+1];
		for(int j = 0; j < dim2; j++){
			assert(threadIdx.x*dim+j < 48000/4);
			assert(dimOffset < dimensions);
			assert(point*dimensions+dimOffset+j < numElements*dimensions);
			temp[threadIdx.x*dim+j] = d_data[point*dimensions+dimOffset+j];
		}
		// Make sure data is sone written into shared memory
		__syncthreads();
		if(offset == nextPrefix){
			assert(point >= offset);
			offset = point-offset;	
			
			for(int j = 0; j < dim2; j++){
				d_outData[offset*dimensions+dimOffset+j] = temp[threadIdx.x*dim+j];
			}
		}

		// make sure everyone is done reading before overwriding
		__syncthreads();

	}
}

template<typename T>
__global__ void gpuDeleteFromArray(T* d_outData,
								   const unsigned int* d_delete_array,
								   const T* d_data,
								   const size_t numElements,
								   const unsigned int dimensions){
	const size_t idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx < numElements*dimensions){


		const size_t pointIdex = idx/dimensions;
		const size_t dimIndex = idx%dimensions;
		const size_t offSet = d_delete_array[pointIdex];
		const size_t nextOffSet = d_delete_array[pointIdex+1];
		const size_t newIndex = (pointIdex-offSet)*dimensions+dimIndex;
		const T theData = d_data[idx];
		if(offSet == nextOffSet){
			d_outData[newIndex] = theData;
		}
		
	}

}


__global__ void gpuDeleteFromArraySpeical(float* d_outData,
								          const unsigned int* d_delete_array,
								          const float* d_data,
								          const size_t numElements,
								          const unsigned int dimensions){
	const size_t idx = blockIdx.x*blockDim.x+threadIdx.x;

	for(size_t howLongOnTheData = 0 ; howLongOnTheData < numElements*dimensions ; howLongOnTheData+=4*blockDim.x){

		const size_t advIdex1 = idx+howLongOnTheData;
		const size_t advIdex2 = idx+howLongOnTheData+blockDim.x;
		const size_t advIdex3 = idx+howLongOnTheData+2*blockDim.x;
		const size_t advIdex4 = idx+howLongOnTheData+3*blockDim.x;
		float theData1;
		float theData2;
		float theData3;
		float theData4;

		if(advIdex1 < numElements*dimensions){
			theData1 = d_data[advIdex1];
			if(advIdex2 < numElements*dimensions){
				theData2 = d_data[advIdex2];
				if(advIdex3 < numElements*dimensions){
					theData3 = d_data[advIdex3];
					if(advIdex4 < numElements*dimensions){
						theData4 = d_data[advIdex4];
					}
				}
			}
		}

		if(advIdex1 < numElements*dimensions){
			{
				const size_t pointIdex = advIdex1/dimensions;
				const size_t dimIndex = advIdex1%dimensions;
				const size_t offSet = d_delete_array[pointIdex];
				const size_t nextOffSet = d_delete_array[pointIdex+1];
				const size_t newIndex = (pointIdex-offSet)*dimensions+dimIndex;

				if(offSet == nextOffSet){
					d_outData[newIndex] = theData1;
				}
			}

			if(advIdex2 < numElements*dimensions){
				{
					const size_t pointIdex = advIdex2/dimensions;
					const size_t dimIndex = advIdex2%dimensions;
					const size_t offSet = d_delete_array[pointIdex];
					const size_t nextOffSet = d_delete_array[pointIdex+1];
					const size_t newIndex = (pointIdex-offSet)*dimensions+dimIndex;

					if(offSet == nextOffSet){
						d_outData[newIndex] = theData2;
					}
				}
				if(advIdex3 < numElements*dimensions){
					{
						const size_t pointIdex = advIdex3/dimensions;
						const size_t dimIndex = advIdex3%dimensions;
						const size_t offSet = d_delete_array[pointIdex];
						const size_t nextOffSet = d_delete_array[pointIdex+1];
						const size_t newIndex = (pointIdex-offSet)*dimensions+dimIndex;

						if(offSet == nextOffSet){
							d_outData[newIndex] = theData3;
						}
					}
					if(advIdex4 < numElements*dimensions){
						{
							const size_t pointIdex = advIdex4/dimensions;
							const size_t dimIndex = advIdex4%dimensions;
							const size_t offSet = d_delete_array[pointIdex];
							const size_t nextOffSet = d_delete_array[pointIdex+1];
							const size_t newIndex = (pointIdex-offSet)*dimensions+dimIndex;

							if(offSet == nextOffSet){
								d_outData[newIndex] = theData4;
							}
						}
					}
				}
			}
		}


	}



}


template<typename T>
__global__ void gpuDeleteFromArrayTrasformed(T* d_outData,
								             const unsigned int* d_delete_array,
								             const T* d_data,
								             const size_t numElements,
								             const unsigned int dimensions){
	const size_t idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx < numElements*dimensions){
		const T theData = d_data[idx];
		const unsigned int whatPoint = idx%numElements;
		const unsigned int whatDim = idx/numElements;
		const unsigned int offSet = d_delete_array[whatPoint];
		const unsigned int nextOffSet = d_delete_array[whatPoint+1];
		const unsigned int maxOffSet = d_delete_array[numElements];
		const unsigned int newIndex = whatDim*(numElements-maxOffSet)+whatPoint-offSet;
		if(offSet == nextOffSet){
			d_outData[newIndex] = theData;
		}
		//printf(" theData: %f \n whatPoint %u \n whatDim %u \n offSet %u \n nextOffSet %u \n maxOffSet %u \n newIndex %u \n",theData,whatPoint,whatDim,offSet,nextOffSet,maxOffSet,newIndex);
	}

}



void sum_scan_blelloch(cudaStream_t stream, 
					   unsigned int* const d_out,
					   bool* d_in,
					   const size_t numElems,
					   bool inverted)
{
	// Zero out d_out
	checkCudaErrors(cudaMemsetAsync(d_out, 0, numElems * sizeof(unsigned int), stream));

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
	checkCudaErrors(cudaMemsetAsync(d_block_sums, 0, sizeof(unsigned int) * grid_sz, stream));

	// Sum scan data allocated to each block
	//gpu_sum_scan_blelloch<<<grid_sz, block_sz, sizeof(unsigned int) * max_elems_per_block >>>(d_out, d_in, d_block_sums, numElems);
	gpu_prescan<<<grid_sz, block_sz, sizeof(unsigned int) * shmem_sz, stream>>>(d_out,
																				d_in,
																				d_block_sums,
																				numElems,
																				shmem_sz,
																				max_elems_per_block, inverted);

	// Sum scan total sums produced by each block
	// Use basic implementation if number of total sums is <= 2 * block_sz
	//  (This requires only one block to do the scan)
	if (grid_sz <= max_elems_per_block)
	{
		unsigned int* d_dummy_blocks_sums;
		checkCudaErrors(cudaMalloc(&d_dummy_blocks_sums, sizeof(unsigned int)));
		checkCudaErrors(cudaMemsetAsync(d_dummy_blocks_sums, 0, sizeof(unsigned int), stream));
		//gpu_sum_scan_blelloch<<<1, block_sz, sizeof(unsigned int) * max_elems_per_block>>>(d_block_sums, d_block_sums, d_dummy_blocks_sums, grid_sz);
		gpu_prescan<<<1, block_sz, sizeof(unsigned int) * shmem_sz, stream>>>(d_block_sums,
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
		checkCudaErrors(cudaMemcpyAsync(d_in_block_sums, d_block_sums, sizeof(unsigned int) * grid_sz, cudaMemcpyDeviceToDevice, stream));
		sum_scan_blelloch(stream, d_block_sums, d_in_block_sums, grid_sz);
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
	gpu_add_block_sums<<<grid_sz, block_sz, 0, stream>>>(d_out, d_out, d_block_sums, numElems);

	checkCudaErrors(cudaFree(d_block_sums));
}




void sum_scan_blelloch_managed(cudaStream_t stream, cudaStream_t stream_preprocess, 
							   unsigned int* const d_out,
							   bool* d_in,
							   const size_t numElems,
							   bool inverted)
{
	// Zero out d_out
	checkCudaErrors(cudaMemPrefetchAsync(d_in, numElems * sizeof(bool),0, stream_preprocess));
	checkCudaErrors(cudaMemPrefetchAsync(d_out, numElems * sizeof(unsigned int),0, stream_preprocess));
	checkCudaErrors(cudaMemsetAsync(d_out, 0, numElems * sizeof(unsigned int), stream_preprocess));

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
	checkCudaErrors(cudaMallocManaged(&d_block_sums, sizeof(unsigned int) * grid_sz));
	checkCudaErrors(cudaMemPrefetchAsync(d_block_sums, sizeof(unsigned int) * grid_sz, 0, stream_preprocess));
	checkCudaErrors(cudaMemsetAsync(d_block_sums, 0, sizeof(unsigned int) * grid_sz, stream_preprocess));
	//	checkCudaErrors(cudaStreamSynchronize(stream_preprocess)); 

	// Sum scan data allocated to each block
	//gpu_sum_scan_blelloch<<<grid_sz, block_sz, sizeof(unsigned int) * max_elems_per_block >>>(d_out, d_in, d_block_sums, numElems);
	gpu_prescan<<<grid_sz, block_sz, sizeof(unsigned int) * shmem_sz, stream>>>(d_out,
																				d_in,
																				d_block_sums,
																				numElems,
																				shmem_sz,
																				max_elems_per_block, inverted);

	// Sum scan total sums produced by each block
	// Use basic implementation if number of total sums is <= 2 * block_sz
	//  (This requires only one block to do the scan)
	if (grid_sz <= max_elems_per_block)
	{
		unsigned int* d_dummy_blocks_sums;
		checkCudaErrors(cudaMallocManaged(&d_dummy_blocks_sums, sizeof(unsigned int)));
		checkCudaErrors(cudaMemPrefetchAsync(d_dummy_blocks_sums, sizeof(unsigned int), 0, stream_preprocess));
		checkCudaErrors(cudaMemsetAsync(d_dummy_blocks_sums, 0, sizeof(unsigned int), stream_preprocess));
		//	checkCudaErrors(cudaStreamSynchronize(stream_preprocess)); 
		gpu_prescan<<<1, block_sz, sizeof(unsigned int) * shmem_sz, stream>>>(d_block_sums,
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
		checkCudaErrors(cudaMallocManaged(&d_in_block_sums, sizeof(unsigned int) * grid_sz));
		checkCudaErrors(cudaMemPrefetchAsync(d_in_block_sums, sizeof(unsigned int) * grid_sz, 0, stream_preprocess));
		checkCudaErrors(cudaMemcpyAsync(d_in_block_sums, d_block_sums, sizeof(unsigned int) * grid_sz, cudaMemcpyDeviceToDevice, stream));
		sum_scan_blelloch_managed(stream, stream_preprocess, d_block_sums, d_in_block_sums, grid_sz);
		checkCudaErrors(cudaFree(d_in_block_sums));
	}

	// Add each block's total sum to its scan output
	// in order to get the final, global scanned array
	gpu_add_block_sums<<<grid_sz, block_sz, 0, stream>>>(d_out, d_out, d_block_sums, numElems);

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
void deleteFromArrayOld(cudaStream_t stream,
					    float* d_outData,
					    bool* d_delete_array,
					    const float* d_data,
					    const unsigned long numElements,
					    const unsigned int dimension,
					    bool inverted,
					    float* time){

	// Set up device-side memory for output
	unsigned int* d_out_blelloch;
	checkCudaErrors(cudaMalloc(&d_out_blelloch, sizeof(unsigned int) * (numElements+1)));

	sum_scan_blelloch(stream, d_out_blelloch,d_delete_array,(numElements+1), inverted);


	//unsigned int* const d_outData, const unsigned int* delete_array, const float* data,unsigned int* indexes , const size_t numElements
	const unsigned int threadsUsed = 1024;
	unsigned int numberOfvaluesPrThread = 10; // hardcoded, could be up to 11,... it is bounded by the size of shared memory
	unsigned int numberOfThreadsPrPoint = ceilf((float)dimension/(float)numberOfvaluesPrThread);
	
	
	unsigned int smem = threadsUsed*sizeof(float)*numberOfvaluesPrThread;

	unsigned int blocksNeccesary = (numElements*numberOfThreadsPrPoint)/threadsUsed;
	if((numElements*numberOfThreadsPrPoint)%1024 != 0){
		blocksNeccesary++;
	}
	cudaEvent_t start, stop;
	if(time != nullptr){
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);
	}
	gpuDeleteFromArrayOld<<<blocksNeccesary,threadsUsed,smem, stream>>>(d_outData,d_out_blelloch,d_data,numElements,dimension,numberOfThreadsPrPoint);

	cudaFree(d_out_blelloch);
	if(time != nullptr){
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(time, start, stop);
	}


}


/*
 * This is take two of the fuction , i make different versions to keep track of performance changes
 * this fuction takes the data , and an array of bools that is one longer than the data where the last bool is not relevant.
 * return the list where entry i in the data is deleted if entry i in the bools array is 0.
 * it also deletes from the indexes keeping track of what is what.
 * but it does not resize the indexs , meaning that some if the indexs array can be garbage.
 */
void deleteFromArray(cudaStream_t stream,
					 float* d_outData,
					 bool* d_delete_array,
					 const float* d_data,
					 const unsigned long numElements,
					 const unsigned int dimension,
					 const bool inverted,
					 float* time){
	const unsigned int threadsUsed = 1024;
	// Set up device-side memory for output
	unsigned int* d_out_blelloch;
	checkCudaErrors(cudaMalloc(&d_out_blelloch, sizeof(unsigned int) * (numElements+1)));

	sum_scan_blelloch(stream, d_out_blelloch,d_delete_array,(numElements+1), inverted);

	unsigned int blocksToUse = numElements*dimension/threadsUsed;
	if((numElements*dimension)%threadsUsed!=0){
		blocksToUse++;
	}
	cudaEvent_t start, stop;

	if(time != nullptr){
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);
	}

	gpuDeleteFromArray<<<blocksToUse,threadsUsed,0, stream>>>(d_outData,d_out_blelloch,d_data,numElements,dimension);

	checkCudaErrors(cudaFree(d_out_blelloch));
	if(time != nullptr){
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(time, start, stop);
	}

};

void deleteFromArrayWrapper(unsigned int dimGrid, unsigned int dimBlock, cudaStream_t stream,
							float* data, unsigned int* prefixSum, unsigned int numberOfElements,
							unsigned int dim, float* output){
	gpuDeleteFromArray<<<dimGrid, dimBlock, 0, stream>>>(output,
														 prefixSum,
														 data,
														 numberOfElements, dim);
};


void deleteFromArrayWrapper(unsigned int dimGrid, unsigned int dimBlock, cudaStream_t stream,
							unsigned int* data, unsigned int* prefixSum, unsigned int numberOfElements,
							unsigned int dim, unsigned* output){
	gpuDeleteFromArray<<<dimGrid, dimBlock, 0, stream>>>(output,
																prefixSum,
																data,
																numberOfElements, dim);
};

/*
 * This is take two of the fuction , i make different versions to keep track of performance changes
 * this fuction takes the data , and an array of bools that is one longer than the data where the last bool is not relevant.
 * return the list where entry i in the data is deleted if entry i in the bools array is 0.
 * it also deletes from the indexes keeping track of what is what.
 * but it does not resize the indexs , meaning that some if the indexs array can be garbage.
 */
void deleteFromArraySpecial(cudaStream_t stream,
					        float* d_outData,
					        bool* d_delete_array,
					        const float* d_data,
					        const unsigned long numElements,
					        const unsigned int dimension,
					        const bool inverted,
					        float* time){
	const unsigned int threadsUsed = 1024;
	// Set up device-side memory for output
	unsigned int* d_out_blelloch;
	checkCudaErrors(cudaMalloc(&d_out_blelloch, sizeof(unsigned int) * (numElements+1)));

	sum_scan_blelloch(stream, d_out_blelloch,d_delete_array,(numElements+1), inverted);

	unsigned int blocksToUse = 1;
	cudaEvent_t start, stop;

	if(time != nullptr){
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);
	}

	gpuDeleteFromArraySpeical<<<blocksToUse,threadsUsed,0, stream>>>(d_outData,d_out_blelloch,d_data,numElements,dimension);

	checkCudaErrors(cudaFree(d_out_blelloch));
	if(time != nullptr){
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(time, start, stop);
	}

}



/*
 * This fuction odes the same as the not transformed it just get the data in another formatting.
 * This is take two of the fuction , i make different versions to keep track of performance changes
 * this fuction takes the data , and an array of bools that is one longer than the data where the last bool is not relevant.
 * return the list where entry i in the data is deleted if entry i in the bools array is 0.
 * it also deletes from the indexes keeping track of what is what.
 * but it does not resize the indexs , meaning that some if the indexs array can be garbage.
 */
void deleteFromArrayTrasformedData(cudaStream_t stream,
					               float* d_outData,
					               bool* d_delete_array,
					               const float* d_data,
					               const unsigned long numElements,
					               const unsigned int dimension,
					               const bool inverted,
					               float* time){

	const unsigned int threadsUsed = 1024;
	// Set up device-side memory for output
	unsigned int* d_out_blelloch;
	checkCudaErrors(cudaMalloc(&d_out_blelloch, sizeof(unsigned int) * (numElements+1)));

	sum_scan_blelloch(stream, d_out_blelloch,d_delete_array,(numElements+1), inverted);

	unsigned int blocksToUse = numElements*dimension/threadsUsed;
	if((numElements*dimension)%threadsUsed!=0){
		blocksToUse++;
	}
	cudaEvent_t start, stop;

	if(time != nullptr){
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);
	}

	gpuDeleteFromArrayTrasformed<float><<<blocksToUse,threadsUsed,0, stream>>>(d_outData,d_out_blelloch,d_data,numElements,dimension);


	if(time != nullptr){
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(time, start, stop);
	}

	checkCudaErrors(cudaFree(d_out_blelloch));
};


void deleteFromArrayTransfomedDataWrapper(unsigned int dimGrid, unsigned int dimBlock, cudaStream_t stream,
										  float* data, unsigned int* prefixSum, unsigned int numberOfElements,
										  unsigned int dim, float* output){
	gpuDeleteFromArrayTrasformed<float><<<dimGrid, dimBlock, 0, stream>>>(output, prefixSum, data, numberOfElements, dim);
};

void deleteFromArrayTransfomedDataWrapper(unsigned int dimGrid, unsigned int dimBlock, cudaStream_t stream,
										  unsigned int* data, unsigned int* prefixSum, unsigned int numberOfElements,
										  unsigned int dim, unsigned int* output){
	gpuDeleteFromArrayTrasformed<unsigned int><<<dimGrid, dimBlock, 0, stream>>>(output, prefixSum, data, numberOfElements, dim);
};

void deleteFromArray(float* d_outData,
					 bool* d_delete_array,
					 const float* d_data,
					 const unsigned long numElements,
					 const unsigned int dimension,
					 bool inverted,
					 float* time){


	cudaStream_t stream;
	checkCudaErrors(cudaStreamCreate(&stream));
	deleteFromArray(stream, d_outData, d_delete_array, d_data, numElements, dimension, inverted,time);
	checkCudaErrors(cudaStreamDestroy(stream));
};
