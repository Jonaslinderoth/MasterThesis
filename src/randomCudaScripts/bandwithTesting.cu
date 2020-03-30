#include "src/randomCudaScripts/bandwithTesting.h"
#include <iostream>

__global__
void gpuTestBandWith(const unsigned int* d_data,
					 size_t dataSize){
	const size_t idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx < dataSize){
		if(d_data[idx] == 0){
			printf("failed test , need to have non zero in data");
		}
	}
}

__global__
void gpuTestBandWithNotCoalesced(const unsigned int* d_data,
					 size_t dataSize){
	const int idx = blockIdx.x*blockDim.x+threadIdx.x;
	const int offSet = (2*(1-(idx%2)))-1; //if 0 -> 1 ; if 1 -> -1
	const int newIdx = idx+offSet;
	//printf(" idx: %i offset: %i \n ", idx , offSet);

	if(newIdx < dataSize){
		if(d_data[newIdx] == 0){
			printf("failed test , need to have non zero in data");
		}
	}
}




void testBandWith(const unsigned int* d_data, size_t dataSize ,bool coalesced, bool print){


	const unsigned  int threadSize = 1024;
	unsigned int blockNeded = dataSize/threadSize;
	if(dataSize%threadSize != 0){
		blockNeded++;
	}

	float time = 0.0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	if(coalesced){
		gpuTestBandWith<<<blockNeded,threadSize>>>(d_data,dataSize);
	}else{
		gpuTestBandWithNotCoalesced<<<blockNeded,threadSize>>>(d_data,dataSize);
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	if(print)
	{
		std::cout << "time: " << std::to_string(time) << std::endl;
	}

}
