#include "MergeCandidates.h"
#include <assert.h>
#include <iostream>
#include "../randomCudaScripts/Utils.h"

__device__ __forceinline__ unsigned int computeI(unsigned int n, unsigned int t){
	double tmp = __dsqrt_rn((double)(-8)*t + 4*n*(n-1)-7);
	
	unsigned int i = n - 2- floor( tmp/ 2.0 - 0.5);	
	return i;
}

__device__  __forceinline__  unsigned int computeJ(unsigned int n, unsigned int t, unsigned int i){
	unsigned int j = t + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2;
	return j;
}

__device__ __forceinline__ unsigned int computeK(unsigned int i, unsigned int j, unsigned int n){
	unsigned int k = n*(n-1)/2 - (n-i)*((n-i)-1)/2 + j - i - 1;
	return k;
}

__device__ __forceinline__ unsigned int numberOfPairs(unsigned int n){
	unsigned int res = n*(n+1)/2 - n;
	return res;
}

__global__ void mergeCandidates(unsigned int* candidates, unsigned int numberOfCandidates, unsigned int dim, unsigned int iterNr,
								unsigned int* output, bool* toBeDeleted){
	unsigned int t = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int i = computeI(numberOfCandidates,t);
	unsigned int j = computeJ(numberOfCandidates,t,i);
	unsigned int numberOfNewCandidates = numberOfPairs(numberOfCandidates);
	unsigned int numberOfBlocks = ceilf((float)dim/32);
	
	assert(iterNr >= 2);
	if(t < numberOfNewCandidates){
		unsigned int interSectionCount=0;
		for(size_t a = 0; a < numberOfBlocks; a++){
			assert(a*numberOfNewCandidates+t < numberOfBlocks*numberOfNewCandidates);
			assert(a*numberOfCandidates+i < numberOfCandidates*numberOfBlocks);
			assert(a*numberOfCandidates+j < numberOfCandidates*numberOfBlocks);
			output[a*numberOfNewCandidates+t] = (candidates[a*numberOfCandidates+i] | candidates[a*numberOfCandidates+j]);
			interSectionCount += __popc(candidates[a*numberOfCandidates+i] & candidates[a*numberOfCandidates+j]);
		}
		toBeDeleted[t] = !((int)interSectionCount == (((int)iterNr)-2));
	}
}


__global__ void mergeCandidatesEarlyStopping(unsigned int* candidates, unsigned int numberOfCandidates, unsigned int dim, unsigned int iterNr,
								unsigned int* output, bool* toBeDeleted){
	unsigned int t = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int i = computeI(numberOfCandidates,t);
	unsigned int j = computeJ(numberOfCandidates,t,i);
	unsigned int numberOfNewCandidates = numberOfPairs(numberOfCandidates);
	unsigned int numberOfBlocks = ceilf((float)dim/32);
	
	assert(iterNr >= 2);
	if(t < numberOfNewCandidates){
		unsigned int interSectionCount=0;
		for(size_t a = 0; a < numberOfBlocks; a++){
			assert(a*numberOfNewCandidates+t < numberOfBlocks*numberOfNewCandidates);
			assert(a*numberOfCandidates+i < numberOfCandidates*numberOfBlocks);
			assert(a*numberOfCandidates+j < numberOfCandidates*numberOfBlocks);
			output[a*numberOfNewCandidates+t] = (candidates[a*numberOfCandidates+i] | candidates[a*numberOfCandidates+j]);
			interSectionCount += __popc(candidates[a*numberOfCandidates+i] & candidates[a*numberOfCandidates+j]);
			if((int)interSectionCount > (((int)iterNr)-2)){ // if the intersection is too big, then break
				break;
			}
		}
		toBeDeleted[t] = !((int)interSectionCount == (((int)iterNr)-2));
	}
}


__global__ void mergeCandidatesSmem(unsigned int* candidates,
									unsigned int numberOfCandidates,
									unsigned int dim,
									unsigned int iterNr,
									unsigned int chunkSize,
									unsigned int* output,
									bool* toBeDeleted){
	extern __shared__ unsigned int s[];
	unsigned int numberOfBlocks = ceilf((float)dim/32);
	unsigned int numberOfChunks = ceilf((float)numberOfCandidates/chunkSize);
	unsigned int* chunkI = (unsigned int*) s;
	unsigned int* chunkJ = (unsigned int*) &chunkI[chunkSize*numberOfBlocks];
	 
	// if only one chunk
	if(numberOfChunks == 1){
		for(unsigned int k = 0; k < ceilf((float)numberOfCandidates*numberOfBlocks/blockDim.x); k++){
			unsigned int a = (k*blockDim.x+ threadIdx.x)%numberOfCandidates; // current candidate
			unsigned int b = (k*blockDim.x+ threadIdx.x)/numberOfCandidates; // current block
			if(b*numberOfCandidates + a < numberOfCandidates*numberOfBlocks){
				chunkI[b*numberOfCandidates + a] = candidates[b*numberOfCandidates + a]; // number of canidates
			}
		}

		__syncthreads();

		
		for(unsigned int k = 0; k < ceilf((float)numberOfPairs(numberOfCandidates)/blockDim.x); k++){
			if(k*blockDim.x +threadIdx.x < numberOfPairs(numberOfCandidates)){
				unsigned int a = computeI((numberOfCandidates), (k*blockDim.x+ threadIdx.x)); // current candidate
				unsigned int b = computeJ((numberOfCandidates), (k*blockDim.x+ threadIdx.x), a); // current candidate
				unsigned int interSectionCount=0;
				unsigned int numberOfOutput = numberOfPairs(numberOfCandidates);
				for(unsigned int d = 0; d < numberOfBlocks; d++){
					output[numberOfOutput*d+k*blockDim.x +threadIdx.x] = (chunkI[numberOfCandidates*d+a] | chunkI[numberOfCandidates*d+b]);
					interSectionCount += __popc(chunkI[numberOfCandidates*d+a] & chunkI[numberOfCandidates*d+b]);
				}
				toBeDeleted[k*blockDim.x +threadIdx.x] = !((int)interSectionCount == (((int)iterNr)-2));				
			}

		}
		
	}else{
		unsigned int numberOfPartitions = numberOfPairs(numberOfChunks);
		unsigned int i = computeI(numberOfChunks, blockIdx.x);
		unsigned int j = computeJ(numberOfChunks, blockIdx.x, i);
		unsigned int chunkSizeI = min(numberOfCandidates - i*chunkSize, chunkSize);
		unsigned int chunkSizeJ = min(numberOfCandidates - j*chunkSize, chunkSize);

		for(unsigned int k = 0; k < ceilf((float)chunkSizeI*numberOfBlocks/blockDim.x); k++){
			unsigned int a = (k*blockDim.x+ threadIdx.x)%chunkSizeI; // current candidate
			unsigned int b = (k*blockDim.x+ threadIdx.x)/chunkSizeI; // current block
			if(b*chunkSizeI + a < chunkSizeI*numberOfBlocks){
				chunkI[k*blockDim.x+ threadIdx.x] = candidates[b*numberOfCandidates + i*chunkSize + a]; // number of canidates
			}
		}

		for(unsigned int k = 0; k < ceilf((float)chunkSizeJ*numberOfBlocks/blockDim.x); k++){
			unsigned int a = (k*blockDim.x+ threadIdx.x)%chunkSizeJ; // current candidate
			unsigned int b = (k*blockDim.x+ threadIdx.x)/chunkSizeJ; // current block
			if(b*chunkSizeJ + a < chunkSizeJ*numberOfBlocks){
				chunkJ[k*blockDim.x+ threadIdx.x] = candidates[b*numberOfCandidates + j*chunkSize + a]; // number of canidates
			}
		}

		__syncthreads();
		unsigned int outCand = numberOfPairs(numberOfCandidates);
		for(unsigned int k = 0; k < ceilf((float)chunkSizeI*chunkSizeJ/blockDim.x); k++){
			if(k*blockDim.x+threadIdx.x < chunkSizeI*chunkSizeJ){
				unsigned int a = (k*blockDim.x+ threadIdx.x)%chunkSizeI; // current candidate
				unsigned int b = (k*blockDim.x+ threadIdx.x)/chunkSizeI; // current block
				unsigned int pos = computeK(i*chunkSize + a,j*chunkSize + b,numberOfCandidates);
				unsigned int interSectionCount=0;
				for(unsigned int d = 0; d < numberOfBlocks; d++){
					output[outCand*d+pos] = (chunkI[chunkSizeI*d+a] | chunkJ[chunkSizeJ*d+b]);
					interSectionCount += __popc(chunkI[chunkSizeI*d+a] & chunkJ[chunkSizeJ*d+b]);
				}
				toBeDeleted[pos] = !((int)interSectionCount == (((int)iterNr)-2));
			}
		}

		
		// Take care of the half blocks below
		if(j-i == 1){
			for(unsigned int k = 0; k < ceilf((float)numberOfPairs(chunkSizeI)/blockDim.x); k++){
				if(k*blockDim.x+threadIdx.x < numberOfPairs(chunkSizeI)){
					unsigned int a = computeI((chunkSizeI), (k*blockDim.x+ threadIdx.x)); // current candidate
					unsigned int b = computeJ((chunkSizeI), (k*blockDim.x+ threadIdx.x), a); // current candidate
					unsigned int pos = computeK(i*chunkSize + a, (j-1)*chunkSize+b,numberOfCandidates);
					unsigned int interSectionCount=0;
					for(unsigned int d = 0; d < numberOfBlocks; d++){
						output[outCand*d+pos] = (chunkI[chunkSizeI*d+a] | chunkI[chunkSizeI*d+b]);
						interSectionCount += __popc(chunkI[chunkSizeI*d+a] & chunkI[chunkSizeI*d+b]);
					}
					toBeDeleted[pos] = !((int)interSectionCount == (((int)iterNr)-2));
				}
			}
		}

		
		// Take care of the last half block
		if(i == 0 and j == numberOfChunks-1){
			for(unsigned int k = 0; k < ceilf((float)numberOfPairs(chunkSizeJ)/blockDim.x); k++){
				if(k*blockDim.x+threadIdx.x < numberOfPairs(chunkSizeJ)){
					unsigned int a = computeI(chunkSizeJ, (k*blockDim.x+ threadIdx.x)); // current candidate
					unsigned int b = computeJ(chunkSizeJ, (k*blockDim.x+ threadIdx.x), a); // current candidate
					unsigned int pos = computeK((numberOfChunks-1)*chunkSize + a,
												(numberOfChunks-1)*chunkSize + b,numberOfCandidates);
					unsigned int interSectionCount=0;
					for(unsigned int d = 0; d < numberOfBlocks; d++){
						output[outCand*d+pos] = (chunkJ[chunkSizeJ*d+a] | chunkJ[chunkSizeJ*d+b]);
						interSectionCount += __popc(chunkJ[chunkSizeJ*d+a] & chunkJ[chunkSizeJ*d+b]);
					}
					toBeDeleted[pos] = !((int)interSectionCount == (((int)iterNr)-2));
				}
			}
		}

	}
	
}




/**
   Thin wrapper for mergeCandidates
*/
void mergeCandidatesWrapper(unsigned int dimGrid,
							unsigned int dimBlock,
							cudaStream_t stream,
							unsigned int* candidates,
							unsigned int numberOfCandidates,
							unsigned int dim,
							unsigned int itrNr,
							unsigned int* output,
							bool* toBeDeleted,
							mergeCandidatesType version,
							unsigned int chunkSize
							){
	
	if(version == NaiveMerge){
		mergeCandidates<<<dimGrid, dimBlock, 0, stream>>>(candidates, numberOfCandidates, dim, itrNr, output, toBeDeleted);
	}else if(version == EarlyStoppingMerge){
		mergeCandidatesEarlyStopping<<<dimGrid, dimBlock, 0, stream>>>(candidates, numberOfCandidates, dim, itrNr, output, toBeDeleted);
	}else if(version == SharedMemoryMerge){
		unsigned int numberOfBlocks = ceilf((float)dim/32);
		unsigned int largestChunkSize = ((48000/4)/2/numberOfBlocks);
		largestChunkSize = log2(largestChunkSize);
		largestChunkSize = pow(2, largestChunkSize);
		chunkSize = min(largestChunkSize, chunkSize);
		unsigned int smemSize = chunkSize*2*sizeof(unsigned int)*numberOfBlocks;
		unsigned int gridDimmension = ceilf(((float)numberOfCandidates/chunkSize));
		gridDimmension = ceilf((float)(gridDimmension*(gridDimmension+1)/2) -gridDimmension);
		gridDimmension = max(gridDimmension,1);

		dimBlock = min(dimBlock, numberOfBlocks*numberOfBlocks);
		dimBlock = min(dimBlock, 256);

		mergeCandidatesSmem<<<gridDimmension, dimBlock, smemSize, stream>>>(candidates, numberOfCandidates, dim, itrNr, chunkSize, output, toBeDeleted);
	}
};


std::pair<std::vector<unsigned int>,std::vector<bool>> mergeCandidatesTester(std::vector<std::vector<bool>> candidates, unsigned int itrNr, mergeCandidatesType version, unsigned int chunkSize){
	unsigned int numberOfCandidates = candidates.size();
	unsigned int dim = candidates.at(0).size();
	unsigned int numberOfNewCandidates = ((float)(numberOfCandidates*(numberOfCandidates+1)) / 2) - numberOfCandidates;
	unsigned int numberOfBlocks = ceilf((float)dim/32);
	
	size_t sizeOfOutput = numberOfNewCandidates*numberOfBlocks*sizeof(unsigned int);
	size_t sizeOfCandidates = numberOfCandidates*numberOfBlocks*sizeof(unsigned int);

	
	size_t sizeOfToBeDeleted = numberOfNewCandidates*sizeof(bool);
	
	unsigned int dimBlock = 1024;
	unsigned int dimGrid = ceilf((float)numberOfNewCandidates/dimBlock);
	dimGrid = dimGrid == 0 ? 1 : dimGrid;

	unsigned int* candidates_h;
	unsigned int* output_h;

	unsigned int* candidates_d;
	unsigned int* output_d;

	bool* toBeDeleted_h;
	bool* toBeDeleted_d;

	checkCudaErrors(cudaMallocHost((void**) &candidates_h, sizeOfCandidates));
	checkCudaErrors(cudaMallocHost((void**) &output_h, sizeOfOutput));
	checkCudaErrors(cudaMallocHost((void**) &toBeDeleted_h, sizeOfToBeDeleted));

	checkCudaErrors(cudaMalloc((void**) &candidates_d, sizeOfCandidates));
	checkCudaErrors(cudaMalloc((void**) &output_d, sizeOfOutput));
	checkCudaErrors(cudaMalloc((void**) &toBeDeleted_d, sizeOfToBeDeleted));



	for(unsigned int i = 0; i < numberOfCandidates; i++){
		unsigned int block = 0;
		unsigned int blockNr = 0;
		for(unsigned int j = 0; j < dim; j++){
			if (j % 32 == 0 && j != 0){
				candidates_h[(size_t)i+blockNr*numberOfCandidates] = block;
				block = 0;
				blockNr++;
			}
			block |= (candidates.at(i).at(j) << j);
		}
		candidates_h[i+blockNr*numberOfCandidates] = block;
	}
	checkCudaErrors(cudaMemcpy(candidates_d, candidates_h, sizeOfCandidates, cudaMemcpyHostToDevice));
	if(version == NaiveMerge){
		mergeCandidates<<<dimGrid, dimBlock>>>(candidates_d, numberOfCandidates, dim, itrNr, output_d, toBeDeleted_d);
	}else if(version == EarlyStoppingMerge){
		mergeCandidatesEarlyStopping<<<dimGrid, dimBlock>>>(candidates_d, numberOfCandidates, dim, itrNr, output_d, toBeDeleted_d);
	}else if(version == SharedMemoryMerge){

		unsigned int largestChunkSize = ((48000/4)/2/numberOfBlocks);
		largestChunkSize = log2(largestChunkSize);
		largestChunkSize = pow(2, largestChunkSize);
		chunkSize = min(largestChunkSize, chunkSize);
		unsigned int smemSize = chunkSize*2*sizeof(unsigned int)*numberOfBlocks;
		unsigned int gridDimmension = ceilf(((float)numberOfCandidates/chunkSize));
		gridDimmension = ceilf((float)(gridDimmension*(gridDimmension+1)/2) -gridDimmension);
		gridDimmension = max(gridDimmension,1);
		dimBlock = 256;
		mergeCandidatesSmem<<<gridDimmension, dimBlock, smemSize>>>(candidates_d, numberOfCandidates, dim, itrNr, chunkSize, output_d, toBeDeleted_d);
	}

	

	checkCudaErrors(cudaMemcpy(output_h, output_d, sizeOfOutput, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(toBeDeleted_h, toBeDeleted_d, sizeOfToBeDeleted, cudaMemcpyDeviceToHost));

	auto result = std::vector<unsigned int>();
	for(size_t i = 0; i < numberOfNewCandidates*numberOfBlocks; i++){
		
		result.push_back(output_h[i]);
	}

	auto result2 = std::vector<bool>();
	for(size_t i = 0; i < numberOfNewCandidates; i++){
		
		result2.push_back(toBeDeleted_h[i]);
	}

	if(numberOfCandidates > 2){
		checkCudaErrors(cudaFreeHost(candidates_h));
		checkCudaErrors(cudaFreeHost(output_h));
		checkCudaErrors(cudaFreeHost(toBeDeleted_h));

		checkCudaErrors(cudaFree(candidates_d));
		checkCudaErrors(cudaFree(output_d));
		checkCudaErrors(cudaFree(toBeDeleted_d));
	}
	return std::make_pair(result, result2);
			   
	
}



