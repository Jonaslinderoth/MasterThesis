#include "FindDublicates.h"
#include <vector>

__global__ void findDublicatesNaive(unsigned int* candidates, unsigned int numberOfCandidates, unsigned int dim,
									bool* isAlreadyDeleted, bool* output){
	unsigned int candidate = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int numberOfBlocks = ceilf((float)dim/32);
	
	if(candidate < numberOfCandidates && !isAlreadyDeleted[candidate]){
		for(unsigned int i = candidate+1; i < numberOfCandidates; i++){
			bool equal = true;
			for(unsigned int j = 0; j < numberOfBlocks; j++){
				equal &= (candidates[candidate + numberOfCandidates*j] == candidates[i + numberOfCandidates*j]);
			}
			if(equal){
				output[i] = true;
			}
		}		
	}
}



__global__ void findDublicatesBreaking(unsigned int* candidates, unsigned int numberOfCandidates, unsigned int dim,
									bool* isAlreadyDeleted, bool* output){
	unsigned int candidate = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int numberOfBlocks = ceilf((float)dim/32);
	if(candidate < numberOfCandidates && !isAlreadyDeleted[candidate]){
		for(unsigned int i = candidate+1; i < numberOfCandidates; i++){
			bool equal = true;
			for(unsigned int j = 0; j < numberOfBlocks; j++){
				equal &= (candidates[candidate + numberOfCandidates*j] == candidates[i + numberOfCandidates*j]);
			}
			if(equal){
				output[i] = true;
				break;
			}
		}
	}
}

__global__ void findDublicatesMoreBreaking(unsigned int* candidates, unsigned int numberOfCandidates, unsigned int dim,
									bool* isAlreadyDeleted, bool* output){
	unsigned int candidate = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int numberOfBlocks = ceilf((float)dim/32);
	if(candidate < numberOfCandidates && !isAlreadyDeleted[candidate]){
		for(unsigned int i = candidate+1; i < numberOfCandidates; i++){
			bool equal = true;
			for(unsigned int j = 0; j < numberOfBlocks; j++){
				equal &= (candidates[candidate + numberOfCandidates*j] == candidates[i + numberOfCandidates*j]);
				if(!equal){
					break;
				}
			}
			if(equal){
				output[i] = true;
				break;
			}
		}
	}
}


/**
   Thin Wrapper for findDublicates
*/
void findDublicatesWrapper(unsigned int dimGrid,
						   unsigned int dimBlock,
						   cudaStream_t stream,
						   unsigned int* candidates,
						   unsigned int numberOfCandidates,
						   unsigned int dim,
						   bool* alreadyDeleted,
						   bool* output,
						   dublicatesType version
						   ){
	if(version == Naive){
		findDublicatesNaive<<<dimGrid, dimBlock, 0, stream>>>(candidates, numberOfCandidates, dim, alreadyDeleted, output);			
	}else if(version == Breaking){
		findDublicatesBreaking<<<dimGrid, dimBlock, 0, stream>>>(candidates, numberOfCandidates, dim, alreadyDeleted, output);			
	}else if(version == MoreBreaking){
		findDublicatesMoreBreaking<<<dimGrid, dimBlock, 0, stream>>>(candidates, numberOfCandidates, dim, alreadyDeleted, output);			
	}else if(version == Hash){
		findDublicatesHashTableWrapper(dimGrid, dimBlock, stream, candidates, numberOfCandidates, dim, alreadyDeleted, output);		 
	}
};

/**
   ONLY FOR TESTING
*/
std::vector<bool> findDublicatesTester(std::vector<std::vector<bool>> candidates, dublicatesType version){
	size_t numberOfCandidates = candidates.size();
	size_t dim = candidates.at(0).size();
	size_t numberOfBlocks = ceilf((float)dim/32);

	size_t dimBlock = 1024;
	size_t dimGrid = ceilf((float)numberOfCandidates/dimBlock);

	size_t sizeOfCandidates = numberOfCandidates*numberOfBlocks*sizeof(unsigned int);
	size_t sizeOfOutput = numberOfCandidates*sizeof(bool);

	unsigned int* candidates_h;
	bool* output_h;
	
	unsigned int* candidates_d;
	bool* output_d;

	cudaMallocHost((void**) &candidates_h, sizeOfCandidates);
	cudaMallocHost((void**) &output_h, sizeOfOutput);


	cudaMalloc((void**) &candidates_d, sizeOfCandidates);
	cudaMalloc((void**) &output_d, sizeOfOutput);



	bool* alreadyDeleted_d;
	cudaMalloc((void**) &alreadyDeleted_d, numberOfCandidates*sizeof(bool));
	cudaMemset(alreadyDeleted_d, 0, numberOfCandidates*sizeof(bool));

	for(size_t i = 0; i < numberOfCandidates; i++){
		size_t block = 0;
		size_t blockNr = 0;
		for(size_t j = 0; j < dim; j++){
			if (j % 32 == 0 && j != 0){
				candidates_h[i+blockNr*numberOfCandidates] = block;
				block = 0;
				blockNr++;
			}
			block |= (candidates.at(i).at(j) << j);
		}
		candidates_h[i+blockNr*numberOfCandidates] = block;
	}



	cudaMemcpy(candidates_d, candidates_h, sizeOfCandidates, cudaMemcpyHostToDevice);
	cudaMemset(output_d, false, sizeOfOutput);
	if(version == Naive){
		findDublicatesNaive<<<dimGrid, dimBlock>>>(candidates_d, numberOfCandidates, dim, alreadyDeleted_d, output_d);
	}else if(version == Breaking){
		findDublicatesBreaking<<<dimGrid, dimBlock>>>(candidates_d, numberOfCandidates, dim, alreadyDeleted_d, output_d);		
	}else if(version == MoreBreaking){
		findDublicatesMoreBreaking<<<dimGrid, dimBlock>>>(candidates_d, numberOfCandidates, dim, alreadyDeleted_d, output_d);		
	}else if(version == Hash){
		findDublicatesHashTableTester(dimGrid, dimBlock, candidates_d, numberOfCandidates, dim, alreadyDeleted_d, output_d);
	}


	cudaMemcpy(output_h, output_d, sizeOfOutput, cudaMemcpyDeviceToHost);

	auto result = std::vector<bool>();
	for(size_t i = 0; i < numberOfCandidates; i++){
		result.push_back(output_h[i]);
	}

	return result;
}
