#include "MineClusKernels.h"
#include <assert.h>
#include <iostream>
#include <tuple>
#include "../randomCudaScripts/Utils.h"

/*
  Naive kernel for creating the itemSet. 
  Takes the data, and a index for a centroid, and creates the itemSet 
  The items are stored with the points as columns, and the dimensions as rows, 
  and then a row major fasion
*/
__global__ void createItemSet(float* data, unsigned int dim, unsigned int numberOfPoints, unsigned int centroidId, float width, unsigned int* output){
	unsigned int point = blockIdx.x*blockDim.x+threadIdx.x;
	if(point < numberOfPoints){
		unsigned int numberOfOutputs = ceilf((float)dim/32);
		// For each of the blocks in the output in this dimension
		for(unsigned int i = 0; i < numberOfOutputs; i++){
			unsigned int output_block = 0;
			// for each bit in a block
			for(unsigned int j = 0; j < 32; j++){
				// break if the last block dont line up with 32 bits
				if(i == numberOfOutputs-1 && j >= dim%32){
					break;
				}else{
					assert(dim*centroidId+i*32+j < numberOfPoints*dim);
					assert(point*dim+i*32+j < numberOfPoints*dim);
					// Check if the dimension are within the width, and write to block in register
					output_block |= ((abs(data[dim*centroidId+i*32+j] - data[point*dim+i*32+j]) < width) << j);
				}
			}
			// write block to global memory.
			assert(numberOfPoints*i+point < numberOfPoints*ceilf((float)dim/32));
			output[numberOfPoints*i+point] = output_block;
		}	
	}
}

/**
Thin wrapper for createItemSet
*/
void createItemSetWrapper(unsigned int dimGrid,
						  unsigned int dimBlock,
						  cudaStream_t stream,
						  float* data,
						  unsigned int dim,
						  unsigned int numberOfPoints,
						  unsigned int centroidId,
						  float width,
						  unsigned int* output){
	createItemSet<<<dimGrid, dimBlock, 0, stream>>>(data, dim, numberOfPoints, centroidId, width, output);
}


/**
   This function is only for testing that the kernel works correctly
*/
std::vector<unsigned int> createItemSetTester(std::vector<std::vector<float>*>* data, unsigned int centroid, float width){
	uint size = data->size();
	uint dim = data->at(0)->size();
	uint size_of_data = size*dim*sizeof(float);
	size_t size_of_output = size*ceilf((float)dim/32)*sizeof(unsigned int);
	float* data_h;
	unsigned int* output_h;
	cudaMallocHost((void**) &data_h, size_of_data);
	cudaMallocHost((void**) &output_h, size_of_output);
	for(int i = 0; i < size; i++){
		for(int j = 0; j < dim; j++){
			data_h[i*dim+j] = data->at(i)->at(j);
		}
	}
	float* data_d;
	cudaMalloc((void**) &data_d, size_of_data);
	cudaMemcpy(data_d, data_h, size_of_data, cudaMemcpyHostToDevice);
		
	unsigned int* output_d;
	cudaMalloc((void**) &output_d, size_of_output);

	unsigned int dimBlock = 1024;
	unsigned int dimGrid = ceilf((float)size/dimBlock);
	createItemSet<<<dimGrid, dimBlock>>>(data_d, dim, size, centroid, width, output_d);

	
	
	cudaMemcpy(output_h, output_d, size_of_output, cudaMemcpyDeviceToHost);
	std::vector<unsigned int> res;
	for(int i = 0; i < ceilf((float)dim/32)*size;i++){
		res.push_back(output_h[i]);
	}
	return res;
}


/**
   Creates the initial candidtes given the dimensions. 
*/
__global__ void createInitialCandidates(unsigned int dim, unsigned int* output){
	unsigned int candidate = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int numberOfBlocksPrPoint = ceilf((float)dim/32);
	unsigned int myBlock = candidate/32;
	if(candidate < dim){
	
		// make sure all are 0;
		for(int i = 0; i < numberOfBlocksPrPoint; i++){
			assert(candidate+dim * i < dim*(ceilf((float)dim/32)));
			output[candidate+dim * i] = 0;
		}
		// set the correct candidate
		unsigned int output_block = (1 << (candidate%32));
		output[candidate+dim*myBlock] = output_block;
	}
}


/**
   Thin wrapper for CreateInitialCandidates
*/
void createInitialCandidatesWrapper(unsigned int dimGrid,
									unsigned int dimBlock,
									cudaStream_t stream,
									unsigned int dim,
									unsigned int* output
									){
	createInitialCandidates<<<dimGrid, dimBlock, 0, stream>>>(dim, output);
}

/**
   This function is only for testing
*/
std::vector<unsigned int> createInitialCandidatesTester(unsigned int dim){
	unsigned int dimBlock = 1024;
	unsigned int dimGrid = ceilf((float)dim/dimBlock);

	size_t sizeof_output = dim*ceilf((float)dim/32)*sizeof(unsigned int);

	unsigned int* output_h = (unsigned int*) malloc(sizeof_output);
	unsigned int* output_d;

	cudaMalloc((void**) &output_d, sizeof_output);

	createInitialCandidates<<<dimGrid, dimBlock>>>(dim, output_d);

	cudaMemcpy(output_h, output_d, sizeof_output, cudaMemcpyDeviceToHost);
	
	std::vector<unsigned int> res;
	for(int i = 0; i < ceilf((float)dim/32)*dim;i++){
		res.push_back(output_h[i]);
	}
	return res;
}

__global__ void countSupport(unsigned int* candidates, unsigned int* itemSet,
							 unsigned int dim, unsigned int numberOfItems,
							 unsigned int numberOfCandidates,
							 unsigned int minSupp, float beta,
							 unsigned int* outSupp, float* outScore,
							 bool* outToBeDeleted){
	
	unsigned int candidate = blockIdx.x*blockDim.x+threadIdx.x;

	unsigned int numberOfBlocksPrItem = ceilf((float)dim/32);
	if(candidate < numberOfCandidates){
		unsigned int count = 0;
		for(unsigned int i = 0; i < numberOfItems; i++){
			bool isSubset = true;
			for(unsigned int j = 0; j < numberOfBlocksPrItem; j++){
				unsigned int itemBlock = itemSet[j*numberOfItems+i];
				unsigned int candidateBlock = candidates[j*numberOfCandidates + candidate];
				unsigned int candidateBlockCount = __popc(candidateBlock);
				unsigned int unionCount = __popc(itemBlock&candidateBlock);
				isSubset &= candidateBlockCount == unionCount;
			}
			
			count += isSubset;
		}
		outSupp[candidate] = count;
		// the subspace count below could be done in the loop above, to have one less load of the candidate.
		
		unsigned int subSpaceCount = 0;
		for(unsigned int j = 0; j < numberOfBlocksPrItem; j++){
			unsigned int candidateBlock = candidates[j*numberOfCandidates + candidate];
			subSpaceCount += __popc(candidateBlock);
		}
		outScore[candidate] = count*pow(((float) 1/beta),subSpaceCount) ; // calculate score and store
		outToBeDeleted[candidate] = count < minSupp;
	}
}

/**
Thin wrapper for CountSupport kernel
*/
void countSupportWrapper(unsigned int dimGrid,
						 unsigned int dimBlock,
						 cudaStream_t stream,
						 unsigned int* candidates,
						 unsigned int* itemSet,
						 unsigned int dim,
						 unsigned int numberOfItems,
						 unsigned int numberOfCandidates,
						 unsigned int minSupp,
						 float beta,
						 unsigned int* outSupp,
						 float* outScore,
						 bool* outToBeDeleted
						 ){
	countSupport<<<dimGrid, dimBlock, 0, stream>>>(candidates,
												   itemSet,
												   dim,
												   numberOfItems,
												   numberOfCandidates,
												   minSupp,
												   beta,
												   outSupp,
												   outScore,
												   outToBeDeleted);
};


/**
   ONLY For testing the kernel countSupport
*/
std::tuple<
	std::vector<unsigned int>,
	std::vector<float>,
	std::vector<bool>> countSupportTester(std::vector<std::vector<bool>> candidates, std::vector<std::vector<bool>> itemSet,
							 unsigned int minSupp, float beta){
	unsigned int numberOfCandidates = candidates.size();
	unsigned int numberOfItems = itemSet.size();
	unsigned int dim = itemSet.at(0).size();
	unsigned int numberOfBlocksPrElement = ceilf((float)dim/32);
	unsigned int bitsInLastBlock = dim%32;

	size_t sizeOfCandidates = numberOfCandidates*numberOfBlocksPrElement*sizeof(unsigned int);
	size_t sizeOfItemSet = numberOfItems*numberOfBlocksPrElement*sizeof(unsigned int);
	size_t sizeOfScores = numberOfCandidates*sizeof(float);
	size_t sizeOfSupport = numberOfCandidates*sizeof(unsigned int);
	size_t sizeOfToBeDeleted = numberOfCandidates*sizeof(bool);

	unsigned int dimBlock = 1024;
	unsigned int dimGrid = ceilf((float)dim/1024);

	unsigned int* candidates_h;
	unsigned int* itemSet_h;
	unsigned int* outSupport_h;
	float* outScores_h;
	bool* outToBeDeleted_h;

	unsigned int* candidates_d;
	unsigned int* itemSet_d;
	unsigned int* outSupport_d;
	float* outScores_d;
	bool* outToBeDeleted_d;

	cudaMallocHost((void**) &candidates_h, sizeOfCandidates);
	cudaMallocHost((void**) &itemSet_h, sizeOfItemSet);
	cudaMallocHost((void**) &outSupport_h, sizeOfSupport);
	cudaMallocHost((void**) &outScores_h, sizeOfScores);
	cudaMallocHost((void**) &outToBeDeleted_h, sizeOfToBeDeleted);

	cudaMalloc((void**) &candidates_d, sizeOfCandidates);
	cudaMalloc((void**) &itemSet_d, sizeOfItemSet);
	cudaMalloc((void**) &outSupport_d, sizeOfSupport);
	cudaMalloc((void**) &outScores_d, sizeOfScores);
	cudaMalloc((void**) &outToBeDeleted_d, sizeOfToBeDeleted);

	// fill candidates
	for(unsigned int i = 0; i < numberOfCandidates; i++){
		unsigned int block = 0;
		unsigned int blockNr = 0;
		for(int j = 0; j < dim; j++){
			if (j % 32 == 0 && j != 0){
				candidates_h[i+blockNr*numberOfCandidates] = block;
				block = 0;
				blockNr++;
			}
			block |= (candidates.at(i).at(j) << j);
		}
		candidates_h[i+blockNr*numberOfCandidates] = block;
	}

	// fill itemSet
	for(unsigned int i = 0; i < numberOfItems; i++){
		unsigned int block = 0;
		unsigned int blockNr = 0;
		for(int j = 0; j < dim; j++){
			if (j % 32 == 0 && j != 0){
				itemSet_h[i+blockNr*numberOfItems] = block;
				block = 0;
				blockNr++;
			}
			block |= (itemSet.at(i).at(j) << j);
			
		}
		itemSet_h[i+blockNr*numberOfItems] = block;
	}

	checkCudaErrors(cudaMemcpy(candidates_d, candidates_h, sizeOfCandidates, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(itemSet_d, itemSet_h, sizeOfItemSet, cudaMemcpyHostToDevice));
	
	countSupport<<<dimGrid, dimBlock>>>(candidates_d, itemSet_d, dim, numberOfItems, numberOfCandidates, minSupp, beta, outSupport_d, outScores_d, outToBeDeleted_d); 
	checkCudaErrors(cudaMemcpy(outSupport_h, outSupport_d, sizeOfSupport, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(outScores_h, outScores_d, sizeOfScores, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(outToBeDeleted_h, outToBeDeleted_d, sizeOfToBeDeleted, cudaMemcpyDeviceToHost));
	auto support = std::vector<unsigned int>();
	auto score = std::vector<float>();
	auto toBeDeleted = std::vector<bool>();
	for(unsigned int i = 0; i < numberOfCandidates; i++){
		support.push_back(outSupport_h[i]);
		score.push_back(outScores_h[i]);
		toBeDeleted.push_back(outToBeDeleted_h[i]);
	}
	
	std::tuple<
	std::vector<unsigned int>,
	std::vector<float>,
	std::vector<bool>
		> result;

	std::get<0>(result) = support;
	std::get<1>(result) = score;
	std::get<2>(result) = toBeDeleted;

	return result;
}



__global__ void mergeCandidates(unsigned int* candidates, unsigned int numberOfCandidates, unsigned int dim, unsigned int* output){
	unsigned int k = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int i = numberOfCandidates - 2- floorf(sqrtf(-8*k + 4*numberOfCandidates*(numberOfCandidates-1)-7)/ 2.0 - 0.5);
	unsigned int j = k + i + 1 - numberOfCandidates*(numberOfCandidates-1)/2 + (numberOfCandidates-i)*((numberOfCandidates-i)-1)/2;
	unsigned int numberOfNewCandidates = (numberOfCandidates*(numberOfCandidates+1))/2 - numberOfCandidates;
	unsigned int numberOfBlocks = ceilf((float)dim/32);

	if(k < numberOfNewCandidates){
		for(unsigned int a = 0; a < numberOfBlocks; a++){
			output[a*numberOfNewCandidates+k] = (candidates[a*numberOfCandidates+i] | candidates[a*numberOfCandidates+j]);
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
							unsigned int* output
							){
	mergeCandidates<<<dimGrid, dimBlock, 0, stream>>>(candidates, numberOfCandidates, dim, output);
};

std::vector<unsigned int> mergeCandidatesTester(std::vector<std::vector<bool>> candidates){
	unsigned int numberOfCandidates = candidates.size();
	unsigned int dim = candidates.at(0).size();
	unsigned int numberOfNewCandidates = ((numberOfCandidates*(numberOfCandidates+1)) / 2) - numberOfCandidates;
	unsigned int numberOfBlocks = ceilf((float)dim/32);

	size_t sizeOfOutput = numberOfNewCandidates*numberOfBlocks*sizeof(unsigned int);
	size_t sizeOfCandidates = numberOfCandidates*numberOfBlocks*sizeof(unsigned int);

	unsigned int dimBlock = 1024;
	unsigned int dimGrid = ceilf((float)numberOfNewCandidates/dimBlock);

	unsigned int* candidates_h;
	unsigned int* output_h;

	unsigned int* candidates_d;
	unsigned int* output_d;

	cudaMallocHost((void**) &candidates_h, sizeOfCandidates);
	cudaMallocHost((void**) &output_h, sizeOfOutput);

	cudaMalloc((void**) &candidates_d, sizeOfCandidates);
	cudaMalloc((void**) &output_d, sizeOfOutput);

	for(unsigned int i = 0; i < numberOfCandidates; i++){
		unsigned int block = 0;
		unsigned int blockNr = 0;
		for(int j = 0; j < dim; j++){
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

	mergeCandidates<<<dimGrid, dimBlock>>>(candidates_d, numberOfCandidates, dim, output_d);

	cudaMemcpy(output_h, output_d, sizeOfOutput, cudaMemcpyDeviceToHost);

	auto result = std::vector<unsigned int>();
	for(int i = 0; i < numberOfNewCandidates*numberOfBlocks; i++){
		
		result.push_back(output_h[i]);
	}
	return result;
			   
	
}



__global__ void findDublicatesNaive(unsigned int* candidates, unsigned int numberOfCandidates, unsigned int dim, bool* output){
	unsigned int candidate = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int numberOfBlocks = ceilf((float)dim/32);
	if(candidate < numberOfCandidates){
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


__global__ void findDublicatesBreaking(unsigned int* candidates, unsigned int numberOfCandidates, unsigned int dim, bool* output){
	unsigned int candidate = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int numberOfBlocks = ceilf((float)dim/32);
	if(candidate < numberOfCandidates){
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

__global__ void findDublicatesMoreBreaking(unsigned int* candidates, unsigned int numberOfCandidates, unsigned int dim, bool* output){
	unsigned int candidate = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int numberOfBlocks = ceilf((float)dim/32);
	if(candidate < numberOfCandidates){
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
						   bool* output,
						   dublicatesType version
						   ){
	if(version == Naive){
		findDublicatesNaive<<<dimGrid, dimBlock, 0, stream>>>(candidates, numberOfCandidates, dim, output);			
	}else if(version == Breaking){
		findDublicatesBreaking<<<dimGrid, dimBlock, 0, stream>>>(candidates, numberOfCandidates, dim, output);			
	}else if(version == MoreBreaking){
		findDublicatesMoreBreaking<<<dimGrid, dimBlock, 0, stream>>>(candidates, numberOfCandidates, dim, output);			
	}
};

/**
   ONLY FOR TESTING
*/
std::vector<bool> findDublicatesTester(std::vector<std::vector<bool>> candidates, dublicatesType version){
	unsigned int numberOfCandidates = candidates.size();
	unsigned int dim = candidates.at(0).size();
	unsigned int numberOfBlocks = ceilf((float)dim/32);

	unsigned int dimBlock = 1024;
	unsigned int dimGrid = ceilf((float)numberOfCandidates/dimBlock);

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

	for(unsigned int i = 0; i < numberOfCandidates; i++){
		unsigned int block = 0;
		unsigned int blockNr = 0;
		for(int j = 0; j < dim; j++){
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
		findDublicatesNaive<<<dimGrid, dimBlock>>>(candidates_d, numberOfCandidates, dim, output_d);
	}else if(version == Breaking){
		findDublicatesBreaking<<<dimGrid, dimBlock>>>(candidates_d, numberOfCandidates, dim, output_d);		
	}else if(version == MoreBreaking){
		findDublicatesMoreBreaking<<<dimGrid, dimBlock>>>(candidates_d, numberOfCandidates, dim, output_d);		
	}


	cudaMemcpy(output_h, output_d, sizeOfOutput, cudaMemcpyDeviceToHost);

	auto result = std::vector<bool>();
	for(int i = 0; i < numberOfCandidates; i++){
		result.push_back(output_h[i]);
	}

	return result;
}



