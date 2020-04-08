#include "MineClusKernels.h"
#include <assert.h>
#include <iostream>
#include <tuple>
#include "../randomCudaScripts/Utils.h"
#include "../MineClusGPU/HashTable.h"

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
				if(i == numberOfOutputs-1 && j == dim%32 && j != 0){
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
		//printf("candidat %u have value %u, writing output at position %u\n", candidate, output_block, candidate+dim*myBlock);
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
				// if(candidate == 0){
				// 	printf("candiate %u item value %u \n", candidate, itemBlock);
				// }
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
		//printf("candidate %u, have score %f \n", candidate, outScore[candidate]);
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



__global__ void mergeCandidates(unsigned int* candidates, unsigned int numberOfCandidates, unsigned int dim, unsigned int iterNr,
								unsigned int* output, bool* toBeDeleted){
	unsigned int k = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int i = numberOfCandidates - 2- floorf(sqrtf(-8*k + 4*numberOfCandidates*(numberOfCandidates-1)-7)/ 2.0 - 0.5);
	unsigned int j = k + i + 1 - numberOfCandidates*(numberOfCandidates-1)/2 + (numberOfCandidates-i)*((numberOfCandidates-i)-1)/2;
	unsigned int numberOfNewCandidates = (numberOfCandidates*(numberOfCandidates+1))/2 - numberOfCandidates;
	unsigned int numberOfBlocks = ceilf((float)dim/32);
	
	assert(iterNr >= 2);
	if(k < numberOfNewCandidates){
		unsigned int interSectionCount=0;
		for(unsigned int a = 0; a < numberOfBlocks; a++){
			assert(a*numberOfNewCandidates+k < numberOfBlocks*numberOfNewCandidates);
			output[a*numberOfNewCandidates+k] = (candidates[a*numberOfCandidates+i] | candidates[a*numberOfCandidates+j]);
			interSectionCount += __popc(candidates[a*numberOfCandidates+i] & candidates[a*numberOfCandidates+j]);
		}
		toBeDeleted[k] = !((int)interSectionCount == (((int)iterNr)-2));
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
							bool* toBeDeleted
							){
	mergeCandidates<<<dimGrid, dimBlock, 0, stream>>>(candidates, numberOfCandidates, dim, itrNr, output, toBeDeleted);
};


std::pair<std::vector<unsigned int>,std::vector<bool>> mergeCandidatesTester(std::vector<std::vector<bool>> candidates, unsigned int itrNr){
	unsigned int numberOfCandidates = candidates.size();
	unsigned int dim = candidates.at(0).size();
	unsigned int numberOfNewCandidates = ((numberOfCandidates*(numberOfCandidates+1)) / 2) - numberOfCandidates;
	unsigned int numberOfBlocks = ceilf((float)dim/32);

	size_t sizeOfOutput = numberOfNewCandidates*numberOfBlocks*sizeof(unsigned int);
	size_t sizeOfCandidates = numberOfCandidates*numberOfBlocks*sizeof(unsigned int);
	size_t sizeOfToBeDeleted = numberOfNewCandidates*sizeof(bool);
	
	unsigned int dimBlock = 1024;
	unsigned int dimGrid = ceilf((float)numberOfNewCandidates/dimBlock);

	unsigned int* candidates_h;
	unsigned int* output_h;

	unsigned int* candidates_d;
	unsigned int* output_d;

	bool* toBeDeleted_h;
	bool* toBeDeleted_d;

	cudaMallocHost((void**) &candidates_h, sizeOfCandidates);
	cudaMallocHost((void**) &output_h, sizeOfOutput);
	cudaMallocHost((void**) &toBeDeleted_h, sizeOfToBeDeleted);

	cudaMalloc((void**) &candidates_d, sizeOfCandidates);
	cudaMalloc((void**) &output_d, sizeOfOutput);
	cudaMalloc((void**) &toBeDeleted_d, sizeOfToBeDeleted);
	

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

	mergeCandidates<<<dimGrid, dimBlock>>>(candidates_d, numberOfCandidates, dim, itrNr, output_d, toBeDeleted_d);

	cudaMemcpy(output_h, output_d, sizeOfOutput, cudaMemcpyDeviceToHost);
	cudaMemcpy(toBeDeleted_h, toBeDeleted_d, sizeOfToBeDeleted, cudaMemcpyDeviceToHost);

	auto result = std::vector<unsigned int>();
	for(int i = 0; i < numberOfNewCandidates*numberOfBlocks; i++){
		
		result.push_back(output_h[i]);
	}

	auto result2 = std::vector<bool>();
	for(int i = 0; i < numberOfNewCandidates; i++){
		
		result2.push_back(toBeDeleted_h[i]);
	}
	return std::make_pair(result, result2);
			   
	
}



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



	bool* alreadyDeleted_d;
	cudaMalloc((void**) &alreadyDeleted_d, numberOfCandidates*sizeof(bool));
	cudaMemset(alreadyDeleted_d, 0, numberOfCandidates*sizeof(bool));

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
	for(int i = 0; i < numberOfCandidates; i++){
		result.push_back(output_h[i]);
	}

	return result;
}


__global__ void extractMax(unsigned int* candidates, float* scores, unsigned int centroid, unsigned int numberOfCandidates,
					  unsigned int* bestIndex,
					  unsigned int dim, unsigned int* bestCandidate, float* bestScore, unsigned int* bestCentroid){
	unsigned int block = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int numberOfBlocks = ceilf((float)dim/32);
	if(scores[0] > bestScore[0]){
		if(block < numberOfBlocks){
			bestCandidate[block] = candidates[block*numberOfCandidates+bestIndex[0]];
		}
		if(block == numberOfBlocks){
			bestCentroid[0] = centroid;
		}
		if(block == numberOfBlocks+1){
			bestScore[0] = scores[bestIndex[0]];
		}	
	}
}



void extractMaxWrapper(unsigned int dimGrid, unsigned int dimBlock, cudaStream_t stream,
					   unsigned int* candidates, float* scores, unsigned int centroid, unsigned int numberOfCandidates,
					   unsigned int* bestIndex,
					   unsigned int dim, unsigned int* bestCandidate, float* bestScore, unsigned int* bestCentroid){
	extractMax<<<dimGrid, dimBlock, 0, stream>>>(candidates, scores, centroid, numberOfCandidates,
												 bestIndex,
												 dim, bestCandidate, bestScore, bestCentroid);
}


std::pair<std::vector<unsigned int>, float> extractMaxTester(std::vector<bool> oldCandidate,
															 unsigned int oldScore, unsigned int oldCentroid,
															 std::vector<std::vector<bool>> newCandidates,
															 std::vector<float> newScores, unsigned int newCentroid,
															 unsigned int index){
	unsigned int numberOfBlocks = ceilf((float)oldCandidate.size()/32);
	unsigned int dim = oldCandidate.size();
	unsigned int numberOfCandidates = newCandidates.size();

	unsigned int* oldCandidate_h;
	unsigned int* newCandidates_h;

	unsigned int* oldCentroid_h;
	unsigned int* newCentroid_h;

	float* oldScore_h;
	float* newScores_h;

	cudaMallocHost((void**) &oldCandidate_h, numberOfBlocks*sizeof(unsigned int));
	cudaMallocHost((void**) &newCandidates_h, numberOfCandidates*numberOfBlocks*sizeof(unsigned int));

	cudaMallocHost((void**) &oldCentroid_h, sizeof(unsigned int));
	cudaMallocHost((void**) &newCentroid_h, sizeof(unsigned int));

	cudaMallocHost((void**) &oldScore_h, sizeof(float));
	cudaMallocHost((void**) &newScores_h, numberOfCandidates*sizeof(float));


	unsigned int* oldCandidate_d;
	unsigned int* newCandidates_d;

	unsigned int* oldCentroid_d;
	unsigned int* newCentroid_d;

	float* oldScore_d;
	float* newScores_d;
	
	cudaMalloc((void**) &oldCandidate_d, numberOfBlocks*sizeof(unsigned int));
	cudaMalloc((void**) &newCandidates_d, numberOfCandidates*numberOfBlocks*sizeof(unsigned int));

	cudaMalloc((void**) &oldCentroid_d, sizeof(unsigned int));
	cudaMalloc((void**) &newCentroid_d, sizeof(unsigned int));

	cudaMalloc((void**) &oldScore_d, sizeof(float));
	cudaMalloc((void**) &newScores_d, numberOfCandidates*sizeof(float));

	for(unsigned int i = 0; i < newScores.size(); i++){
		newScores_h[i] = newScores.at(i);
	}
	
	unsigned int value = 0;
	unsigned int blockNr = 0;
	for(unsigned int i = 0; i < oldCandidate.size(); i++){
		if(i %32 == 0 && i !=0){
			oldCandidate_h[blockNr] = value;
			blockNr++;
			value = 0;
		}
		value |= (oldCandidate.at(i) << i);
	}
	oldCandidate_h[blockNr] = value;

	oldScore_h[0] = oldScore;
	oldCentroid_h[0] = oldCentroid;


	for(unsigned int i = 0; i < numberOfCandidates; i++){
		unsigned int block = 0;
		unsigned int blockNr = 0;
		for(int j = 0; j < dim; j++){
			if (j % 32 == 0 && j != 0){
				newCandidates_h[i+blockNr*numberOfCandidates] = block;
				block = 0;
				blockNr++;
			}
			block |= (newCandidates.at(i).at(j) << j);
		}
		newCandidates_h[i+blockNr*numberOfCandidates] = block;
	}
	
	unsigned int* index_h;
	cudaMallocHost((void**) &index_h, sizeof(unsigned int));
	index_h[0] = index;
	
	unsigned int* index_d;
	cudaMalloc((void**) &index_d, sizeof(unsigned int));
	
	cudaMemcpy(index_d, index_h, sizeof(unsigned int), cudaMemcpyDeviceToHost);

	newCentroid_h[0] = newCentroid;

	cudaMemcpy(oldCandidate_d, oldCandidate_h, numberOfBlocks*sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(oldScore_d, oldScore_h, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(oldCentroid_d, oldCentroid_h, sizeof(unsigned int), cudaMemcpyHostToDevice);

	
	cudaMemcpy(newCandidates_d, newCandidates_h, numberOfCandidates*numberOfBlocks*sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(newScores_d, newScores_h, numberOfCandidates*sizeof(float), cudaMemcpyHostToDevice);
	//cudaMemcpy(newCentroid_d, newCentroid_h, sizeof(unsigned int), cudaMemcpyHostToDevice);

	extractMax<<<ceilf((float)(dim+2)/1024), 1024>>>(newCandidates_d, newScores_d, newCentroid, newCandidates.size(), index_d, dim,
												 oldCandidate_d, oldScore_d, oldCentroid_d);

	cudaMemcpy(oldCandidate_h, oldCandidate_d, numberOfBlocks*sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(oldScore_h, oldScore_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(oldCentroid_h, oldCentroid_d, sizeof(float), cudaMemcpyDeviceToHost);

	std::vector<unsigned int> bestCandidate;
	for(int i = 0; i < numberOfBlocks; i++){
		bestCandidate.push_back(oldCandidate_h[i]);
	}

	bestCandidate.push_back(oldCentroid_h[0]);

	std::pair<std::vector<unsigned int>, float> result;
	result = make_pair(bestCandidate, oldScore_h[0]);
	return result;

	
	}


__global__ void findPointsInCluster(unsigned int* candidate, float* data, float* centroid, unsigned int dim, unsigned int numberOfPoints, float width, bool* pointsContained){
	unsigned int point = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int numberOfBlocks = ceilf((float)dim/32);
	if(point < numberOfPoints){
		bool isContained = true;
		for(unsigned int i = 0; i < numberOfBlocks; i++){
			unsigned int block = candidate[i];
			for(unsigned int j = 0; j < 32; j++){
				if(i*32+j < dim){
					bool isDimChosen = (block >> j) & 1;
					float cent = centroid[i*32+j];
					float poin = data[point*dim+i*32+j];
					bool r = (not(isDimChosen)) || ((abs(cent - poin)) < width);
					isContained &= r;
				}else{
					break;
				}
			}
		}
		pointsContained[point] = isContained;
	}
}


void findPointInClusterWrapper(unsigned int dimGrid, unsigned int dimBlock, cudaStream_t stream,
							   unsigned int* candidate, float* data, float* centroid, unsigned int dim,
							   unsigned int numberOfPoints, float width, bool* pointsContained){
	findPointsInCluster<<<dimGrid, dimBlock, 0, stream>>>(candidate, data, centroid, dim, numberOfPoints, width, pointsContained);
}



std::vector<bool> findPointsInClusterTester(std::vector<bool> candidate, std::vector<std::vector<float>*>* data, unsigned int centroid, float width){
	unsigned int numberOfPoints = data->size();
	unsigned int dim = data->at(0)->size();
	unsigned int numberOfBlocks = ceilf((float)dim/32);
	
	size_t sizeOfCandidate = numberOfBlocks*sizeof(unsigned int);
	size_t sizeOfCentroid = dim*sizeof(float);
	size_t sizeOfData = numberOfPoints * dim * sizeof(float);
	size_t sizeOfPointsContained = numberOfPoints * sizeof(bool);
	
	unsigned int* candidate_h;
	float* centroid_h;
	float* data_h;
	bool* pointsContained_h;

	cudaMallocHost((void**) &candidate_h, sizeOfCandidate);
	cudaMallocHost((void**) &centroid_h, sizeOfCentroid);
	cudaMallocHost((void**) &data_h, sizeOfData);
	cudaMallocHost((void**) &pointsContained_h, sizeOfPointsContained);

	unsigned int* candidate_d;
	float* centroid_d;
	float* data_d;
	bool* pointsContained_d;

	cudaMalloc((void**) &candidate_d, sizeOfCandidate);
	cudaMalloc((void**) &centroid_d, sizeOfCentroid);
	cudaMalloc((void**) &data_d, sizeOfData);	
	cudaMalloc((void**) &pointsContained_d, sizeOfPointsContained);
	
	unsigned int value = 0;
	unsigned int blockNr = 0;
	for(unsigned int i = 0; i < candidate.size(); i++){
		if(i %32 == 0 && i !=0){
			candidate_h[blockNr] = value;
			blockNr++;
			value = 0;
		}
		value |= (candidate.at(i) << i);
	}
	candidate_h[blockNr] = value;

	for(unsigned int i = 0; i < numberOfPoints; i++){
		for(unsigned int j = 0; j < dim; j++){
			data_h[i*dim+j] = data->at(i)->at(j);
		}
	}

	for(int i = 0; i < dim; i++){
		centroid_h[i] = data->at(centroid)->at(i);
	}

	
	
	unsigned int dimBlock = 1024;
	unsigned int dimGrid = ceilf((float)numberOfPoints/dimBlock);

	cudaMemcpy(candidate_d, candidate_h, sizeOfCandidate, cudaMemcpyHostToDevice);
	cudaMemcpy(data_d, data_h, sizeOfData, cudaMemcpyHostToDevice);
	cudaMemcpy(centroid_d, centroid_h, sizeOfCentroid, cudaMemcpyHostToDevice);


	
	findPointsInCluster<<<dimGrid, dimBlock>>>(candidate_d, data_d, centroid_d, dim, numberOfPoints, width, pointsContained_d);

	cudaMemcpy(pointsContained_h, pointsContained_d, sizeOfPointsContained, cudaMemcpyDeviceToHost);
	
	auto result = std::vector<bool>();
	for(unsigned int i = 0; i < numberOfPoints; i++){
		result.push_back(pointsContained_h[i]);
	}
	return result;
}


__global__ void orKernel(unsigned int numberOfElements, bool* a, bool* b){
	unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
	if(i < numberOfElements){
		a[i] |= b[i];	
	}
}

void orKernelWrapper(unsigned int dimGrid, unsigned int dimBlock,  cudaStream_t stream,
					  unsigned int numberOfElements, bool* a, bool* b){
	orKernel<<<dimGrid, dimBlock, 0, stream>>>(numberOfElements, a, b);
}



__global__ void disjointClusters(unsigned int* centroids, float* scores, unsigned int* subspaces, float* data, const unsigned int numberOfClusters, unsigned int dim, float width, unsigned int* output){
	extern __shared__ unsigned int out[];
	unsigned int k = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int numberOfComparisons = (numberOfClusters*(numberOfClusters+1))/2 - numberOfClusters;

	
	// setting the output
	if(k < numberOfClusters){
		output[k] = true;
	}
	
	
	// setting the shared memory
	for(unsigned int i = 0; i < ceilf((float)numberOfClusters/blockDim.x); i++){
		if(threadIdx.x+blockDim.x*i < numberOfClusters){
			out[threadIdx.x+blockDim.x*i] = 1;	
		}
	}
		
	if(k < numberOfComparisons){
		unsigned int i = numberOfClusters - 2- floorf(sqrtf(-8*k + 4*numberOfClusters*(numberOfClusters-1)-7)/ 2.0 - 0.5);
		unsigned int j = k + i + 1 - numberOfClusters*(numberOfClusters-1)/2 + (numberOfClusters-i)*((numberOfClusters-i)-1)/2;
		bool isDisjoint = true;
		unsigned int blockNr = 0;
		unsigned int currentBlock = 0;
		unsigned int centroidI = centroids[i];
		unsigned int centroidJ = centroids[j];
		unsigned int numberOfBlocks = ceilf((float)dim/32);

		for(unsigned int a = 0; a < dim; a++){
			if(a%32 == 0){
				blockNr = a/32;
				assert(i*numberOfBlocks+blockNr < numberOfBlocks*numberOfClusters);
				assert(j*numberOfBlocks+blockNr < numberOfBlocks*numberOfClusters);
				currentBlock = subspaces[i*numberOfBlocks+blockNr] & subspaces[j*numberOfBlocks+blockNr];
			}
			isDisjoint &= (!(currentBlock >> a%32) & 1) || ((abs(data[centroidI*dim+a] - data[centroidJ*dim+a]) >= 2*width));	
		}
		
		if(isDisjoint){
			//printf("k,i,j: %u, %u, %u are disjoint\n",k, i,j);
			atomicAnd(&out[i], 1);
			atomicAnd(&out[j], 1);
		}else if(scores[i] < scores[j]){
			//printf("k,i,j: %u,  %u, %u; score %f < %f, keeping %u, deleting %u\n",k, i,j,scores[i], scores[j], j,i);
			atomicAnd(&out[i], 0);
			atomicAnd(&out[j], 1);
		}else if(scores[i] == scores[j]){
			//printf("k,i,j: %u, %u, %u; score %f == %f, keeping %u, deleting %u\n",k, i,j,scores[i], scores[j], min(i,j),max(i,j));
			atomicAnd(&out[min(i,j)], 1);
			atomicAnd(&out[max(i,j)], 0);
		}else{
			//printf("k,i,j: %u, %u, %u; score %f > %f, keeping %u, deleting %u\n",k, i,j,scores[i], scores[j], i,j);
			atomicAnd(&out[i], 1);				
			atomicAnd(&out[j], 0);
		}
	}

	
	for(unsigned int i = 0; i < ceilf((float)numberOfClusters/blockDim.x); i++){
		if(threadIdx.x+blockDim.x*i < numberOfClusters){
			atomicAnd(&output[threadIdx.x+blockDim.x*i], out[threadIdx.x+blockDim.x*i]);	
		}
	}
}

void disjointClustersWrapper(unsigned int dimGrid, unsigned int dimBlock, cudaStream_t stream,
							 unsigned int* centroids, float* scores, unsigned int* subspaces,
							 float* data, unsigned int numberOfClusters, unsigned int dim,
							 float width, unsigned int* output){


	unsigned int smem = numberOfClusters*sizeof(unsigned int);
	disjointClusters<<<dimGrid, dimBlock, smem, stream>>>(centroids, scores, subspaces,
														  data, numberOfClusters, dim,
														  width, output);
}

std::vector<bool> disjointClustersTester(std::vector<std::vector<float>*>* data_v, std::vector<unsigned int> centroids_v, std::vector<unsigned int> subspaces_v, std::vector<float> scores_v){
	unsigned int width = 10;
	unsigned int dim = data_v->at(0)->size();
	unsigned int numberOfPoints = data_v->size();
	unsigned int smem = scores_v.size()*sizeof(unsigned int);
	
	float* data;
	float* scores;
	unsigned int* centroids;
	unsigned int* subspaces;
	unsigned int* output;

	unsigned int numberOfComparisons = (scores_v.size()*(scores_v.size()+1))/2-scores_v.size();

	unsigned int dimBlock = 1024;
	unsigned int dimGrid = ceilf((float)numberOfComparisons/dimBlock);
	assert(dimGrid <= 1);

	size_t sizeofData = numberOfPoints*dim*sizeof(float);
	size_t sizeofScores = scores_v.size()*sizeof(float);
	size_t sizeofCentroids = centroids_v.size()*sizeof(unsigned int);
	size_t sizeofSubspaces = subspaces_v.size()*sizeof(unsigned int);
	size_t sizeofOutput = centroids_v.size()*sizeof(unsigned int);


	cudaMallocManaged((void**) &data, sizeofData);
	cudaMallocManaged((void**) &scores, sizeofScores);
	cudaMallocManaged((void**) &centroids, sizeofCentroids);
	cudaMallocManaged((void**) &subspaces, sizeofSubspaces);
	cudaMallocManaged((void**) &output, sizeofOutput);

	for(int i = 0; i < data_v->size(); i++){
		for(int j = 0; j < dim; j++){
			data[i*dim+j] = data_v->at(i)->at(j);	
		}
	}

	for(int i = 0; i < scores_v.size(); i++){
		scores[i] = scores_v.at(i);
	}

	for(int i = 0; i < subspaces_v.size(); i++){
		subspaces[i] = subspaces_v.at(i);
	}

	for(int i = 0; i < centroids_v.size(); i++){
		centroids[i] = centroids_v.at(i);
	}

	disjointClusters<<<dimGrid, dimBlock, smem>>>(centroids, scores, subspaces,
														  data, scores_v.size(), dim,
														  width, output);

	cudaDeviceSynchronize();
	std::vector<bool> output_v;

	for(int i = 0; i < centroids_v.size(); i++){
		output_v.push_back(output[i]);
	}

	cudaFree(data);
	cudaFree(centroids);
	cudaFree(scores);
	cudaFree(subspaces);
	cudaFree(output);
	return output_v;
	
}


__global__ void unsignedIntToBoolArray(unsigned int* input, unsigned int numberOfElements, bool* output){
	unsigned int k = blockIdx.x*blockDim.x+threadIdx.x;
	if(k < numberOfElements){
		output[k] = input[k];
	}
}


void unsignedIntToBoolArrayWrapper(unsigned int dimGrid, unsigned int dimBlock, cudaStream_t stream,
							unsigned int* input, unsigned int numberOfElements, bool* output){
	unsignedIntToBoolArray<<<dimGrid, dimBlock, 0, stream>>>(input, numberOfElements, output);
}


__global__ void copyCentroid(unsigned int* centroids, float* data, unsigned int dim, unsigned int numberOfCentroids, float* centroidsOut){
	unsigned int k = blockIdx.x*blockDim.x+threadIdx.x;
	if(k < dim*numberOfCentroids){
		unsigned int centroidToUse = k/dim;
		unsigned int dimInCentroid = k%dim;
		unsigned int centroidIndex = centroids[centroidToUse];
		centroidsOut[centroidToUse*dim+dimInCentroid] = data[centroidIndex*dim+dimInCentroid];
	}
}

void copyCentroidWrapper(unsigned int dimGrid, unsigned int dimBlock, cudaStream_t stream,
						 unsigned int* centroids, float* data, unsigned int dim,
						 unsigned int numberOfCentroids, float* centroidsOut){
	copyCentroid<<<dimGrid, dimBlock, 0, stream>>>(centroids, data, dim, numberOfCentroids, centroidsOut);
}