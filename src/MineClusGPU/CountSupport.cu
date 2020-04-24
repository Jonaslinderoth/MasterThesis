#include "CountSupport.h"

__device__ __forceinline__ bool  countSupportBlock(unsigned int* items, unsigned int* candidates, size_t itemsIndex, size_t candidateIndex){
	unsigned int itemBlock = items[itemsIndex];
	unsigned int candidateBlock = candidates[candidateIndex];
	unsigned int candidateBlockCount = __popc(candidateBlock);
	unsigned int unionCount = __popc(itemBlock&candidateBlock);
	
	return candidateBlockCount == unionCount;
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
				isSubset &=  countSupportBlock(itemSet, candidates, j*numberOfItems+i, j*numberOfCandidates + candidate);
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


__device__ __forceinline__ unsigned int writeBit(unsigned int value, unsigned int bitnr){
	if(bitnr < 32){
		return value | (1 << (bitnr));
	}else{
		return 0;
	}
}


__global__ void countSupportSharedMemory(unsigned int* candidates, unsigned int* transactions,
										 unsigned int dim, unsigned int numberOfTransactions,
										 unsigned int numberOfCandidates,
										 unsigned int minSupp, float beta,
										 unsigned int* outSupp, float* outScore,
										 bool* outToBeDeleted){

	extern __shared__ unsigned int transactions_s[];

	unsigned int candidates_chunk[6]; // in register allocation

	unsigned int isSubset_r[2];

	unsigned int subspaceCount = 0;
	unsigned int supportCount = 0;
	//unsigned int nPoints = 62;
	//unsigned int chunkSize = 6; // time 32 dimensions



	unsigned int transactionsProcessed = 0;
	unsigned int dimsProcessed = 0; 
	
	unsigned int currentTransactionChunk = 0;
	unsigned int currentTransactionChunkSize = 0;

	unsigned int currentDimChunk = 0;
	unsigned int currentDimChunkSize = 0;

	unsigned int smemUtilised = 0;
	while(transactionsProcessed < numberOfTransactions){
		isSubset_r[0] = 0xffffffff;
		isSubset_r[1] = 0xffffffff;

		// compute the number of transactions each chunk is responsible for
		if(transactionsProcessed+62*32 < numberOfTransactions){
			currentTransactionChunkSize = 62;
			transactionsProcessed += currentTransactionChunkSize*32;
			smemUtilised = 62*32;
		}else{
			unsigned int diff = transactionsProcessed - numberOfTransactions;
			currentTransactionChunkSize = diff/32;
			if(diff%32 < threadIdx.x/32){
				currentTransactionChunkSize++;
				transactionsProcessed += (diff+1)*(threadIdx.x/32);
				
			}else{
				transactionsProcessed += (diff%32) + diff*(threadIdx.x/32);
			}
			smemUtilised = diff;
		}
		
		dimsProcessed = 0;
		while(dimsProcessed < dim){
			if(dimsProcessed+6 < dim){
				currentDimChunkSize = 6;
				dimsProcessed += currentDimChunkSize;
			}else{
				currentDimChunkSize = dimsProcessed+6 - dim;
				dimsProcessed += currentDimChunkSize;
			}
			smemUtilised *= currentDimChunkSize;

			// Load candidate into registers			
			{
				unsigned int id = threadIdx.x%32 + blockIdx.x*32; /*the index of the candidate*/
				if(id < numberOfTransactions){
					for(unsigned int i = 0; i < currentDimChunkSize; i++){ // loads a candidate into registers
						candidates_chunk[i] = candidates[id + (i+dimsProcessed)*numberOfTransactions];
						subspaceCount += __popc(candidates_chunk[i]); // done in all warps, but only used in last warp, only one extra instruction, and one register read.
						candidateIndex += numberOfCandidates;
					}				
				}
			}

			__syncthreads();

			
			// Load transactions into shared memory
			for(unsigned int i = 0; i < ceilf((float)currentTransactionChunkSize/32); i++){
				unsigned int currentWarpPos = threadIdx.x%32;
				unsigned int id = transactionsProcessed + i*32+currentWarpPos;
				for(unsigned int j = 0; j < currentDimChunkSize; j++){
					if(id+j*numberofTransactions < transactionsProcessed+currentTransactionChunkSize){
						transactions_s[i*32+currentWarpPos+j*currentTransactionChunkSize] = transactions[id+j*numberofTransactions];
					}
				}
			}

			
			// for(unsigned int i = 0; i < ceilf((float)nPoints*32*chunkSize/blockDim.x); i++){
			// 	unsigned int j = i*blockDim.x+threadIdx.x; // the if if it were a normal for loop
			// 	unsigned int transactionNumber = (j % (nPoints*32)) +  currentChunk*nPoints*32;
			// 	unsigned int currentDim = (j / (nPoints*32)) + currentDimChunk*chunkSize;
			// 	unsigned int id = currentDim*numberOfTransactions + transactionNumber;
			// 	if((id < numberOfTransactions*ceilf((float)dim/32))  && (j < nPoints*chunkSize*32)){
			// 		transactions_s[j] = transactions[id];
			// 		printf("%u thread: %u, transaction %u value %u id %u transaction number %u, currentDim %u\n",
			// 			   i, threadIdx.x, j, transactions_s[j], id, transactionNumber, currentDim);
			// 	}
			// }

			__syncthreads();
			// Compute the count
			unsigned int temp = 0;
			unsigned int currentBank = threadIdx.x/32;
			for(unsigned int j = 0; j < currentDimChunkSize; j++){
				bool isSubset = true;
				for(unsigned int i = 0; i < currentTransactionChunkSize; i++){
					unsigned int transactionIndex = i+currentBank + j*32*currentTransactionChunkSize;
					if(transactionIndex < smemUtilised){
						isSubset = isSubset && countSupportBlock(transactions_s, candidates_chunk, transactionIndex ,i);
					}
					
				}
				if(isSubset){
					temp = writeBit(temp, i%32);
				}
				if(i != 0 && i%32 == 0 || i == currentTransactionChunkSize-1){
					isSubset_r[i/32] &= temp;
					temp = 0;
				}
			}
			
			// unsigned int temp = 0;
			// for(unsigned int j = 0; j < nPoints; j++){
			// 	bool isSubset = true;
			// 	for(unsigned int i = 0; i < chunkSize; i++){
			// 		unsigned int currentBank = threadIdx.x/32;
			// 		unsigned int transactionIndex = 32*j + currentBank + nPoints*i;
			// 		if(transactionIndex < numberOfTransactions*ceilf((float)dim/32)){
			// 			isSubset = isSubset && countSupportBlock(transactions_s, candidates_chunk, transactionIndex ,i);

			// 		}
			// 	}
			// 	if(isSubset){
			// 		temp = writeBit(temp, j%32);
			// 	}
			// 	if(j != 0 && j%32 == 0 || j == ceilf((float)nPoints/32)-1){
			// 		isSubset_r[j/32] &= temp;
			// 		temp = 0;
			// 	}
			// }
		}
		smemUtilised /= currentDimChunkSize;
		supportCount += __popc(isSubset_r[0]);
		if(currentTransactionChunkSize) >= 32){
			supportCount += __popc(isSubset_r[1]);	
		}
	}
	__syncthreads();
	// Utilise shared memory for the reduction sum, sine we are done using it for the candidates we can simply reuse it
	// Loop unroled, all in a warp should always go in the same condition
	transactions_s[threadIdx.x] = supportCount;
	__syncthreads();
	if((threadIdx.x/32)%2 == 0){
		transactions_s[threadIdx.x] += transactions_s[threadIdx.x+32];
	}
	__syncthreads();
	if((threadIdx.x/32)%4 == 0){
		transactions_s[threadIdx.x] += transactions_s[threadIdx.x+32*2];
	}
	__syncthreads();
	if((threadIdx.x/32)%8 == 0){
		transactions_s[threadIdx.x] += transactions_s[threadIdx.x+32*4];
	}
	__syncthreads();
	if((threadIdx.x/32)%16 == 0){
		transactions_s[threadIdx.x] += transactions_s[threadIdx.x+32*8];
	}
	__syncthreads();
	if((threadIdx.x/32)%32 == 0){
		transactions_s[threadIdx.x] += transactions_s[threadIdx.x+32*16];
	}
	__syncthreads();
	if(threadIdx.x/32 == 0){
		unsigned int i = blockIdx.x*32+threadIdx.x;
		outSupp[i] = transactions_s[threadIdx.x];
		outScore[i] = transactions_s[threadIdx.x]*pow(((float) 1/beta),subspaceCount) ; // calculate score and store
		outToBeDeleted[i] = transactions_s[threadIdx.x] < minSupp;
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
										  unsigned int minSupp, float beta,
										 countSupportType version){
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

	if(version == NaiveCount){
		countSupport<<<dimGrid, dimBlock>>>(candidates_d, itemSet_d, dim, numberOfItems, numberOfCandidates, minSupp, beta, outSupport_d, outScores_d, outToBeDeleted_d);		
	}else if(version == SmemCount){
		dimGrid = ceilf((float)numberOfCandidates/(32));
		unsigned int smemSize = ceilf((float)dim/32) > 6 ? 6 : ceilf((float)dim/32);
		smemSize = smemSize*sizeof(unsigned int)*32*62;
		smemSize = smemSize < dimBlock*sizeof(unsigned int) ? dimBlock*sizeof(unsigned int) : smemSize;
		std::cout << "smsm: " << smemSize << " dimGrid "<< dimGrid << " dim: " << dim << std::endl;
		
		
		countSupportSharedMemory<<<dimGrid, dimBlock,smemSize>>>(candidates_d, itemSet_d, dim, numberOfItems, numberOfCandidates, minSupp, beta, outSupport_d, outScores_d, outToBeDeleted_d);				
	}



	
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
