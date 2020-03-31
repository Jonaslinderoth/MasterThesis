#include "MineClusGPU.h"
#include <stdexcept>   // for exception, runtime_error, out_of_range
#include "../MineClus/MineClusKernels.h"
#include "../DOC_GPU/DOCGPU_Kernels.h"
#include "../randomCudaScripts/DeleteFromArray.h"

MineClusGPU::MineClusGPU(std::vector<std::vector<float>*>* input, float alpha, float beta, float width) {
	this->data = input;
	this->alpha = alpha;
	this->width = width;
	this->beta = beta;
	this->size = data->size();
	this->dim = data->at(0)->size();
}


MineClusGPU::~MineClusGPU() {
	// TODO Auto-generated destructor stub
}


/**
 * Used for loading the data from the data reader into main memory.
 */
std::vector<std::vector<float>*>* MineClusGPU::initDataReader(DataReader* dr){
	auto size = dr->getSize();
	std::vector<std::vector<float>*>* data = new std::vector<std::vector<float>*>(0);
	data->reserve(size);
	while(dr->isThereANextBlock()){
		std::vector<std::vector<float>*>* block = dr->next();
		data->insert(data->end(), block->begin(), block->end());
		delete block;
	}
	return data;
};


/**
 * Allocate space for the dataset, 
 * and transform it from vectors to a single float array.
 */
float* MineClusGPU::transformData(){
	uint size = this->data->size();
	uint dim = this->data->at(0)->size();
	uint size_of_data = size*dim*sizeof(float);
	float* data_h;
	cudaMallocHost((void**) &data_h, size_of_data);
	
	for(int i = 0; i < size; i++){
		for(int j = 0; j < dim; j++){
			data_h[i*dim+j] = data->at(i)->at(j);
		}
	}
	return data_h;
};


/**
 * Find a single cluster, uses the function to find multiple clusters
 */
std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> MineClusGPU::findCluster(){
	auto result = findKClusters(1).at(0);
	return result;
};

/**

   TODO in this function
   Make the centroids run in seperate threads
   Make it possible to make multible clusters
   Free memory after the loop

*/
std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*>> MineClusGPU::findKClusters(int k){
	if(k > 1){
		throw std::runtime_error("Not implemented yet");
	}
	
	auto result = std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*>>();
	// Calculate the "parameters of algorithm"
	unsigned int numberOfPoints = this->data->size();
	unsigned int dim = this->data->at(0)->size();
	unsigned int numberOfBlocksPrPoint = ceilf((float)dim/32);
	unsigned int dimBlock = 1024;
	unsigned int minSupp = size*this->alpha;
	unsigned int numberOfCentroids = (float)2/this->alpha;

	// finding memory sizes
	int smemSize, maxBlock;
	cudaDeviceGetAttribute(&smemSize, 
						   cudaDevAttrMaxSharedMemoryPerBlock, 0);
	cudaDeviceGetAttribute(&maxBlock, 
						   cudaDevAttrMaxThreadsPerBlock, 0); 

	// create streams
	cudaStream_t stream1_1;
	cudaStreamCreate(&stream1_1);
	cudaStream_t stream1_2;
	cudaStreamCreate(&stream1_2);

	// Transfer data the data
	unsigned int sizeOfData = size*dim*sizeof(float);
	float* data_h = this->transformData();
	float* data_d;
	checkCudaErrors(cudaMalloc((void**) &data_d, sizeOfData));
	checkCudaErrors(cudaMemcpyAsync(data_d, data_h, sizeOfData, cudaMemcpyHostToDevice, stream1_2));

	float* bestScore_d;
	unsigned int* bestCentroid_d;
	unsigned int* bestCandidate_d;
	cudaMalloc((void**) &bestScore_d, sizeof(float));
	cudaMalloc((void**) &bestCentroid_d, sizeof(unsigned int));
	cudaMalloc((void**) &bestCandidate_d, numberOfBlocksPrPoint*sizeof(unsigned int));

	// Create long living variables
	unsigned int currentCentroidIndex = 0;
	
	for(unsigned int i = 0; i < numberOfCentroids; i++){
		currentCentroidIndex = this->randInt(0,numberOfPoints-1,1).at(0);
		//std::cout << "centroid: " << currentCentroidIndex << std::endl;
	
		// Create the itemSet
		size_t sizeOfItemSet = numberOfPoints*numberOfBlocksPrPoint*sizeof(unsigned int);
		unsigned int* itemSet_d;
		checkCudaErrors(cudaMalloc((void**) &itemSet_d, sizeOfItemSet));
		createItemSetWrapper(ceilf((float)size/dimBlock), dimBlock, stream1_2, data_d, dim,
							 numberOfPoints, currentCentroidIndex, this->width, itemSet_d); 


		// Create the initial candidates
		size_t sizeOfCandidates = dim*numberOfBlocksPrPoint*sizeof(unsigned int);
		unsigned int* candidates_d;
		checkCudaErrors(cudaMalloc((void**) &candidates_d, sizeOfCandidates));
		createInitialCandidatesWrapper(ceilf((float)dim/dimBlock), dimBlock, stream1_1, dim, candidates_d);


	
		// Count the support of the initial candidates
		size_t sizeOfSupport = dim*sizeof(unsigned int);
		size_t sizeOfScore = dim*sizeof(float);
		size_t sizeOfToBeDeleted = (dim+1)*sizeof(bool);

		unsigned int* support_d;
		float* score_d;
		bool* toBeDeleted_d;

		checkCudaErrors(cudaMalloc((void**) &support_d, sizeOfSupport));
		checkCudaErrors(cudaMalloc((void**) &score_d, sizeOfScore));
		checkCudaErrors(cudaMalloc((void**) &toBeDeleted_d, sizeOfToBeDeleted));

		// Synchronize to make sure candidates and item set are donee
		checkCudaErrors(cudaStreamSynchronize(stream1_1));
		checkCudaErrors(cudaStreamSynchronize(stream1_2));
		countSupportWrapper(ceilf((float)dim/dimBlock), dimBlock, stream1_1,
							candidates_d, itemSet_d, dim, numberOfPoints, dim, minSupp,
							this->beta, support_d, score_d, toBeDeleted_d);



		// Create the PrefixSum, and find the number of elements left 
		size_t sizeOfPrefixSum = (dim+1)*sizeof(unsigned int);
		unsigned int* prefixSum_d;
		checkCudaErrors(cudaMalloc((void**) &prefixSum_d, sizeOfPrefixSum));
		sum_scan_blelloch(stream1_1, prefixSum_d,toBeDeleted_d,(dim+1), false);

		// Find the number of candidates after deletion of the small candidates
		unsigned int* sum_h;
		checkCudaErrors(cudaMallocHost((void**) &sum_h, sizeof(unsigned int)));
		checkCudaErrors(cudaMemcpyAsync(sum_h, prefixSum_d+dim, sizeof(unsigned int), cudaMemcpyDeviceToHost,stream1_1));
		checkCudaErrors(cudaStreamSynchronize(stream1_1));
		unsigned int oldNumberOfCandidates = dim;	
		unsigned int numberOfCandidates = dim-sum_h[0];
		
		//std::cout << "numberOfCandidates: " << numberOfCandidates << std::endl;
		//if there are any candidates left
		if(numberOfCandidates > 0){
			
			// delete all the small candidates
			unsigned int* newCandidates_d;
			sizeOfCandidates = dim*numberOfBlocksPrPoint*sizeof(unsigned int);
			checkCudaErrors(cudaMalloc((void**) &newCandidates_d, sizeOfCandidates));
			deleteFromArrayTransfomedDataWrapper(ceilf((float)dim/32), dimBlock, stream1_1,
												 candidates_d, prefixSum_d, oldNumberOfCandidates, numberOfBlocksPrPoint, newCandidates_d);

			// delete the scores that matches the deleted candidates			
			float* newScore_d;
			sizeOfScore = dim*sizeof(unsigned int);
			checkCudaErrors(cudaMalloc((void**) &newScore_d, sizeOfScore));
			deleteFromArrayTransfomedDataWrapper(ceilf((float)dim/32), dimBlock, stream1_1, score_d, prefixSum_d, oldNumberOfCandidates, 1, newScore_d);

			// free all the memory that is not needed any more
			checkCudaErrors(cudaFree(prefixSum_d));
			checkCudaErrors(cudaFree(candidates_d));
			checkCudaErrors(cudaFree(score_d));
			checkCudaErrors(cudaFree(support_d));
			checkCudaErrors(cudaFree(toBeDeleted_d));
			
			candidates_d = newCandidates_d;
			score_d = newScore_d;


			// find the current max, and save it for later
			unsigned int* index_d;
			checkCudaErrors(cudaMalloc((void **) &index_d, numberOfCandidates*sizeof(unsigned int)));
			createIndicesKernel(ceilf((float)numberOfCandidates/dimBlock), dimBlock, stream1_1, index_d, numberOfCandidates);
			argMaxKernel(ceilf((float)numberOfCandidates/dimBlock), dimBlock, smemSize ,stream1_1, score_d, index_d, numberOfCandidates);

			// If it is the first iteration set the current best score to 0
			float* bestScore_h;
			if(i == 0){
				cudaMallocHost((void**) &bestScore_h, sizeof(float));
				bestScore_h[0] = 0;
				cudaMemcpy(bestScore_d, bestScore_h, sizeof(float), cudaMemcpyHostToDevice);				
			}

			// find the best score, then copy that score and the coresponding candidate and centroid to the result
			extractMaxWrapper(ceilf((float)numberOfBlocksPrPoint/dimBlock), dimBlock, stream1_1,
							  candidates_d, score_d, currentCentroidIndex, numberOfCandidates, index_d,
							  dim, bestCandidate_d, bestScore_d, bestCentroid_d);

		
			/**** Only for debugging ***/
			// float* score_h;
			// checkCudaErrors(cudaMallocHost((void **) &score_h, 4));
			// cudaMemcpyAsync(score_h, score_d, sizeof(float), cudaMemcpyDeviceToHost, stream1_1);
			// cudaStreamSynchronize(stream1_1);
			// std::cout << "curret best: " << score_h[0] << std::endl;
			/***************************/
			
			checkCudaErrors(cudaFree(score_d)); // The score vector is not be needed any more, the candidates are.
		
			unsigned int iterationNr = 1;
			//std::cout << "iteration number: " << iterationNr << std::endl;
			while(numberOfCandidates > 1 && iterationNr <= dim+1){
				iterationNr++;

				// Merge candidates
				oldNumberOfCandidates = numberOfCandidates;
				numberOfCandidates = (numberOfCandidates*(numberOfCandidates+1))/2-numberOfCandidates;
				//std::cout << "iteration number: " << iterationNr << " numberOfCandidates: " << numberOfCandidates << std::endl;
				sizeOfCandidates = (numberOfCandidates)*numberOfBlocksPrPoint * sizeof(unsigned int);
				//std::cout << "number of candidates at beginning of loop: " << numberOfCandidates<< " numberOfBlocksPrPoint: " << numberOfBlocksPrPoint << std::endl;
				unsigned int* newCandidates_d;

				bool* deletedFromCount_d;
				sizeOfToBeDeleted = (numberOfCandidates+1)*sizeof(bool);
				checkCudaErrors(cudaMalloc((void**) &deletedFromCount_d, sizeOfToBeDeleted));
				
				checkCudaErrors(cudaMalloc((void**) &newCandidates_d, sizeOfCandidates));
				
				mergeCandidatesWrapper(ceilf((float)numberOfCandidates/dimBlock), dimBlock, stream1_1,
									   candidates_d, oldNumberOfCandidates, dim, iterationNr, newCandidates_d, deletedFromCount_d);

				checkCudaErrors(cudaFree(candidates_d)); // delete the old candidate array
				candidates_d = newCandidates_d;


				checkCudaErrors(cudaMalloc((void**) &toBeDeleted_d, sizeOfToBeDeleted));

				/**** Only for debugging ***/
				// {
				// 	unsigned int* candidates_h;
				// 	cudaMallocHost((void**) &candidates_h, numberOfCandidates*sizeof(unsigned int)*numberOfBlocksPrPoint);
				// 	cudaMemcpyAsync(candidates_h, candidates_d, numberOfCandidates*sizeof(unsigned int)*numberOfBlocksPrPoint, cudaMemcpyDeviceToHost, stream1_1);
				// 	checkCudaErrors(cudaStreamSynchronize(stream1_1));
				// 	for(int i = 0; i< numberOfCandidates; i++){
				// 		std::cout << i << ": ";
				// 		for(int j = 0; j < numberOfBlocksPrPoint; j++){
				// 			std::cout << candidates_h[j*numberOfCandidates+i] << ", ";
				// 		}
				// 		std::cout << std::endl;
				// 	}
				// }
				/***************************/

				
				// Find all the dublicate candidates 
				checkCudaErrors(cudaMemsetAsync(toBeDeleted_d, 0, sizeOfToBeDeleted, stream1_1)); // not needed any more, it is handled by the merge kernel now
				findDublicatesWrapper(ceilf((float)numberOfCandidates/dimBlock), dimBlock, stream1_1, candidates_d, numberOfCandidates, dim, deletedFromCount_d, toBeDeleted_d, Breaking);
				andKernelWrapper(ceilf((float)numberOfCandidates+1/dimBlock), dimBlock, stream1_1, numberOfCandidates+1, toBeDeleted_d, deletedFromCount_d);
				/**** Only for debugging ***/
				// bool* toBeDeleted_h;
				// cudaMallocHost((void**) &toBeDeleted_h, sizeOfToBeDeleted);
				// cudaMemcpyAsync(toBeDeleted_h, toBeDeleted_d, sizeOfToBeDeleted, cudaMemcpyDeviceToHost, stream1_1);
				// checkCudaErrors(cudaStreamSynchronize(stream1_1));		
				// for(int i = 0; i< numberOfCandidates; i++){
				// 	std::cout << i << ": ";
				// 	std::cout << toBeDeleted_h[i] << ", ";
				// 	std::cout << std::endl;
				// }

				/***************************/

				// Compute the prefix sum and then find the number of dublicates
				sizeOfPrefixSum = (numberOfCandidates+1)*sizeof(unsigned int);
				checkCudaErrors(cudaMalloc((void**) &prefixSum_d, sizeOfPrefixSum));
				sum_scan_blelloch(stream1_1, prefixSum_d,toBeDeleted_d,(numberOfCandidates+1), false);


				/**** Only for debugging ***/
				// {
				// 	unsigned int* prefixSum_h;
				// 	cudaMallocHost((void**) &prefixSum_h, numberOfCandidates*sizeof(unsigned int));
				// 	checkCudaErrors(cudaMemcpyAsync(prefixSum_h, prefixSum_d, numberOfCandidates*sizeof(unsigned int), cudaMemcpyDeviceToHost,stream1_1));
				// 	checkCudaErrors(cudaStreamSynchronize(stream1_1));
				// 	std::cout << "prefixSum: ";
				// 	for(int i = 0; i < numberOfCandidates; i++){
				// 		std::cout << prefixSum_h[i] << ", ";
				// 	}
				// 	std::cout << std::endl;
				// }
				/***************************/


				
				checkCudaErrors(cudaMemcpyAsync(sum_h, prefixSum_d+numberOfCandidates, sizeof(unsigned int), cudaMemcpyDeviceToHost,stream1_1));
				checkCudaErrors(cudaStreamSynchronize(stream1_1));
				assert(numberOfCandidates >= sum_h[0]); // avoid underflow
				
				// delete the dublicate candidates
				checkCudaErrors(cudaStreamSynchronize(stream1_1));
				oldNumberOfCandidates = numberOfCandidates;
				numberOfCandidates = numberOfCandidates-sum_h[0];
				//std::cout << "number of candidates after deleted dublicates: " << numberOfCandidates << std::endl;
				if(numberOfCandidates <= 0){
					//	std::cout << "no more candidates, dublicates removed all" << std::endl;
					break;
				}
				// delete candidates
				sizeOfCandidates = numberOfCandidates*numberOfBlocksPrPoint*sizeof(unsigned int);
				checkCudaErrors(cudaMalloc((void**) &newCandidates_d, sizeOfCandidates));
				deleteFromArrayTransfomedDataWrapper(ceilf(((float)oldNumberOfCandidates*numberOfBlocksPrPoint)/dimBlock), dimBlock, stream1_1,
													 candidates_d, prefixSum_d, oldNumberOfCandidates, numberOfBlocksPrPoint, newCandidates_d);
				// delete matching scores, there are no scores yet, only after count support
				// sizeOfScore = numberOfCandidates*sizeof(float);
				// checkCudaErrors(cudaMalloc((void**) &newScore_d, sizeOfScore));
				// deleteFromArrayTransfomedDataWrapper(ceilf((float)oldNumberOfCandidates/32), dimBlock, stream1_1,
				// 									 score_d, prefixSum_d, oldNumberOfCandidates, 1, newScore_d);
				// clean up after the deletion
				checkCudaErrors(cudaFree(candidates_d));
				checkCudaErrors(cudaFree(toBeDeleted_d));
				checkCudaErrors(cudaFree(prefixSum_d));
				
				candidates_d = newCandidates_d;
				score_d = newScore_d;

				/**** Only for debugging ***/
				//{
				// 	unsigned int* candidates_h;
				// 	cudaMallocHost((void**) &candidates_h, numberOfCandidates*sizeof(unsigned int)*numberOfBlocksPrPoint);
				// 	cudaMemcpyAsync(candidates_h, newCandidates_d, numberOfCandidates*sizeof(unsigned int)*numberOfBlocksPrPoint, cudaMemcpyDeviceToHost, stream1_1);
				// 	checkCudaErrors(cudaStreamSynchronize(stream1_1));
				// 	for(int i = 0; i< numberOfCandidates; i++){
				// 		std::cout << "after dublicates are removed: " << i << ": ";
				// 		for(int j = 0; j < numberOfBlocksPrPoint; j++){
				// 			std::cout << candidates_h[j*numberOfCandidates+i] << ", ";
				// 		}
				// 		std::cout << std::endl;
				// 	}
				// }
				/***************************/
				// Count support
				sizeOfSupport = numberOfCandidates*sizeof(unsigned int);
				sizeOfScore = numberOfCandidates*sizeof(float);
				sizeOfToBeDeleted = (numberOfCandidates+1)*sizeof(bool);

				checkCudaErrors(cudaMalloc((void**) &support_d, sizeOfSupport));
				checkCudaErrors(cudaMalloc((void**) &score_d, sizeOfScore));
				checkCudaErrors(cudaMalloc((void**) &toBeDeleted_d, sizeOfToBeDeleted));

				countSupportWrapper(ceilf((float)numberOfCandidates/dimBlock), dimBlock, stream1_1,
									candidates_d, itemSet_d, dim, numberOfPoints, numberOfCandidates, minSupp,
									this->beta, support_d, score_d, toBeDeleted_d);
				/**** Only for debugging ***/
				// {
				// 	bool* toBeDeleted_h;
				// 	cudaMallocHost((void**) &toBeDeleted_h, (numberOfCandidates+1)*sizeof(bool));
				// 	checkCudaErrors(cudaMemcpyAsync(toBeDeleted_h, toBeDeleted_d, (numberOfCandidates+1)*sizeof(bool), cudaMemcpyDeviceToHost,stream1_1));
				// 	checkCudaErrors(cudaStreamSynchronize(stream1_1));
				// 	std::cout << "toBeDeleted: ";
				// 	for(int i = 0; i < numberOfCandidates+1; i++){
				// 		std::cout << toBeDeleted_h[i] << ", ";
				// 	}
				// 	std::cout << std::endl;
				// }
				/***************************/

				
				// delete small candidates
				sizeOfPrefixSum = (numberOfCandidates+1)*sizeof(unsigned int);
				checkCudaErrors(cudaMalloc((void**) &prefixSum_d, sizeOfPrefixSum));
				sum_scan_blelloch(stream1_1, prefixSum_d, toBeDeleted_d,(numberOfCandidates+1), false);
				checkCudaErrors(cudaMemcpyAsync(sum_h, prefixSum_d+numberOfCandidates, sizeof(unsigned int), cudaMemcpyDeviceToHost,stream1_1));

				/**** Only for debugging ***/
				// {
				// 	unsigned int* prefixSum_h;
				// 	cudaMallocHost((void**) &prefixSum_h, numberOfCandidates*sizeof(unsigned int));
				// 	checkCudaErrors(cudaMemcpyAsync(prefixSum_h, prefixSum_d, numberOfCandidates*sizeof(unsigned int), cudaMemcpyDeviceToHost,stream1_1));
				// 	checkCudaErrors(cudaStreamSynchronize(stream1_1));
				// 	std::cout << "prefixSum: ";
				// 	for(int i = 0; i < numberOfCandidates; i++){
				// 		std::cout << prefixSum_h[i] << ", ";
				// 	}
				// 	std::cout << std::endl;
				// }
				/***************************/



				
				checkCudaErrors(cudaStreamSynchronize(stream1_1));
				oldNumberOfCandidates = numberOfCandidates;

				assert(numberOfCandidates >= sum_h[0]); // avoid underflow
				numberOfCandidates = numberOfCandidates-sum_h[0];
				if(numberOfCandidates <= 0){
					//std::cout << "no more candidates after small candidates are deleted" << std::endl;
					break;
				}


				sizeOfCandidates = numberOfCandidates*numberOfBlocksPrPoint*sizeof(unsigned int);
				sizeOfScore = numberOfCandidates*sizeof(float);

				checkCudaErrors(cudaMalloc((void**) &newCandidates_d, sizeOfCandidates));
				checkCudaErrors(cudaMalloc((void**) &newScore_d, sizeOfScore));
				
				deleteFromArrayTransfomedDataWrapper(ceilf((float)oldNumberOfCandidates*numberOfBlocksPrPoint/dimBlock), dimBlock, stream1_1,
													 candidates_d, prefixSum_d, oldNumberOfCandidates, numberOfBlocksPrPoint, newCandidates_d);
				deleteFromArrayTransfomedDataWrapper(ceilf((float)oldNumberOfCandidates*numberOfBlocksPrPoint/dimBlock), dimBlock, stream1_1,
													 score_d, prefixSum_d, oldNumberOfCandidates, 1, newScore_d);


				checkCudaErrors(cudaStreamSynchronize(stream1_1));
				checkCudaErrors(cudaFree(support_d));
				checkCudaErrors(cudaFree(score_d));
				checkCudaErrors(cudaFree(toBeDeleted_d));
				checkCudaErrors(cudaFree(candidates_d));
				checkCudaErrors(cudaFree(prefixSum_d));
				candidates_d = newCandidates_d;
				score_d = newScore_d;


		
				// find top k (top 1 as a beginning)
				checkCudaErrors(cudaMalloc((void **) &index_d, numberOfCandidates*sizeof(unsigned int)));
				createIndicesKernel(ceilf((float)numberOfCandidates/dimBlock), dimBlock, stream1_1, index_d, numberOfCandidates);
				argMaxKernel(ceilf((float)numberOfCandidates/dimBlock), dimBlock, smemSize ,stream1_1, score_d, index_d, numberOfCandidates);


				extractMaxWrapper(ceilf((float)numberOfBlocksPrPoint/dimBlock), dimBlock, stream1_1,
								  candidates_d, score_d, currentCentroidIndex, numberOfCandidates, index_d,
								  dim, bestCandidate_d, bestScore_d, bestCentroid_d);

				/**** Only for debugging ***/
				// {
				// 	float* score_h;
				// 	checkCudaErrors(cudaMallocHost((void**) &score_h, sizeof(float)));
				// 	checkCudaErrors(cudaMemcpyAsync(score_h, score_d, 4, cudaMemcpyDeviceToHost, stream1_1));
				// 	checkCudaErrors(cudaStreamSynchronize(stream1_1));		
				// 	std::cout << "candidate score: " << score_h[0] << std::endl;
				// }

				/**** Only for debugging ***/
				// {
				// 	float* score_h;
				// 	checkCudaErrors(cudaMallocHost((void**) &score_h, sizeof(float)));
				// 	checkCudaErrors(cudaMemcpyAsync(score_h, bestScore_d, 4, cudaMemcpyDeviceToHost, stream1_1));
				// 	checkCudaErrors(cudaStreamSynchronize(stream1_1));		
				// 	std::cout << "curret best: " << score_h[0] << std::endl;
				// }
				/**************************/
				checkCudaErrors(cudaFree(index_d));
				checkCudaErrors(cudaFree(score_d));
			}
		}
	}

	bool* pointsContained_d;
	cudaMalloc((void**) &pointsContained_d, (numberOfPoints+1)*sizeof(bool));
		
	findPointInClusterWrapper(ceilf((float)numberOfPoints/dimBlock), dimBlock, stream1_1,
							  bestCandidate_d, data_d, bestCentroid_d, dim,
							  numberOfPoints, width, pointsContained_d);
		
	checkCudaErrors(cudaStreamSynchronize(stream1_1));

	/**** Only for debugging ***/
	// bool* pointsContained_h;
	// checkCudaErrors(cudaMallocHost((void**) &pointsContained_h, numberOfPoints*sizeof(bool)));
	// checkCudaErrors(cudaMemcpyAsync(pointsContained_h, pointsContained_d, numberOfPoints*sizeof(bool), cudaMemcpyDeviceToHost, stream1_1));
	// checkCudaErrors(cudaStreamSynchronize(stream1_1));
		
	// std::cout << "points contained: ";
	// for(int i = 0; i < numberOfPoints; i++){
	// 	std::cout<< pointsContained_h[i] << ", ";
	// }
	// std::cout << std::endl;
	/**** Only for debugging ***/
		

	
	unsigned int* prefixSum_d;
	size_t sizeOfPrefixSum = (numberOfPoints+1)*sizeof(unsigned int);
	checkCudaErrors(cudaMalloc((void**) &prefixSum_d, sizeOfPrefixSum));

	sum_scan_blelloch(stream1_1, prefixSum_d,pointsContained_d,(numberOfPoints+1), true);


	unsigned int* sum2_h;
	unsigned int* sum_h;
	checkCudaErrors(cudaMallocHost((void**) &sum_h, sizeof(unsigned int)));
	checkCudaErrors(cudaMallocHost((void**) &sum2_h, 22*sizeof(unsigned int)));
	//checkCudaErrors(cudaMemcpyAsync(sum2_h, prefixSum_d, 22*sizeof(unsigned int), cudaMemcpyDeviceToHost,stream1_1));
		
	checkCudaErrors(cudaMemcpyAsync(sum_h, prefixSum_d+numberOfPoints, sizeof(unsigned int), cudaMemcpyDeviceToHost,stream1_1));
	checkCudaErrors(cudaStreamSynchronize(stream1_1));
	unsigned int oldNumberOfPoints = numberOfPoints;
	numberOfPoints = sum_h[0];


	//std::cout << "prefix sum: ";
	//for(unsigned int i = 0; i < 21; i++){
	//	std::cout << sum2_h[i] << ", ";
	//}
	//std::cout << std::endl;


	float* outputCluster_d;
	float* outputCluster_h;
	float* newData_d;

	size_t sizeOfNewData = numberOfPoints*dim*sizeof(unsigned int);
	size_t sizeOfOutputCluster = (oldNumberOfPoints - sum_h[0])*dim*sizeof(unsigned int);
		

	checkCudaErrors(cudaMalloc((void**) &outputCluster_d, sizeOfOutputCluster));
		
	checkCudaErrors(cudaStreamSynchronize(stream1_1));

	deleteFromArrayWrapper(ceilf((float)(oldNumberOfPoints*dim)/dimBlock), dimBlock, stream1_1, data_d, prefixSum_d, oldNumberOfPoints, dim, outputCluster_d);
		
	checkCudaErrors(cudaStreamSynchronize(stream1_1));
		
		
	sum_scan_blelloch(stream1_1, prefixSum_d,pointsContained_d,(oldNumberOfPoints+1), false);	
	checkCudaErrors(cudaMalloc((void**) &newData_d, sizeOfNewData));
	deleteFromArrayWrapper(ceilf((float)oldNumberOfPoints*dim/dimBlock),  dimBlock, stream1_1, data_d, prefixSum_d, oldNumberOfPoints, dim, newData_d);

	checkCudaErrors(cudaMallocHost((void**) &outputCluster_h, sizeOfOutputCluster));
	cudaMemcpyAsync(outputCluster_h, outputCluster_d, sizeOfOutputCluster, cudaMemcpyDeviceToHost, stream1_1);

	unsigned int* bestCandidate_h;
	checkCudaErrors(cudaMallocHost((void**) &bestCandidate_h, numberOfBlocksPrPoint*sizeof(unsigned int)));
	checkCudaErrors(cudaMemcpyAsync(bestCandidate_h, bestCandidate_d, numberOfBlocksPrPoint*sizeof(unsigned int), cudaMemcpyDeviceToHost, stream1_1));
	
	checkCudaErrors(cudaStreamSynchronize(stream1_1));
		
	auto outputCluster = new std::vector<std::vector<float>*>;
	for(unsigned int i = 0; i < oldNumberOfPoints - sum_h[0]; i++){
		auto point = new std::vector<float>;
		for(unsigned int j = 0; j < dim; j++){
			point->push_back(outputCluster_h[i*dim+j]);
		}
		outputCluster->push_back(point);
	}

	auto outputDim = new std::vector<bool>;
	unsigned int count = 0;
	for(unsigned int i = 0; i < numberOfBlocksPrPoint;i++){
		for(unsigned int j = 0; j < 32; j++){
			outputDim->push_back((bestCandidate_h[i] >> j) & (unsigned int)1);
			//		std::cout << ((bestCandidate_h[i] >> j) & (unsigned int)1);
			count++;
			if(count >= dim){
				break;
			}
		}
	}
	//std::cout << std::endl;

	result.push_back(std::make_pair(outputCluster,outputDim));
	checkCudaErrors(cudaStreamSynchronize(stream1_2));
	cudaFree(data_d);
	data_d = newData_d;

	cudaStreamDestroy(stream1_1);
	cudaStreamDestroy(stream1_2);


	return result;

	//throw std::runtime_error("Not implemented Yet");
	
	/*
	  create itemSet
	  create initial candidates
	  count support
	  delete small clusters
	  for i < k:
	  Merge candidates
	  remove dublicates
	  Count support
	  delete small candidates
	  find top k (top 1 as a beginning)

	  kernels to be made:
	  - createItemSet
	  - createInitialCandidates
	  - countSupport
	  - mergeCandidates
	  - removeDublicates



	  createItemSet
	  - arguments: 
	  data: the "raw" dataset
	  dim: number of dimensions
	  numberOfPoints: number of points in data set 
	  output: ceil(dim/32)*numberOfPoint unsigned int array. 
	  centroid: id of the centroid
	  width: width of hypercube
	  - Description: create a boolean vector for all centroids and all points 
	  with the dimmensions they are within width of each other. 
	  IMPORTANT: the layout of the item set should be the first 32 dimensions of all points fist, 
	  then the next, so on to make the reads coaceled coaceled
	     
 
	  createInitialCandidates
	  - arguments: 
	  dim: number of dims
	  output: ceil(dim/32)*dim array unsigned int 
	  - Description: creates all candidates with one dimension. 
	  Store it in the same way as the itemSet.

	  countSupport
	  - Arguments: 
	  Candidates: array of candidates
	  dim: number of dimensions
	  itemsSet: array of itemSet
	  numberOfItems: number of item in the itemSet. 
	  numberOfCandidates: numberOfDandidates
	  minSupp: minimum support 
	  alpha: 
	  beta: 
	  outSupp: array of unsigned ints to store the support
	  outScore: array of floats to store the score
	  outToBeDeleted: array of bools to use for deleteions
	 
	  MergeCandidates: 
	  - Arguments: 
	  candidates: array of candidates
	  dim: dimension of candidates
	  numberOfCandidates: number of candidates
	  outNumberOfCandidates: number of new candidates
	  output: array for the new candidates should be of size (outNumberOfCandidates)*ceil(dim/32)


	  CountSupport

	
	*/
	
}
