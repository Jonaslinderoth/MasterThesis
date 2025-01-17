#include "MineClusGPUnified.h"
#include <stdexcept>   // for exception, runtime_error, out_of_range
#include "MineClusKernels.h"
#include "CountSupport.h"
#include "CreateTransactions.h"
#include "FindDublicates.h"
#include "../DOC_GPU/DOCGPU_Kernels.h"
#include "../DOC_GPU/ArgMax.h"
#include "MergeCandidates.h"
#include "../randomCudaScripts/DeleteFromArray.h"
#include <algorithm>

MineClusGPUnified::MineClusGPUnified(std::vector<std::vector<float>*>* input, float alpha, float beta, float width) {
	this->data = input;
	this->alpha = alpha;
	this->width = width;
	this->beta = beta;
	this->size = data->size();
	this->dim = data->at(0)->size();
}


MineClusGPUnified::~MineClusGPUnified() {
	// TODO Auto-generated destructor stub
}


/**
 * Used for loading the data from the data reader into main memory.
 */
std::vector<std::vector<float>*>* MineClusGPUnified::initDataReader(DataReader* dr){
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
float* MineClusGPUnified::transformData(){
	unsigned int size = this->data->size();
	unsigned int dim = this->data->at(0)->size();
	size_t size_of_data = size*dim*sizeof(float);
	float* data_d;
	checkCudaErrors(cudaMallocManaged((void**) &data_d, size_of_data));
	checkCudaErrors(cudaMemPrefetchAsync(data_d, size_of_data, cudaCpuDeviceId, NULL));
	
	for(unsigned int i = 0; i < size; i++){
		for(unsigned int j = 0; j < dim; j++){
			
			data_d[(size_t)i*dim+j] = data->at(i)->at(j);
		}
	}
	return data_d;
};

/**
 * Find a single cluster, uses the function to find multiple clusters
 */
std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> MineClusGPUnified::findCluster(){
	auto result = findKClusters(1);
	if (result.size() == 0){
		return std::make_pair(new std::vector<std::vector<float>*>, new std::vector<bool>);
	}else{
		return result.at(0);	
	}
};



/**
   Finds multiple clusters
   TODO in this function
   Free memory after the loop
*/
std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*>> MineClusGPUnified::findKClusters(int k){

	// Create the final output
	auto result = std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*>>();

	// Calculate the "parameters of algorithm"
	unsigned int numberOfPoints = this->data->size();
	unsigned int dim = this->data->at(0)->size();
	unsigned int numberOfBlocksPrPoint = ceilf((float)dim/32);
	unsigned int dimBlock = 1024;
	unsigned int minSupp = size*this->alpha;
	unsigned int numberOfCentroids = (float)2/this->alpha;

	// finding memory sizes
	int device = -1;
	cudaGetDevice(&device);
	int smemSize, maxBlock;
	cudaDeviceGetAttribute(&smemSize, 
						   cudaDevAttrMaxSharedMemoryPerBlock, device);
	cudaDeviceGetAttribute(&maxBlock, 
						   cudaDevAttrMaxThreadsPerBlock, device);

	// create streams
	cudaStream_t stream1_1;
	checkCudaErrors(cudaStreamCreate(&stream1_1));
	cudaStream_t stream1_2;
	checkCudaErrors(cudaStreamCreate(&stream1_2));

	// Transfer data the data
	size_t sizeOfData = size*dim*sizeof(float);
	float* data_d = this->transformData();

	// This loop is running until all clusters are found
	while(k > result.size()){
		// std::cout << result.size() << std::endl;
		
		// Allocate the space for the best clusters for all centroids
		float* bestScore_d;
		unsigned int* bestCentroid_d;
		unsigned int* bestCandidate_d;
		checkCudaErrors(cudaMallocManaged((void**) &bestScore_d,  numberOfCentroids*sizeof(float)));
		checkCudaErrors(cudaMallocManaged((void**) &bestCentroid_d,  numberOfCentroids*sizeof(unsigned int)));
		checkCudaErrors(cudaMallocManaged((void**) &bestCandidate_d,
										  numberOfCentroids*numberOfBlocksPrPoint*sizeof(unsigned int)));


		// If it is the first iteration set the current best score to 0 for all future centroids too
		checkCudaErrors(cudaMemPrefetchAsync(bestScore_d, numberOfCentroids*sizeof(float), cudaCpuDeviceId, stream1_1));
		checkCudaErrors(cudaStreamSynchronize(stream1_1));
		for(int j = 0; j < numberOfCentroids; j++){
			bestScore_d[j] = 0;
		}
		
		// Create long living variables
		unsigned int currentCentroidIndex = 0;

		// Go through all centroids.
		for(unsigned int i = 0; i < numberOfCentroids; i++){

			// The current centroid index, so here the random index is generated by host, because it is few.
			currentCentroidIndex = this->randInt(0,numberOfPoints-1,1).at(0);

			// The itemSet and the initial candidates can be created concurently,
			// hereby they use two different streams			
			// Create the itemSet
			size_t sizeOfItemSet = numberOfPoints*numberOfBlocksPrPoint*sizeof(unsigned int);
			unsigned int* itemSet_d;
			checkCudaErrors(cudaMallocManaged((void**) &itemSet_d,  sizeOfItemSet));
			checkCudaErrors(cudaMemPrefetchAsync(itemSet_d, sizeOfItemSet, device, stream1_2));
			checkCudaErrors(cudaMemPrefetchAsync(data_d, numberOfPoints*dim*sizeof(float), device, stream1_2));

			
			createTransactionsWrapper(ceilf((float)size/dimBlock), dimBlock, smemSize ,stream1_2, data_d, dim,
								 numberOfPoints, currentCentroidIndex, this->width, itemSet_d);


			// Create the initial candidates
			size_t sizeOfCandidates = dim*numberOfBlocksPrPoint*sizeof(unsigned int);
			unsigned int* candidates_d;
			checkCudaErrors(cudaMallocManaged((void**) &candidates_d,  sizeOfCandidates));
			checkCudaErrors(cudaMemPrefetchAsync(candidates_d, sizeOfCandidates, device, stream1_1));
			createInitialCandidatesWrapper(ceilf((float)dim/dimBlock), dimBlock, stream1_1,
										   dim, candidates_d);


			// Allocate the arrays for the support counting, scores,
			// and the bool arrays of elements to be deleted
			size_t sizeOfSupport = dim*sizeof(unsigned int);
			size_t sizeOfScore = dim*sizeof(float);
			size_t sizeOfToBeDeleted = (dim+1)*sizeof(bool);

			unsigned int* support_d;
			float* score_d;
			bool* toBeDeleted_d;

			checkCudaErrors(cudaMallocManaged((void**) &support_d,   sizeOfSupport));
			checkCudaErrors(cudaMallocManaged((void**) &score_d,   sizeOfScore));
			checkCudaErrors(cudaMallocManaged((void**) &toBeDeleted_d,   sizeOfToBeDeleted));

			// Synchronize to make sure candidates and item set are donee
			checkCudaErrors(cudaStreamSynchronize(stream1_1));
			checkCudaErrors(cudaStreamSynchronize(stream1_2));

			// Count the support
			checkCudaErrors(cudaMemPrefetchAsync(itemSet_d, sizeOfItemSet, device, stream1_1));
			checkCudaErrors(cudaMemPrefetchAsync(candidates_d, sizeOfCandidates, device, stream1_1));
			checkCudaErrors(cudaMemPrefetchAsync(score_d, sizeOfScore, device, stream1_1));
			countSupportWrapper(ceilf((float)dim/dimBlock), dimBlock, stream1_1,
								candidates_d, itemSet_d, dim, numberOfPoints, dim, minSupp,
								this->beta, support_d, score_d, toBeDeleted_d, this->countSupportKernelVersion);

			// Create the PrefixSum, and find the number of elements left
			// The number elements to be deleted is the last value in the prefixSum
			size_t sizeOfPrefixSum = (dim+1)*sizeof(unsigned int); // +1 because of the prefixSum
			unsigned int* prefixSum_d;
			checkCudaErrors(cudaMallocManaged((void**) &prefixSum_d,   sizeOfPrefixSum));

			checkCudaErrors(cudaMemPrefetchAsync(prefixSum_d, sizeOfPrefixSum, device, stream1_1));
			checkCudaErrors(cudaMemPrefetchAsync(toBeDeleted_d, sizeOfToBeDeleted, device, stream1_1));
			sum_scan_blelloch_managed(stream1_1,stream1_1, prefixSum_d,toBeDeleted_d,(dim+1), false);


			// Find the number of candidates after deletion of the small candidates
			checkCudaErrors(cudaMemPrefetchAsync(prefixSum_d+dim, sizeof(unsigned int), cudaCpuDeviceId, stream1_1));			
			checkCudaErrors(cudaStreamSynchronize(stream1_1));
			size_t oldNumberOfCandidates = dim;	
			size_t numberOfCandidates = dim-prefixSum_d[dim];
			

			// if there are any candidates left
			if(numberOfCandidates > 0){

				// Delete all candidates smaller than minSupp,
				// the prefix sum is calculated before the "if" to get the number of candidates left
				checkCudaErrors(cudaMemPrefetchAsync(candidates_d, sizeOfCandidates, device, stream1_1));
				unsigned int* newCandidates_d;
				sizeOfCandidates = numberOfCandidates*numberOfBlocksPrPoint*sizeof(unsigned int);  //<--- if there is an error this could be the case, changed without testing numberOfCandidates were dim before
				checkCudaErrors(cudaMallocManaged((void**) &newCandidates_d,   sizeOfCandidates));
				
				checkCudaErrors(cudaMemPrefetchAsync(candidates_d, dim*numberOfBlocksPrPoint*sizeof(unsigned int), device, stream1_1));
				checkCudaErrors(cudaMemPrefetchAsync(newCandidates_d, sizeOfCandidates, device, stream1_1));
				checkCudaErrors(cudaMemPrefetchAsync(prefixSum_d, sizeOfPrefixSum, device, stream1_1));
				
				deleteFromArrayTransfomedDataWrapper(ceilf((float)dim/32), dimBlock, stream1_1,
													 candidates_d, prefixSum_d, oldNumberOfCandidates,
													 numberOfBlocksPrPoint, newCandidates_d);


				// Also delete the scores
				float* newScore_d;
				checkCudaErrors(cudaMemPrefetchAsync(score_d, sizeOfScore, device, stream1_1));
				sizeOfScore = numberOfCandidates*sizeof(unsigned int);
				checkCudaErrors(cudaMallocManaged((void**) &newScore_d,   sizeOfScore));
				checkCudaErrors(cudaMemPrefetchAsync(score_d, sizeOfScore, device, stream1_1));
				checkCudaErrors(cudaMemPrefetchAsync(prefixSum_d, sizeOfPrefixSum, device, stream1_1));
				deleteFromArrayTransfomedDataWrapper(ceilf((float)dim/32), dimBlock, stream1_1,
													 score_d, prefixSum_d,
													 oldNumberOfCandidates, 1, newScore_d);

				// create indeces for argmax, it is here because it can be done concurently with the above kernel, and the frees below are blocking
				unsigned int* index_d;
				checkCudaErrors(cudaMallocManaged((void **) &index_d, numberOfCandidates*sizeof(unsigned int)));
				checkCudaErrors(cudaMemPrefetchAsync(index_d, numberOfCandidates*sizeof(unsigned int), device, stream1_2));
				createIndicesKernel(ceilf((float)numberOfCandidates/dimBlock), dimBlock, stream1_2,
									index_d, numberOfCandidates);

				// free all the memory that is not needed any more
				checkCudaErrors(cudaFree(prefixSum_d));
				checkCudaErrors(cudaFree(candidates_d));
				checkCudaErrors(cudaFree(score_d));
				checkCudaErrors(cudaFree(support_d));
				candidates_d = newCandidates_d;
				score_d = newScore_d;

				// find the current highest scoring, and save it for later
				checkCudaErrors(cudaMemPrefetchAsync(score_d, sizeOfScore, device, stream1_2));
				checkCudaErrors(cudaMemPrefetchAsync(index_d, numberOfCandidates*sizeof(unsigned int), device, stream1_1));
				checkCudaErrors(cudaStreamSynchronize(stream1_2)); // make sure indeces are done
				argMaxKernel(ceilf((float)numberOfCandidates/dimBlock), dimBlock,
							 smemSize ,stream1_1, score_d, index_d, numberOfCandidates);



				checkCudaErrors(cudaMemPrefetchAsync(candidates_d, sizeOfCandidates, device, stream1_1));
				checkCudaErrors(cudaMemPrefetchAsync(score_d, sizeof(float), device, stream1_1));
				checkCudaErrors(cudaMemPrefetchAsync(bestScore_d+i, sizeof(float), device, stream1_1));
				checkCudaErrors(cudaMemPrefetchAsync(bestCentroid_d+i, sizeof(unsigned int), device, stream1_1));
				checkCudaErrors(cudaMemPrefetchAsync(bestCandidate_d+i*numberOfBlocksPrPoint, numberOfBlocksPrPoint*sizeof(unsigned int), device, stream1_1));
				checkCudaErrors(cudaMemPrefetchAsync(index_d, sizeof(unsigned int), device, stream1_1));
				// Find the best score, then copy that score and the coresponding candidate and centroid to the result
				// All centroids best are stored in the same array, one after each other,
				//   to make it possible to find the disjoint clusters later.
				//   unlike everywhere else, the candidates are written one after each other,
				//   just to make it simpler to write ant read, because it is done by multible kernel calls.
				// 	 Since the number of elements are so small, the performance should not suffer that much.
				extractMaxWrapper(ceilf((float)numberOfBlocksPrPoint/dimBlock), dimBlock, stream1_1,
								  candidates_d, score_d, currentCentroidIndex, numberOfCandidates, index_d,
								  dim, bestCandidate_d+i*numberOfBlocksPrPoint, bestScore_d+i, bestCentroid_d+i);

				checkCudaErrors(cudaFree(index_d));
				checkCudaErrors(cudaFree(score_d)); // The score vector is not be needed any more, the candidates are.

				// loop until there are no more candidates, or there are no more dimensions. 
				unsigned int iterationNr = 1;
				while(numberOfCandidates > 1 && iterationNr <= dim+1){

					
					iterationNr++;

					// Merge candidates
					// The number of candidates after a merge can be seen as
					//   an upper triangular matrix without the diagonal
					oldNumberOfCandidates = numberOfCandidates;
					numberOfCandidates = ((size_t)numberOfCandidates*(numberOfCandidates+1))/2-numberOfCandidates;
					sizeOfCandidates = (numberOfCandidates)*numberOfBlocksPrPoint * sizeof(unsigned int);
					sizeOfToBeDeleted = (numberOfCandidates+1)*sizeof(bool);
					
					unsigned int* newCandidates_d;
					bool* deletedFromCount_d;

					checkCudaErrors(cudaMallocManaged((void**) &deletedFromCount_d,   sizeOfToBeDeleted));
					checkCudaErrors(cudaMallocManaged((void**) &newCandidates_d,   sizeOfCandidates));

					checkCudaErrors(cudaMemPrefetchAsync(candidates_d, oldNumberOfCandidates*numberOfBlocksPrPoint * sizeof(unsigned int), device, stream1_1));
					checkCudaErrors(cudaMemPrefetchAsync(deletedFromCount_d, sizeOfToBeDeleted, device, stream1_1));
					checkCudaErrors(cudaMemPrefetchAsync(newCandidates_d, sizeOfCandidates, device, stream1_1));
					
					mergeCandidatesWrapper(ceilf((float)numberOfCandidates/dimBlock), dimBlock, stream1_1,
										   candidates_d, oldNumberOfCandidates, dim, iterationNr,
										   newCandidates_d, deletedFromCount_d);

					//std::cout << "oldNumberOfCandidates: " << oldNumberOfCandidates << std::endl;
					//std::cout << "numberOfCandidates: " << numberOfCandidates << std::endl;
					checkCudaErrors(cudaFree(candidates_d)); // delete the old candidate array
					candidates_d = newCandidates_d;

					checkCudaErrors(cudaMallocManaged((void**) &toBeDeleted_d, sizeOfToBeDeleted));
					
					// Find all the dublicate candidates
					// For the find dublicates call the output vector needs to be sat to 0,
					//    because it only writes the true values
					//checkCudaErrors(cudaMemsetAsync(toBeDeleted_d, 0, sizeOfToBeDeleted, stream1_1));
					checkCudaErrors(cudaMemPrefetchAsync(candidates_d, sizeOfCandidates, device, stream1_1));
					checkCudaErrors(cudaMemPrefetchAsync(deletedFromCount_d, sizeOfToBeDeleted, device, stream1_1));
					checkCudaErrors(cudaMemPrefetchAsync(toBeDeleted_d, sizeOfToBeDeleted, device, stream1_1));
					
					findDublicatesWrapper_mananged(ceilf((float)numberOfCandidates/dimBlock), dimBlock, stream1_1,
										  candidates_d, numberOfCandidates, dim,
										  deletedFromCount_d, toBeDeleted_d, this->duplicateKernelVerison);
					
					
					
					orKernelWrapper(ceilf((float)(numberOfCandidates+1)/dimBlock), dimBlock, stream1_1,
									numberOfCandidates+1, toBeDeleted_d, deletedFromCount_d);
					

					checkCudaErrors(cudaFree(deletedFromCount_d));
					
					// Compute the prefix sum and then find the number of dublicates
					sizeOfPrefixSum = (numberOfCandidates+1)*sizeof(unsigned int);
					checkCudaErrors(cudaMallocManaged((void**) &prefixSum_d,   sizeOfPrefixSum));
					checkCudaErrors(cudaMemPrefetchAsync(toBeDeleted_d, sizeOfToBeDeleted, device, stream1_1));
					checkCudaErrors(cudaMemPrefetchAsync(prefixSum_d, sizeOfPrefixSum, device, stream1_1));
					sum_scan_blelloch_managed(stream1_1,stream1_1, prefixSum_d,toBeDeleted_d,(numberOfCandidates+1), false);
					
					checkCudaErrors(cudaFree(toBeDeleted_d));

					// always sync streams before reading value from host
					checkCudaErrors(cudaStreamSynchronize(stream1_1));

					checkCudaErrors(cudaMemPrefetchAsync(prefixSum_d+numberOfCandidates, sizeof(unsigned int), cudaCpuDeviceId, stream1_1));			
					assert(numberOfCandidates >= prefixSum_d[numberOfCandidates]); // avoid underflow
					
				
					// Calculate the new number of candidates, based on the amount that should be deleted
					oldNumberOfCandidates = numberOfCandidates;
					numberOfCandidates = numberOfCandidates-prefixSum_d[numberOfCandidates];
					if(numberOfCandidates <= 0){
						break;
					}
					//std::cout <<i<< " "<< iterationNr << " numberOfCandidates: " << numberOfCandidates << std::endl;
										
					// Delete the dublicate candidates
					checkCudaErrors(cudaMemPrefetchAsync(candidates_d, sizeOfCandidates, device, stream1_1));
					sizeOfCandidates = numberOfCandidates*numberOfBlocksPrPoint*sizeof(unsigned int);
					checkCudaErrors(cudaMallocManaged((void**) &newCandidates_d,   sizeOfCandidates));
					unsigned int delTrfmDataDimGrid = ceilf(((float)oldNumberOfCandidates*numberOfBlocksPrPoint)
															/dimBlock);
					checkCudaErrors(cudaMemPrefetchAsync(newCandidates_d, sizeOfCandidates, device, stream1_1));
					checkCudaErrors(cudaMemPrefetchAsync(prefixSum_d, sizeOfPrefixSum, device, stream1_1));
					
					deleteFromArrayTransfomedDataWrapper(delTrfmDataDimGrid, dimBlock, stream1_1,
														 candidates_d, prefixSum_d, oldNumberOfCandidates,
														 numberOfBlocksPrPoint, newCandidates_d);

					// clean up after the deletion
					checkCudaErrors(cudaFree(candidates_d));
					checkCudaErrors(cudaFree(prefixSum_d));
				
					candidates_d = newCandidates_d;
					score_d = newScore_d;

					// Count support
					sizeOfSupport = numberOfCandidates*sizeof(unsigned int);
					sizeOfScore = numberOfCandidates*sizeof(float);
					sizeOfToBeDeleted = (numberOfCandidates+1)*sizeof(bool);

					checkCudaErrors(cudaMallocManaged((void**) &support_d,   sizeOfSupport));
					checkCudaErrors(cudaMallocManaged((void**) &score_d,   sizeOfScore));
					checkCudaErrors(cudaMallocManaged((void**) &toBeDeleted_d,   sizeOfToBeDeleted));


					checkCudaErrors(cudaMemPrefetchAsync(support_d, sizeOfSupport, device, stream1_1));
					checkCudaErrors(cudaMemPrefetchAsync(score_d, sizeOfScore, device, stream1_1));
					checkCudaErrors(cudaMemPrefetchAsync(toBeDeleted_d, sizeOfToBeDeleted, device, stream1_1));
					checkCudaErrors(cudaMemPrefetchAsync(itemSet_d, sizeOfItemSet, device, stream1_1));
					checkCudaErrors(cudaMemPrefetchAsync(candidates_d, sizeOfCandidates, device, stream1_1));
						
					countSupportWrapper(ceilf((float)numberOfCandidates/dimBlock), dimBlock, stream1_1,
										candidates_d, itemSet_d, dim, numberOfPoints, numberOfCandidates, minSupp,
										this->beta, support_d, score_d, toBeDeleted_d, this->countSupportKernelVersion);




					// Delete the candidates with support smaller the minSupp.
					sizeOfPrefixSum = (numberOfCandidates+1)*sizeof(unsigned int);
					checkCudaErrors(cudaMallocManaged((void**) &prefixSum_d,   sizeOfPrefixSum));
					checkCudaErrors(cudaMemPrefetchAsync(toBeDeleted_d, sizeOfToBeDeleted, device, stream1_1));
					checkCudaErrors(cudaMemPrefetchAsync(prefixSum_d, sizeOfPrefixSum, device, stream1_1));
					sum_scan_blelloch_managed(stream1_1,stream1_1, prefixSum_d, toBeDeleted_d,(numberOfCandidates+1), false);

					checkCudaErrors(cudaFree(toBeDeleted_d));
					checkCudaErrors(cudaStreamSynchronize(stream1_1));
					
					oldNumberOfCandidates = numberOfCandidates;
					
 					checkCudaErrors(cudaMemPrefetchAsync(prefixSum_d+numberOfCandidates, sizeof(unsigned int), cudaCpuDeviceId, stream1_1));			
  					assert(numberOfCandidates >= prefixSum_d[numberOfCandidates]); // avoid underflow
  					numberOfCandidates = numberOfCandidates-prefixSum_d[numberOfCandidates];
					if(numberOfCandidates <= 0){
						break;
					}


					// Actually delete the candidates
					sizeOfCandidates = numberOfCandidates*numberOfBlocksPrPoint*sizeof(unsigned int);
					sizeOfScore = numberOfCandidates*sizeof(float);

					checkCudaErrors(cudaMallocManaged((void**) &newCandidates_d,   sizeOfCandidates));
					checkCudaErrors(cudaMallocManaged((void**) &newScore_d,   sizeOfScore));

					delTrfmDataDimGrid = ceilf((float)oldNumberOfCandidates*numberOfBlocksPrPoint/dimBlock);
					checkCudaErrors(cudaMemPrefetchAsync(candidates_d, sizeOfCandidates, device, stream1_1));
					checkCudaErrors(cudaMemPrefetchAsync(score_d, sizeOfScore, device, stream1_1));
					checkCudaErrors(cudaMemPrefetchAsync(newScore_d, sizeOfScore, device, stream1_1));
					checkCudaErrors(cudaMemPrefetchAsync(newCandidates_d, sizeOfCandidates, device, stream1_1));
					checkCudaErrors(cudaMemPrefetchAsync(prefixSum_d, sizeOfPrefixSum, device, stream1_1));
					
					deleteFromArrayTransfomedDataWrapper(delTrfmDataDimGrid, dimBlock, stream1_1,
														 candidates_d, prefixSum_d, oldNumberOfCandidates,
														 numberOfBlocksPrPoint, newCandidates_d);
					deleteFromArrayTransfomedDataWrapper(delTrfmDataDimGrid, dimBlock, stream1_2,
														 score_d, prefixSum_d, oldNumberOfCandidates, 1, newScore_d);


					// A synchronize is not needed before free, since free is implicit synchronizing.
					checkCudaErrors(cudaFree(support_d));
					checkCudaErrors(cudaFree(score_d));
					checkCudaErrors(cudaFree(candidates_d));
					checkCudaErrors(cudaFree(prefixSum_d));
					candidates_d = newCandidates_d;
					score_d = newScore_d;

		
					// Find the highest scoring cluster for the next iteration.
					checkCudaErrors(cudaMallocManaged((void **) &index_d, numberOfCandidates*sizeof(unsigned int)));
					checkCudaErrors(cudaMemPrefetchAsync(index_d, numberOfCandidates*sizeof(unsigned int), device, stream1_1));
					createIndicesKernel(ceilf((float)numberOfCandidates/dimBlock), dimBlock, stream1_1,
										index_d, numberOfCandidates);
					checkCudaErrors(cudaMemPrefetchAsync(score_d, sizeOfScore, device, stream1_2));
					checkCudaErrors(cudaStreamSynchronize(stream1_2));
					argMaxKernel(ceilf((float)numberOfCandidates/dimBlock), dimBlock, smemSize ,stream1_1,
								 score_d, index_d, numberOfCandidates);

					checkCudaErrors(cudaMemPrefetchAsync(candidates_d, sizeOfCandidates, device, stream1_1));
					checkCudaErrors(cudaMemPrefetchAsync(score_d, sizeof(float), device, stream1_1));
					checkCudaErrors(cudaMemPrefetchAsync(bestScore_d+i, sizeof(float), device, stream1_1));
					checkCudaErrors(cudaMemPrefetchAsync(bestCentroid_d+i, sizeof(unsigned int), device, stream1_1));
					checkCudaErrors(cudaMemPrefetchAsync(bestCandidate_d+i*numberOfBlocksPrPoint, numberOfBlocksPrPoint*sizeof(unsigned int), device, stream1_1));
					checkCudaErrors(cudaMemPrefetchAsync(index_d, sizeof(unsigned int), device, stream1_1));

					// Store the best in the arrays to look for dublicate arrays. 
					extractMaxWrapper(ceilf((float)numberOfBlocksPrPoint/dimBlock), dimBlock, stream1_1,
									  candidates_d, score_d, currentCentroidIndex, numberOfCandidates, index_d,
									  dim, bestCandidate_d+i*numberOfBlocksPrPoint, bestScore_d+i, bestCentroid_d+i);

					checkCudaErrors(cudaFree(index_d));
					checkCudaErrors(cudaFree(score_d));
				} // Apriori iterations
			}else{ // Number of candidates > 0 for outer
				checkCudaErrors(cudaMemPrefetchAsync(bestScore_d+i, sizeof(float), cudaCpuDeviceId, stream1_1));
				checkCudaErrors(cudaMemPrefetchAsync(bestCentroid_d+i, sizeof(unsigned int), cudaCpuDeviceId, stream1_1));
				checkCudaErrors(cudaStreamSynchronize(stream1_1));
				bestScore_d[i] = 0;
				bestCentroid_d[i] = 0;
				
				checkCudaErrors(cudaFree(prefixSum_d));

				checkCudaErrors(cudaFree(support_d));
				checkCudaErrors(cudaFree(score_d));

			}
			checkCudaErrors(cudaFree(itemSet_d));
			// checkCudaErrors(cudaFreeHost(sum_h));
			checkCudaErrors(cudaFree(candidates_d));
		}// for all centroids
		// Now all the centroids have been looked at, and it is time to find the best scoring disjoint clusters.
		// std::cout << "all centroids done" << std::endl;

		bool* disjointClustersToBeRemoved_b_d; 
		checkCudaErrors(cudaMallocManaged((void**) &disjointClustersToBeRemoved_b_d,   sizeof(bool)*(numberOfCentroids+1)));
		if(this->isConcurentVersion()){
			// std::cout << "concurent version" << std::endl; 
			// Delete the non-disjoint clusters, and keeping the largest of them all. 
			unsigned int* disjointClustersToBeRemoved_d;
			checkCudaErrors(cudaMallocManaged((void**) &disjointClustersToBeRemoved_d,
									   sizeof(unsigned int)*(numberOfCentroids+1)));
			unsigned int numberOfComparisons = (numberOfCentroids*(numberOfCentroids+1))/2 - numberOfCentroids;
			unsigned int dimGrid_disj = ceilf((float)numberOfComparisons/dimBlock);


			checkCudaErrors(cudaMemPrefetchAsync(bestCentroid_d, numberOfCentroids*sizeof(unsigned int), device, stream1_1));
			checkCudaErrors(cudaMemPrefetchAsync(bestScore_d, numberOfCentroids*sizeof(float), device, stream1_1));
			checkCudaErrors(cudaMemPrefetchAsync(bestCandidate_d, numberOfCentroids*numberOfBlocksPrPoint*sizeof(unsigned int), device, stream1_1));
			checkCudaErrors(cudaMemPrefetchAsync(disjointClustersToBeRemoved_d, sizeof(unsigned int)*(numberOfCentroids+1), device, stream1_1));
			// data is intentionally left out, since we only need a few points, so page faults might be worth it

			disjointClustersWrapper(dimGrid_disj, dimBlock, stream1_1, bestCentroid_d,
									bestScore_d, bestCandidate_d, data_d,
									numberOfCentroids, dim, width, disjointClustersToBeRemoved_d);

			checkCudaErrors(cudaMemPrefetchAsync(disjointClustersToBeRemoved_b_d, sizeof(bool)*(numberOfCentroids+1), device, stream1_1));
			unsignedIntToBoolArrayWrapper(ceilf((float)numberOfCentroids/dimBlock), dimBlock, stream1_1,
										  disjointClustersToBeRemoved_d, numberOfCentroids, 
										  disjointClustersToBeRemoved_b_d);

			checkCudaErrors(cudaFree(disjointClustersToBeRemoved_d));
			
		}else{ // just find the best cluster of all

			unsigned int* index_d;
			checkCudaErrors(cudaMallocManaged((void **) &index_d, numberOfCentroids*sizeof(unsigned int)));
			
			checkCudaErrors(cudaMemPrefetchAsync(index_d, sizeof(unsigned int)*(numberOfCentroids), device, stream1_1));
			checkCudaErrors(cudaMemPrefetchAsync(bestScore_d, sizeof(float)*(numberOfCentroids), device, stream1_2));
			checkCudaErrors(cudaMemPrefetchAsync(disjointClustersToBeRemoved_b_d, sizeof(bool)*(numberOfCentroids+1), device, stream1_2));
			
			createIndicesKernel(ceilf((float)numberOfCentroids/dimBlock), dimBlock, stream1_1,
								index_d, numberOfCentroids);
			checkCudaErrors(cudaStreamSynchronize(stream1_2));
			argMaxKernel(ceilf((float)numberOfCentroids/dimBlock), dimBlock,
						 smemSize ,stream1_1, bestScore_d, index_d, numberOfCentroids);
			indexToBoolVectorWrapper(ceilf((float)numberOfCentroids/dimBlock), dimBlock, stream1_1,
									 index_d, numberOfCentroids, disjointClustersToBeRemoved_b_d);
			checkCudaErrors(cudaFree(index_d));

		}

			
	
		unsigned int* prefixSum_d;
		size_t sizeOfPrefixSum = (numberOfCentroids+1)*sizeof(unsigned int);
		checkCudaErrors(cudaMallocManaged((void**) &prefixSum_d,   sizeOfPrefixSum));
		checkCudaErrors(cudaMemPrefetchAsync(disjointClustersToBeRemoved_b_d, sizeof(bool)*(numberOfCentroids+1), device, stream1_1));
		checkCudaErrors(cudaMemPrefetchAsync(prefixSum_d, sizeof(unsigned int)*(numberOfCentroids+1), device, stream1_1));
		sum_scan_blelloch_managed(stream1_1,stream1_1, prefixSum_d,disjointClustersToBeRemoved_b_d,(numberOfCentroids+1), true);


		checkCudaErrors(cudaFree(disjointClustersToBeRemoved_b_d));		
		checkCudaErrors(cudaStreamSynchronize(stream1_1));
		// make sure that there can be no more centroids than there were before
		
		checkCudaErrors(cudaMemPrefetchAsync(prefixSum_d+numberOfCentroids, sizeof(unsigned int), cudaCpuDeviceId, stream1_1));

		assert(numberOfCentroids >= prefixSum_d[numberOfCentroids]);

		unsigned int finalNumberOfClusters = numberOfCentroids-prefixSum_d[numberOfCentroids];
		// checkCudaErrors(cudaFreeHost(newNumberOfCentroids_h));
		if(finalNumberOfClusters == 0){
			checkCudaErrors(cudaFree(bestCentroid_d));
			checkCudaErrors(cudaFree(bestScore_d));
			checkCudaErrors(cudaFree(bestCandidate_d));
			checkCudaErrors(cudaFree(prefixSum_d));
			break;
		}

		// Allocate the space for the final clusters found
		float* finalScore_d;
		unsigned int* finalCentroids_d;
		unsigned int* finalCandidates_d;
		checkCudaErrors(cudaMallocManaged((void**) &finalScore_d,   finalNumberOfClusters*sizeof(float)));
		checkCudaErrors(cudaMallocManaged((void**) &finalCentroids_d,   finalNumberOfClusters*sizeof(unsigned int)));
		checkCudaErrors(cudaMallocManaged((void**) &finalCandidates_d,
								   finalNumberOfClusters*numberOfBlocksPrPoint*sizeof(unsigned int)));

		checkCudaErrors(cudaMemPrefetchAsync(prefixSum_d, sizeOfPrefixSum, device, stream1_1));
		checkCudaErrors(cudaMemPrefetchAsync(bestScore_d, numberOfCentroids*sizeof(float), device, stream1_1));
		checkCudaErrors(cudaMemPrefetchAsync(bestCentroid_d, numberOfCentroids*sizeof(unsigned int), device, stream1_1));
		checkCudaErrors(cudaMemPrefetchAsync(bestCandidate_d, numberOfCentroids*numberOfBlocksPrPoint*sizeof(unsigned int), device, stream1_1));
		checkCudaErrors(cudaMemPrefetchAsync(finalScore_d, finalNumberOfClusters*sizeof(float), device, stream1_1));
		checkCudaErrors(cudaMemPrefetchAsync(finalCentroids_d, finalNumberOfClusters*sizeof(unsigned int), device, stream1_1));
		checkCudaErrors(cudaMemPrefetchAsync(finalCandidates_d, finalNumberOfClusters*numberOfBlocksPrPoint*sizeof(unsigned int), device, stream1_1));						

		// Delete form the arrays
		deleteFromArrayWrapper(ceilf((float)(numberOfCentroids)/dimBlock), dimBlock, stream1_1,
							   bestCentroid_d, prefixSum_d, numberOfCentroids, 1, finalCentroids_d);
		deleteFromArrayWrapper(ceilf((float)(numberOfCentroids)/dimBlock), dimBlock, stream1_1,
							   bestScore_d, prefixSum_d, numberOfCentroids, 1, finalScore_d);
		deleteFromArrayWrapper(ceilf((float)(numberOfCentroids*numberOfBlocksPrPoint)/dimBlock), dimBlock,
							   stream1_2, bestCandidate_d, prefixSum_d, // Uses a different stream, since they can all be concurrent, and this is the largest
							   numberOfCentroids, numberOfBlocksPrPoint, finalCandidates_d);

		checkCudaErrors(cudaFree(bestCentroid_d));
		checkCudaErrors(cudaFree(bestScore_d));
		checkCudaErrors(cudaFree(bestCandidate_d));
		checkCudaErrors(cudaFree(prefixSum_d));
		
		// Copy the centroids to a new array, because the offsets will be scrambled after the first cluster is deleted.
		float* outputCentroids_d;
		checkCudaErrors(cudaMallocManaged((void**) &outputCentroids_d,   dim*finalNumberOfClusters*sizeof(float)));
		checkCudaErrors(cudaMemPrefetchAsync(outputCentroids_d, dim*finalNumberOfClusters*sizeof(float), device, stream1_1));
		copyCentroidWrapper(ceilf((float)finalNumberOfClusters*dim/dimBlock),dimBlock ,stream1_1,
							finalCentroids_d, data_d, dim, finalNumberOfClusters, outputCentroids_d);

		checkCudaErrors(cudaFree(finalCentroids_d));
		// For each of the clusters found
		for(unsigned int i = 0; i < finalNumberOfClusters; i++){
			// Find the points in the i'th cluster
			bool* pointsContained_d;
			checkCudaErrors(cudaMallocManaged((void**) &pointsContained_d,   (numberOfPoints+1)*sizeof(bool)));

			if(numberOfPoints != 0){
				checkCudaErrors(cudaMemPrefetchAsync(data_d, numberOfPoints*dim*sizeof(float), device, stream1_1));
				checkCudaErrors(cudaMemPrefetchAsync(finalCandidates_d+i*numberOfBlocksPrPoint, numberOfBlocksPrPoint*sizeof(float), device, stream1_1));
				checkCudaErrors(cudaMemPrefetchAsync(outputCentroids_d+i*dim, dim*sizeof(float), device, stream1_1));
				
			}			

			findPointInClusterWrapper(ceilf((float)numberOfPoints/dimBlock), dimBlock, stream1_1,
									  finalCandidates_d+i*numberOfBlocksPrPoint, data_d, outputCentroids_d+i*dim, dim,
									  numberOfPoints, width, pointsContained_d);

			unsigned int* prefixSum_d;
			size_t sizeOfPrefixSum = (numberOfPoints+1)*sizeof(unsigned int);
			checkCudaErrors(cudaMallocManaged((void**) &prefixSum_d,   sizeOfPrefixSum));
			checkCudaErrors(cudaMemPrefetchAsync(prefixSum_d, sizeOfPrefixSum, device, stream1_1));
			sum_scan_blelloch_managed(stream1_1,stream1_1, prefixSum_d,pointsContained_d,(numberOfPoints+1), true);

			// Fetch the number of points in the cluster
			checkCudaErrors(cudaMemPrefetchAsync(prefixSum_d+numberOfPoints, sizeof(unsigned int), cudaCpuDeviceId, stream1_1));
			checkCudaErrors(cudaStreamSynchronize(stream1_1));
			unsigned int oldNumberOfPoints = numberOfPoints;
			numberOfPoints = prefixSum_d[numberOfPoints];
			
			// Create the output cluster
			float* outputCluster_d;
	
			size_t sizeOfOutputCluster = (oldNumberOfPoints - numberOfPoints)*dim*sizeof(unsigned int);
			checkCudaErrors(cudaMallocManaged((void**) &outputCluster_d,   sizeOfOutputCluster));
			checkCudaErrors(cudaMemPrefetchAsync(prefixSum_d, sizeOfPrefixSum, device, stream1_1));
			checkCudaErrors(cudaMemPrefetchAsync(outputCluster_d, sizeOfOutputCluster, device, stream1_1));
			if(numberOfPoints > 0){
				checkCudaErrors(cudaMemPrefetchAsync(data_d, numberOfPoints*dim*sizeof(float), device, stream1_1));
			}
			deleteFromArrayWrapper(ceilf((float)(oldNumberOfPoints*dim)/dimBlock), dimBlock, stream1_1,
								   data_d, prefixSum_d, oldNumberOfPoints, dim, outputCluster_d);
		


			// Delete the points already in the cluster.
			float* newData_d;
			size_t sizeOfNewData = numberOfPoints*dim*sizeof(unsigned int);
			if(sizeOfNewData != 0){
				checkCudaErrors(cudaMallocManaged((void**) &newData_d, sizeOfNewData)); // This can be done at the same time as previous delete
				checkCudaErrors(cudaMemPrefetchAsync(prefixSum_d, sizeOfPrefixSum, device, stream1_1));
				checkCudaErrors(cudaMemPrefetchAsync(data_d, numberOfPoints*dim*sizeof(float), device, stream1_1));
				checkCudaErrors(cudaMemPrefetchAsync(newData_d, sizeOfNewData, device, stream1_2));
			}
			sum_scan_blelloch_managed(stream1_1,stream1_1, prefixSum_d,pointsContained_d,(oldNumberOfPoints+1), false);
			
			checkCudaErrors(cudaStreamSynchronize(stream1_2));
			
			checkCudaErrors(cudaMemPrefetchAsync(prefixSum_d, sizeOfPrefixSum, device, stream1_1));
			
			deleteFromArrayWrapper(ceilf((float)oldNumberOfPoints*dim/dimBlock),  dimBlock, stream1_1,
								   data_d, prefixSum_d, oldNumberOfPoints, dim, newData_d);

			
			checkCudaErrors(cudaFree(pointsContained_d));
			checkCudaErrors(cudaFree(prefixSum_d));
			
			// Make the output vectors.
			checkCudaErrors(cudaMemPrefetchAsync(outputCluster_d, sizeOfOutputCluster, cudaCpuDeviceId, stream1_1));
			checkCudaErrors(cudaMemPrefetchAsync(finalCandidates_d+i*numberOfBlocksPrPoint, numberOfBlocksPrPoint*sizeof(unsigned int), cudaCpuDeviceId, stream1_2));
			checkCudaErrors(cudaStreamSynchronize(stream1_1));
			auto outputCluster = new std::vector<std::vector<float>*>;
			for(unsigned int i = 0; i < oldNumberOfPoints - numberOfPoints; i++){
				auto point = new std::vector<float>;
				for(unsigned int j = 0; j < dim; j++){
					point->push_back(outputCluster_d[i*dim+j]);
				}
				outputCluster->push_back(point);
			}
			checkCudaErrors(cudaFree(outputCluster_d));
			checkCudaErrors(cudaStreamSynchronize(stream1_2));
			if(numberOfPoints != 0){
				checkCudaErrors(cudaFree(data_d));
			}
			auto outputDim = new std::vector<bool>;
			unsigned int count = 0;
			for(unsigned int k = 0; k < numberOfBlocksPrPoint;k++){
				for(unsigned int j = 0; j < 32; j++){
					outputDim->push_back((finalCandidates_d[i*numberOfBlocksPrPoint+k] >> j) & (unsigned int)1);
					count++;
					if(count >= dim){
						break;
					}
				}
			}

			
			result.push_back(std::make_pair(outputCluster,outputDim));
			data_d = newData_d;
			// Break if the are no more points.
			if(numberOfPoints == 0){
				break;
			}
		}

		checkCudaErrors(cudaFree(finalScore_d));
		checkCudaErrors(cudaFree(finalCandidates_d));
		checkCudaErrors(cudaFree(outputCentroids_d));
		if(numberOfPoints == 0){
			break;
		}
	}
	if(numberOfPoints !=0){
		checkCudaErrors(cudaFree(data_d));
	}
	checkCudaErrors(cudaStreamDestroy(stream1_1));
	checkCudaErrors(cudaStreamDestroy(stream1_2));
	
	// Sort the result at the end, such that the highest scoring is first
	// This is not guaranteed if the concurrent mode are enabled

	std::sort(result.begin(), result.end(), [&](const std::pair<std::vector<
												std::vector<float>*>*, std::vector<bool>*>& lhs,
												const std::pair<std::vector<std::vector<float>*>*,
												std::vector<bool>*> rhs){
				  unsigned int lhsSupport = lhs.first->size();
				  unsigned int lhsSubspaceSize = 0;
				  for(int i = 0; i < lhs.second->size(); i++){
					  lhsSubspaceSize += lhs.second->at(i);
				  }

				  unsigned int rhsSupport = rhs.first->size();
				  unsigned int rhsSubspaceSize = 0;
				  for(int i = 0; i < rhs.second->size(); i++){
					  rhsSubspaceSize += rhs.second->at(i);
				  }

				  return this->mu(lhsSupport, lhsSubspaceSize) > this->mu(rhsSupport, rhsSubspaceSize);
			
			  });
	
	return result;

	
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
