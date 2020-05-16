#include "MineClusGPU.h"
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
	unsigned int size = this->data->size();
	unsigned int dim = this->data->at(0)->size();
	size_t size_of_data = size*dim*sizeof(float);
	float* data_h;
	checkCudaErrors(cudaMallocHost((void**) &data_h, size_of_data));
	
	for(unsigned int i = 0; i < size; i++){
		for(unsigned int j = 0; j < dim; j++){
			data_h[(size_t)i*dim+j] = data->at(i)->at(j);
		}
	}
	return data_h;
};


/**
 * Find a single cluster, uses the function to find multiple clusters
 */
std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> MineClusGPU::findCluster(){
	auto result = findKClusters(1);
	if (result.size() == 0){
		return std::make_pair(new std::vector<std::vector<float>*>, new std::vector<bool>);
	}else{
		return result.at(0);	
	}
};

/**

   TODO in this function
   Make the centroids run in seperate threads
   Free memory after the loop
*/
std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*>> MineClusGPU::findKClusters(int k){

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
	int smemSize, maxBlock;
	cudaDeviceGetAttribute(&smemSize, 
						   cudaDevAttrMaxSharedMemoryPerBlock, 0);
	cudaDeviceGetAttribute(&maxBlock, 
						   cudaDevAttrMaxThreadsPerBlock, 0); 

	
	// create streams
	cudaStream_t stream1_1;
	checkCudaErrors(cudaStreamCreate(&stream1_1));
	cudaStream_t stream1_2;
	checkCudaErrors(cudaStreamCreate(&stream1_2));

	// Transfer data the data
	size_t sizeOfData = size*dim*sizeof(float);
	float* data_h = this->transformData();
	float* data_d;
	checkCudaErrors(cudaMalloc((void**) &data_d,  sizeOfData));
	checkCudaErrors(cudaMemcpyAsync(data_d, data_h, sizeOfData, cudaMemcpyHostToDevice, stream1_2));
	checkCudaErrors(cudaFreeHost(data_h));

	// This loop is running until all clusters are found
	while(k > result.size()){
		
		// Allocate the space for the best clusters for all centroids
		float* bestScore_d;
		unsigned int* bestCentroid_d;
		unsigned int* bestCandidate_d;
		checkCudaErrors(cudaMalloc((void**) &bestScore_d,  numberOfCentroids*sizeof(float)));
		checkCudaErrors(cudaMalloc((void**) &bestCentroid_d,  numberOfCentroids*sizeof(unsigned int)));
		checkCudaErrors(cudaMalloc((void**) &bestCandidate_d,
								   numberOfCentroids*numberOfBlocksPrPoint*sizeof(unsigned int)));

		// If it is the first iteration set the current best score to 0 for all future centroids too
		float* bestScore_h;
		checkCudaErrors(cudaMallocHost((void**) &bestScore_h, sizeof(float)*numberOfCentroids));
		for(int j = 0; j < numberOfCentroids; j++){
			bestScore_h[j] = 0;
		}
		checkCudaErrors(cudaMemcpy(bestScore_d, bestScore_h, sizeof(float)*numberOfCentroids,
								   cudaMemcpyHostToDevice));				
		checkCudaErrors(cudaFreeHost(bestScore_h));

		
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
			checkCudaErrors(cudaMalloc((void**) &itemSet_d,  sizeOfItemSet));
			createTransactionsWrapper(ceilf((float)size/dimBlock), dimBlock, smemSize ,stream1_2, data_d, dim,
								 numberOfPoints, currentCentroidIndex, this->width, itemSet_d); 

			// Create the initial candidates
			size_t sizeOfCandidates = dim*numberOfBlocksPrPoint*sizeof(unsigned int);
			unsigned int* candidates_d;
			checkCudaErrors(cudaMalloc((void**) &candidates_d,  sizeOfCandidates));
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

			checkCudaErrors(cudaMalloc((void**) &support_d,   sizeOfSupport));
			checkCudaErrors(cudaMalloc((void**) &score_d,   sizeOfScore));
			checkCudaErrors(cudaMalloc((void**) &toBeDeleted_d,   sizeOfToBeDeleted));

			// Synchronize to make sure candidates and item set are donee
			checkCudaErrors(cudaStreamSynchronize(stream1_1));
			checkCudaErrors(cudaStreamSynchronize(stream1_2));

			// Count the support
			countSupportWrapper(ceilf((float)dim/dimBlock), dimBlock, stream1_1,
								candidates_d, itemSet_d, dim, numberOfPoints, dim, minSupp,
								this->beta, support_d, score_d, toBeDeleted_d, this->countSupportKernelVersion);



			// Create the PrefixSum, and find the number of elements left
			// The number elements to be deleted is the last value in the prefixSum
			size_t sizeOfPrefixSum = (dim+1)*sizeof(unsigned int); // +1 because of the prefixSum
			unsigned int* prefixSum_d;
			checkCudaErrors(cudaMalloc((void**) &prefixSum_d,   sizeOfPrefixSum));
			sum_scan_blelloch(stream1_1, prefixSum_d,toBeDeleted_d,(dim+1), false);

			// Find the number of candidates after deletion of the small candidates
			unsigned int* sum_h;
			checkCudaErrors(cudaMallocHost((void**) &sum_h, sizeof(unsigned int)));
			checkCudaErrors(cudaMemcpyAsync(sum_h, prefixSum_d+dim, sizeof(unsigned int),
											cudaMemcpyDeviceToHost,stream1_1));
			checkCudaErrors(cudaFree(toBeDeleted_d));
			checkCudaErrors(cudaStreamSynchronize(stream1_1));
			unsigned int oldNumberOfCandidates = dim;	
			size_t numberOfCandidates = dim-sum_h[0];

			// if there are any candidates left
			if(numberOfCandidates > 0){

				// Delete all candidates smaller than minSupp,
				// the prefix sum is calculated before the "if" to get the number of candidates left
				unsigned int* newCandidates_d;
				sizeOfCandidates = dim*numberOfBlocksPrPoint*sizeof(unsigned int);
				checkCudaErrors(cudaMalloc((void**) &newCandidates_d,   sizeOfCandidates));
				deleteFromArrayTransfomedDataWrapper(ceilf((float)dim/32), dimBlock, stream1_1,
													 candidates_d, prefixSum_d, oldNumberOfCandidates,
													 numberOfBlocksPrPoint, newCandidates_d);


				// Also delete the scores
				float* newScore_d;
				sizeOfScore = dim*sizeof(unsigned int);
				checkCudaErrors(cudaMalloc((void**) &newScore_d,   sizeOfScore));
				deleteFromArrayTransfomedDataWrapper(ceilf((float)dim/32), dimBlock, stream1_1,
													 score_d, prefixSum_d,
													 oldNumberOfCandidates, 1, newScore_d);

				// create indeces for argmax, it is here because it can be done concurently with the above kernel, and the frees below are blocking
				unsigned int* index_d;
				checkCudaErrors(cudaMalloc((void **) &index_d, numberOfCandidates*sizeof(unsigned int)));
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
				checkCudaErrors(cudaStreamSynchronize(stream1_2)); // make sure indeces are done
				argMaxKernel(ceilf((float)numberOfCandidates/dimBlock), dimBlock,
							 smemSize ,stream1_1, score_d, index_d, numberOfCandidates);


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

					checkCudaErrors(cudaMalloc((void**) &deletedFromCount_d,   sizeOfToBeDeleted));
					checkCudaErrors(cudaMalloc((void**) &newCandidates_d,   sizeOfCandidates));
				
					mergeCandidatesWrapper(ceilf((float)numberOfCandidates/dimBlock), dimBlock, stream1_1,
										   candidates_d, oldNumberOfCandidates, dim, iterationNr,
										   newCandidates_d, deletedFromCount_d);

					checkCudaErrors(cudaFree(candidates_d)); // delete the old candidate array
					candidates_d = newCandidates_d;

					checkCudaErrors(cudaMalloc((void**) &toBeDeleted_d, sizeOfToBeDeleted));

					
					// Find all the dublicate candidates
					// For the find dublicates call the output vector needs to be sat to 0,
					//    because it only writes the true values
					checkCudaErrors(cudaMemsetAsync(toBeDeleted_d, 0, sizeOfToBeDeleted, stream1_1));
					
					findDublicatesWrapper(ceilf((float)numberOfCandidates/dimBlock), dimBlock, stream1_1,
										  candidates_d, numberOfCandidates, dim,
										  deletedFromCount_d, toBeDeleted_d, this->duplicateKernelVerison);
					orKernelWrapper(ceilf((float)(numberOfCandidates+1)/dimBlock), dimBlock, stream1_1,
									numberOfCandidates+1, toBeDeleted_d, deletedFromCount_d);

					checkCudaErrors(cudaFree(deletedFromCount_d));
					// Compute the prefix sum and then find the number of dublicates
					sizeOfPrefixSum = (numberOfCandidates+1)*sizeof(unsigned int);
									checkCudaErrors(cudaMalloc((void**) &prefixSum_d,   sizeOfPrefixSum));
					sum_scan_blelloch(stream1_1, prefixSum_d,toBeDeleted_d,(numberOfCandidates+1), false);
				
					checkCudaErrors(cudaMemcpyAsync(sum_h, prefixSum_d+numberOfCandidates,
													sizeof(unsigned int), cudaMemcpyDeviceToHost,stream1_1));
					checkCudaErrors(cudaFree(toBeDeleted_d));

					// always sync streams before reading value from host
					checkCudaErrors(cudaStreamSynchronize(stream1_1)); 
					assert(numberOfCandidates >= sum_h[0]); // avoid underflow
				
					// Calculate the new number of candidates, based on the amount that should be deleted
					oldNumberOfCandidates = numberOfCandidates;
					numberOfCandidates = numberOfCandidates-sum_h[0];
					if(numberOfCandidates <= 0){
						break;
					}
										
					// Delete the dublicate candidates
					sizeOfCandidates = numberOfCandidates*numberOfBlocksPrPoint*sizeof(unsigned int);
					checkCudaErrors(cudaMalloc((void**) &newCandidates_d,   sizeOfCandidates));
					unsigned int delTrfmDataDimGrid = ceilf(((float)oldNumberOfCandidates*numberOfBlocksPrPoint)
															/dimBlock);
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

					checkCudaErrors(cudaMalloc((void**) &support_d,   sizeOfSupport));
					checkCudaErrors(cudaMalloc((void**) &score_d,   sizeOfScore));
					checkCudaErrors(cudaMalloc((void**) &toBeDeleted_d,   sizeOfToBeDeleted));

					countSupportWrapper(ceilf((float)numberOfCandidates/dimBlock), dimBlock, stream1_1,
										candidates_d, itemSet_d, dim, numberOfPoints, numberOfCandidates, minSupp,
										this->beta, support_d, score_d, toBeDeleted_d, this->countSupportKernelVersion);

					// Delete the candidates with support smaller the minSupp.
					sizeOfPrefixSum = (numberOfCandidates+1)*sizeof(unsigned int);
					checkCudaErrors(cudaMalloc((void**) &prefixSum_d,   sizeOfPrefixSum));
					sum_scan_blelloch(stream1_1, prefixSum_d, toBeDeleted_d,(numberOfCandidates+1), false);
					checkCudaErrors(cudaMemcpyAsync(sum_h, prefixSum_d+numberOfCandidates,
													sizeof(unsigned int), cudaMemcpyDeviceToHost,stream1_1));
					checkCudaErrors(cudaFree(toBeDeleted_d));
					checkCudaErrors(cudaStreamSynchronize(stream1_1));
					oldNumberOfCandidates = numberOfCandidates;

					assert(numberOfCandidates >= sum_h[0]); // avoid underflow
					numberOfCandidates = numberOfCandidates-sum_h[0];
					if(numberOfCandidates <= 0){
						break;
					}


					// Actually delete the candidates
					sizeOfCandidates = numberOfCandidates*numberOfBlocksPrPoint*sizeof(unsigned int);
					sizeOfScore = numberOfCandidates*sizeof(float);

					checkCudaErrors(cudaMalloc((void**) &newCandidates_d,   sizeOfCandidates));
					checkCudaErrors(cudaMalloc((void**) &newScore_d,   sizeOfScore));

					delTrfmDataDimGrid = ceilf((float)oldNumberOfCandidates*numberOfBlocksPrPoint/dimBlock);
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
					checkCudaErrors(cudaMalloc((void **) &index_d, numberOfCandidates*sizeof(unsigned int)));
					createIndicesKernel(ceilf((float)numberOfCandidates/dimBlock), dimBlock, stream1_1,
										index_d, numberOfCandidates);
					argMaxKernel(ceilf((float)numberOfCandidates/dimBlock), dimBlock, smemSize ,stream1_1,
								 score_d, index_d, numberOfCandidates);

					// Store the best in the arrays to look for dublicate arrays. 
					extractMaxWrapper(ceilf((float)numberOfBlocksPrPoint/dimBlock), dimBlock, stream1_1,
									  candidates_d, score_d, currentCentroidIndex, numberOfCandidates, index_d,
									  dim, bestCandidate_d+i*numberOfBlocksPrPoint, bestScore_d+i, bestCentroid_d+i);

					checkCudaErrors(cudaFree(index_d));
					checkCudaErrors(cudaFree(score_d));
				} // Apriori iterations
			}else{ // Number of candidates > 0 for outer
				checkCudaErrors(cudaMemsetAsync(bestScore_d+i, 0, sizeof(float), stream1_1));
				checkCudaErrors(cudaMemsetAsync(bestCentroid_d+i, 0, sizeof(unsigned int), stream1_1));

				
				checkCudaErrors(cudaFree(prefixSum_d));

				checkCudaErrors(cudaFree(support_d));
				checkCudaErrors(cudaFree(score_d));

			}
			checkCudaErrors(cudaFree(itemSet_d));
			checkCudaErrors(cudaFreeHost(sum_h));
			checkCudaErrors(cudaFree(candidates_d));
		}// for all centroids
		// Now all the centroids have been looked at, and it is time to find the best scoring disjoint clusters.

		bool* disjointClustersToBeRemoved_b_d; 
		checkCudaErrors(cudaMalloc((void**) &disjointClustersToBeRemoved_b_d,   sizeof(bool)*(numberOfCentroids+1)));
		if(this->isConcurentVersion()){
			// Delete the non-disjoint clusters, and keeping the largest of them all. 
			unsigned int* disjointClustersToBeRemoved_d;
			checkCudaErrors(cudaMalloc((void**) &disjointClustersToBeRemoved_d,
									   sizeof(unsigned int)*(numberOfCentroids+1)));
			unsigned int numberOfComparisons = (numberOfCentroids*(numberOfCentroids+1))/2 - numberOfCentroids;
			unsigned int dimGrid_disj = ceilf((float)numberOfComparisons/dimBlock);

			disjointClustersWrapper(dimGrid_disj, dimBlock, stream1_1, bestCentroid_d,
									bestScore_d, bestCandidate_d, data_d,
									numberOfCentroids, dim, width, disjointClustersToBeRemoved_d);


			unsignedIntToBoolArrayWrapper(ceilf((float)numberOfCentroids/dimBlock), dimBlock, stream1_1,
										  disjointClustersToBeRemoved_d, numberOfCentroids, 
										  disjointClustersToBeRemoved_b_d);

			checkCudaErrors(cudaFree(disjointClustersToBeRemoved_d));
			
		}else{ // just find the best cluster of all

			
			unsigned int* index_d;
			checkCudaErrors(cudaMalloc((void **) &index_d, numberOfCentroids*sizeof(unsigned int)));
			createIndicesKernel(ceilf((float)numberOfCentroids/dimBlock), dimBlock, stream1_1,
								index_d, numberOfCentroids);
			argMaxKernel(ceilf((float)numberOfCentroids/dimBlock), dimBlock,
						 smemSize ,stream1_1, bestScore_d, index_d, numberOfCentroids);
			indexToBoolVectorWrapper(ceilf((float)numberOfCentroids/dimBlock), dimBlock, stream1_1,
									 index_d, numberOfCentroids, disjointClustersToBeRemoved_b_d);
			checkCudaErrors(cudaFree(index_d));

		}


			
		unsigned int* prefixSum_d;
		size_t sizeOfPrefixSum = (numberOfCentroids+1)*sizeof(unsigned int);
		checkCudaErrors(cudaMalloc((void**) &prefixSum_d,   sizeOfPrefixSum));
		sum_scan_blelloch(stream1_1, prefixSum_d,disjointClustersToBeRemoved_b_d,(numberOfCentroids+1), true);

		checkCudaErrors(cudaFree(disjointClustersToBeRemoved_b_d));

		unsigned int* newNumberOfCentroids_h;
		checkCudaErrors(cudaMallocHost((void**) &newNumberOfCentroids_h, sizeof(unsigned int)));
		checkCudaErrors(cudaMemcpyAsync(newNumberOfCentroids_h, prefixSum_d+numberOfCentroids,
										sizeof(unsigned int), cudaMemcpyDeviceToHost,stream1_1));

		checkCudaErrors(cudaStreamSynchronize(stream1_1));
		// make sure that there can be no more centroids than there were before
		assert(numberOfCentroids >= newNumberOfCentroids_h[0]);
		unsigned int finalNumberOfClusters = numberOfCentroids-newNumberOfCentroids_h[0];
		checkCudaErrors(cudaFreeHost(newNumberOfCentroids_h));
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
		checkCudaErrors(cudaMalloc((void**) &finalScore_d,   finalNumberOfClusters*sizeof(float)));
		checkCudaErrors(cudaMalloc((void**) &finalCentroids_d,   finalNumberOfClusters*sizeof(unsigned int)));
		checkCudaErrors(cudaMalloc((void**) &finalCandidates_d,
								   finalNumberOfClusters*numberOfBlocksPrPoint*sizeof(unsigned int)));

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
		checkCudaErrors(cudaMalloc((void**) &outputCentroids_d,   dim*finalNumberOfClusters*sizeof(float)));
		copyCentroidWrapper(ceilf((float)finalNumberOfClusters*dim/dimBlock),dimBlock ,stream1_1,
							finalCentroids_d, data_d, dim, finalNumberOfClusters, outputCentroids_d);

		checkCudaErrors(cudaFree(finalCentroids_d));			


		// For each of the clusters found
		for(unsigned int i = 0; i < finalNumberOfClusters; i++){
			// Find the points in the i'th cluster
			bool* pointsContained_d;
			checkCudaErrors(cudaMalloc((void**) &pointsContained_d,   (numberOfPoints+1)*sizeof(bool)));
			findPointInClusterWrapper(ceilf((float)numberOfPoints/dimBlock), dimBlock, stream1_1,
									  finalCandidates_d+i*numberOfBlocksPrPoint, data_d, outputCentroids_d+i*dim, dim,
									  numberOfPoints, width, pointsContained_d);
					
			unsigned int* prefixSum_d;
			size_t sizeOfPrefixSum = (numberOfPoints+1)*sizeof(unsigned int);
			checkCudaErrors(cudaMalloc((void**) &prefixSum_d,   sizeOfPrefixSum));
			sum_scan_blelloch(stream1_1, prefixSum_d,pointsContained_d,(numberOfPoints+1), true);


			// Fetch the number of points in the cluster
			unsigned int* sum_h;
			checkCudaErrors(cudaMallocHost((void**) &sum_h, sizeof(unsigned int)));
			checkCudaErrors(cudaMemcpyAsync(sum_h, prefixSum_d+numberOfPoints, sizeof(unsigned int),
											cudaMemcpyDeviceToHost,stream1_1));
			checkCudaErrors(cudaStreamSynchronize(stream1_1));
			unsigned int oldNumberOfPoints = numberOfPoints;
			numberOfPoints = sum_h[0];
			// Create the output cluster
			float* outputCluster_d;
			float* outputCluster_h;
	
			size_t sizeOfOutputCluster = (oldNumberOfPoints - sum_h[0])*dim*sizeof(unsigned int);
			checkCudaErrors(cudaMalloc((void**) &outputCluster_d,   sizeOfOutputCluster));
		
			deleteFromArrayWrapper(ceilf((float)(oldNumberOfPoints*dim)/dimBlock), dimBlock, stream1_1,
								   data_d, prefixSum_d, oldNumberOfPoints, dim, outputCluster_d);
		


			// Delete the points already in the cluster.
			float* newData_d;
			size_t sizeOfNewData = numberOfPoints*dim*sizeof(unsigned int);
			sum_scan_blelloch(stream1_1, prefixSum_d,pointsContained_d,(oldNumberOfPoints+1), false);	
			checkCudaErrors(cudaMalloc((void**) &newData_d,   sizeOfNewData)); // This can be done at the same time as previous delete

			deleteFromArrayWrapper(ceilf((float)(oldNumberOfPoints*dim)/dimBlock),  dimBlock, stream1_1,
								   data_d, prefixSum_d, oldNumberOfPoints, dim, newData_d);

			checkCudaErrors(cudaFree(pointsContained_d));
			checkCudaErrors(cudaFree(prefixSum_d));
			checkCudaErrors(cudaMallocHost((void**) &outputCluster_h, sizeOfOutputCluster));
			checkCudaErrors(cudaMemcpyAsync(outputCluster_h, outputCluster_d,
											sizeOfOutputCluster, cudaMemcpyDeviceToHost, stream1_1));
			checkCudaErrors(cudaFree(outputCluster_d));

			unsigned int* bestCandidate_h;
			checkCudaErrors(cudaMallocHost((void**) &bestCandidate_h, numberOfBlocksPrPoint*sizeof(unsigned int)));
			checkCudaErrors(cudaMemcpyAsync(bestCandidate_h, finalCandidates_d+i*numberOfBlocksPrPoint,
											numberOfBlocksPrPoint*sizeof(unsigned int), cudaMemcpyDeviceToHost,
											stream1_1));

			// Make the output vectors.
			checkCudaErrors(cudaStreamSynchronize(stream1_1));

			auto outputCluster = new std::vector<std::vector<float>*>;
			for(unsigned int i = 0; i < oldNumberOfPoints - sum_h[0]; i++){
				auto point = new std::vector<float>;
				for(unsigned int j = 0; j < dim; j++){
					point->push_back(outputCluster_h[i*dim+j]);
				}
				outputCluster->push_back(point);
			}

			(cudaFreeHost(outputCluster_h));
			checkCudaErrors(cudaFree(data_d));
			auto outputDim = new std::vector<bool>;
			unsigned int count = 0;
			for(unsigned int i = 0; i < numberOfBlocksPrPoint;i++){
				for(unsigned int j = 0; j < 32; j++){
					outputDim->push_back((bestCandidate_h[i] >> j) & (unsigned int)1);
					count++;
					if(count >= dim){
						break;
					}
				}
			}
		   (cudaFreeHost(bestCandidate_h));
			
			result.push_back(std::make_pair(outputCluster,outputDim));

			checkCudaErrors(cudaFreeHost(sum_h));
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

	checkCudaErrors(cudaFree(data_d));
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
