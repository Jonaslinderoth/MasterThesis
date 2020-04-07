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

	while(k > result.size()){
		// Allocate the space for the best scores
		float* bestScore_d;
		unsigned int* bestCentroid_d;
		unsigned int* bestCandidate_d;
		cudaMalloc((void**) &bestScore_d, numberOfCentroids*sizeof(float));
		cudaMalloc((void**) &bestCentroid_d, numberOfCentroids*sizeof(unsigned int));
		cudaMalloc((void**) &bestCandidate_d, numberOfCentroids*numberOfBlocksPrPoint*sizeof(unsigned int));

		// Create long living variables
		unsigned int currentCentroidIndex = 0;
	
		for(unsigned int i = 0; i < numberOfCentroids; i++){
			// The current centroid index
			currentCentroidIndex = this->randInt(0,numberOfPoints-1,1).at(0);
	
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
		
			// std::cout << "numberOfCandidates: " << numberOfCandidates << std::endl; 
			//if there are any candidates left
			if(numberOfCandidates > 0){
			
				// delete all the small candidates
				unsigned int* newCandidates_d;
				sizeOfCandidates = dim*numberOfBlocksPrPoint*sizeof(unsigned int);
				checkCudaErrors(cudaMalloc((void**) &newCandidates_d, sizeOfCandidates));
				deleteFromArrayTransfomedDataWrapper(ceilf((float)dim/32), dimBlock, stream1_1,
													 candidates_d, prefixSum_d, oldNumberOfCandidates, numberOfBlocksPrPoint, newCandidates_d);

				// std::cout << "numberOfBlocksPrPoint: " << numberOfBlocksPrPoint << std::endl;
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
					cudaMallocHost((void**) &bestScore_h, sizeof(float)*numberOfCentroids);
					for(int j = 0; j < numberOfCentroids; j++){
						bestScore_h[j] = 0;
					}
					cudaMemcpy(bestScore_d, bestScore_h, sizeof(float)*numberOfCentroids, cudaMemcpyHostToDevice);				
				}

				// find the best score, then copy that score and the coresponding candidate and centroid to the result
				extractMaxWrapper(ceilf((float)numberOfBlocksPrPoint/dimBlock), dimBlock, stream1_1,
								  candidates_d, score_d, currentCentroidIndex, numberOfCandidates, index_d,
								  dim, bestCandidate_d+i*numberOfBlocksPrPoint, bestScore_d+i, bestCentroid_d+i);

			
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
				
					// Find all the dublicate candidates 
					checkCudaErrors(cudaMemsetAsync(toBeDeleted_d, 0, sizeOfToBeDeleted, stream1_1)); // not needed any more, it is handled by the merge kernel now
					findDublicatesWrapper(ceilf((float)numberOfCandidates/dimBlock), dimBlock, stream1_1, candidates_d, numberOfCandidates, dim, deletedFromCount_d, toBeDeleted_d, MoreBreaking);
					orKernelWrapper(ceilf((float)numberOfCandidates+1/dimBlock), dimBlock, stream1_1, numberOfCandidates+1, toBeDeleted_d, deletedFromCount_d);

					//toBeDeleted_d = deletedFromCount_d;


					// Compute the prefix sum and then find the number of dublicates
					sizeOfPrefixSum = (numberOfCandidates+1)*sizeof(unsigned int);
					checkCudaErrors(cudaMalloc((void**) &prefixSum_d, sizeOfPrefixSum));
					sum_scan_blelloch(stream1_1, prefixSum_d,toBeDeleted_d,(numberOfCandidates+1), false);
				
					checkCudaErrors(cudaMemcpyAsync(sum_h, prefixSum_d+numberOfCandidates, sizeof(unsigned int), cudaMemcpyDeviceToHost,stream1_1));
					checkCudaErrors(cudaStreamSynchronize(stream1_1));
					assert(numberOfCandidates >= sum_h[0]); // avoid underflow
				
					// delete the dublicate candidates
					checkCudaErrors(cudaStreamSynchronize(stream1_1));
					oldNumberOfCandidates = numberOfCandidates;g
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

					// clean up after the deletion
					checkCudaErrors(cudaFree(candidates_d));
					checkCudaErrors(cudaFree(toBeDeleted_d));
					checkCudaErrors(cudaFree(prefixSum_d));
				
					candidates_d = newCandidates_d;
					score_d = newScore_d;


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


				
					// delete small candidates
					sizeOfPrefixSum = (numberOfCandidates+1)*sizeof(unsigned int);
					checkCudaErrors(cudaMalloc((void**) &prefixSum_d, sizeOfPrefixSum));
					sum_scan_blelloch(stream1_1, prefixSum_d, toBeDeleted_d,(numberOfCandidates+1), false);
					checkCudaErrors(cudaMemcpyAsync(sum_h, prefixSum_d+numberOfCandidates, sizeof(unsigned int), cudaMemcpyDeviceToHost,stream1_1));
				
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


		
					// find the best 
					checkCudaErrors(cudaMalloc((void **) &index_d, numberOfCandidates*sizeof(unsigned int)));
					createIndicesKernel(ceilf((float)numberOfCandidates/dimBlock), dimBlock, stream1_1, index_d, numberOfCandidates);
					argMaxKernel(ceilf((float)numberOfCandidates/dimBlock), dimBlock, smemSize ,stream1_1, score_d, index_d, numberOfCandidates);


					extractMaxWrapper(ceilf((float)numberOfBlocksPrPoint/dimBlock), dimBlock, stream1_1,
									  candidates_d, score_d, currentCentroidIndex, numberOfCandidates, index_d,
									  dim, bestCandidate_d+i*numberOfBlocksPrPoint, bestScore_d+i, bestCentroid_d+i);

					checkCudaErrors(cudaFree(index_d));
					checkCudaErrors(cudaFree(score_d));
				}
			}
		}


	

		unsigned int* disjointClustersToBeRemoved_d; 
		cudaMalloc((void**) &disjointClustersToBeRemoved_d, sizeof(unsigned int)*(numberOfCentroids+1));
		unsigned int numberOfComparisons = (numberOfCentroids*(numberOfCentroids+1))/2 - numberOfCentroids;
		unsigned int dimGrid_disj = ceilf((float)numberOfComparisons/dimBlock);
		disjointClustersWrapper(dimGrid_disj, dimBlock, stream1_1, bestCentroid_d,
								bestScore_d, bestCandidate_d, data_d,
								numberOfCentroids, dim, width, disjointClustersToBeRemoved_d);

		bool* disjointClustersToBeRemoved_b_d; 
		cudaMalloc((void**) &disjointClustersToBeRemoved_b_d, sizeof(bool)*(numberOfCentroids+1));
		unsignedIntToBoolArrayWrapper(ceilf((float)numberOfCentroids/dimBlock), dimBlock, stream1_1, disjointClustersToBeRemoved_d, numberOfCentroids, disjointClustersToBeRemoved_b_d);
	
	
		unsigned int* prefixSum_d;
		size_t sizeOfPrefixSum = (numberOfCentroids+1)*sizeof(unsigned int);
		checkCudaErrors(cudaMalloc((void**) &prefixSum_d, sizeOfPrefixSum));
		sum_scan_blelloch(stream1_1, prefixSum_d,disjointClustersToBeRemoved_b_d,(numberOfCentroids+1), true);


		unsigned int* newNumberOfCentroids_h;
		checkCudaErrors(cudaMallocHost((void**) &newNumberOfCentroids_h, sizeof(unsigned int)));
		checkCudaErrors(cudaMemcpyAsync(newNumberOfCentroids_h, prefixSum_d+numberOfCentroids, sizeof(unsigned int), cudaMemcpyDeviceToHost,stream1_1));
		checkCudaErrors(cudaStreamSynchronize(stream1_1));
		assert(numberOfCentroids > newNumberOfCentroids_h[0]);
		unsigned int finalNumberOfClusters = numberOfCentroids-newNumberOfCentroids_h[0];


		float* finalScore_d;
		unsigned int* finalCentroids_d;
		unsigned int* finalCandidates_d;
		checkCudaErrors(cudaMalloc((void**) &finalScore_d, finalNumberOfClusters*sizeof(float)));
		checkCudaErrors(cudaMalloc((void**) &finalCentroids_d, finalNumberOfClusters*sizeof(unsigned int)));
		checkCudaErrors(cudaMalloc((void**) &finalCandidates_d, finalNumberOfClusters*numberOfBlocksPrPoint*sizeof(unsigned int)));

	
		deleteFromArrayWrapper(ceilf((float)(numberOfCentroids)/dimBlock), dimBlock, stream1_1, bestCentroid_d, prefixSum_d, numberOfCentroids, 1, finalCentroids_d);
		deleteFromArrayWrapper(ceilf((float)(numberOfCentroids)/dimBlock), dimBlock, stream1_1, bestScore_d, prefixSum_d, numberOfCentroids, 1, finalScore_d);
		deleteFromArrayWrapper(ceilf((float)(numberOfCentroids*numberOfBlocksPrPoint)/dimBlock), dimBlock, stream1_1, bestCandidate_d, prefixSum_d, numberOfCentroids, numberOfBlocksPrPoint, finalCandidates_d);

	

		float* outputCentroids_d;
		checkCudaErrors(cudaMalloc((void**) &outputCentroids_d, dim*finalNumberOfClusters*sizeof(float)));
		copyCentroidWrapper(ceilf((float)finalNumberOfClusters*dim/dimBlock),dimBlock ,stream1_1, finalCentroids_d, data_d, dim, finalNumberOfClusters, outputCentroids_d);
		checkCudaErrors(cudaFree(finalCentroids_d));
	

		for(unsigned int i = 0; i < finalNumberOfClusters; i++){
		
			bool* pointsContained_d;
			cudaMalloc((void**) &pointsContained_d, (numberOfPoints+1)*sizeof(bool));
		
			findPointInClusterWrapper(ceilf((float)numberOfPoints/dimBlock), dimBlock, stream1_1,
									  finalCandidates_d+i*numberOfBlocksPrPoint, data_d, outputCentroids_d+i*dim, dim,
									  numberOfPoints, width, pointsContained_d);
		
			checkCudaErrors(cudaStreamSynchronize(stream1_1));

	
			unsigned int* prefixSum_d;
			size_t sizeOfPrefixSum = (numberOfPoints+1)*sizeof(unsigned int);
			checkCudaErrors(cudaMalloc((void**) &prefixSum_d, sizeOfPrefixSum));
			sum_scan_blelloch(stream1_1, prefixSum_d,pointsContained_d,(numberOfPoints+1), true);


			unsigned int* sum_h;
			checkCudaErrors(cudaMallocHost((void**) &sum_h, sizeof(unsigned int)));
		
			checkCudaErrors(cudaMemcpyAsync(sum_h, prefixSum_d+numberOfPoints, sizeof(unsigned int), cudaMemcpyDeviceToHost,stream1_1));
			checkCudaErrors(cudaStreamSynchronize(stream1_1));
			unsigned int oldNumberOfPoints = numberOfPoints;
			numberOfPoints = sum_h[0];


			float* outputCluster_d;
			float* outputCluster_h;
	
			size_t sizeOfOutputCluster = (oldNumberOfPoints - sum_h[0])*dim*sizeof(unsigned int);

			checkCudaErrors(cudaMalloc((void**) &outputCluster_d, sizeOfOutputCluster));
		
			deleteFromArrayWrapper(ceilf((float)(oldNumberOfPoints*dim)/dimBlock), dimBlock, stream1_1, data_d, prefixSum_d, oldNumberOfPoints, dim, outputCluster_d);
		
			checkCudaErrors(cudaStreamSynchronize(stream1_1));
		

			float* newData_d;
			size_t sizeOfNewData = numberOfPoints*dim*sizeof(unsigned int);
			sum_scan_blelloch(stream1_1, prefixSum_d,pointsContained_d,(oldNumberOfPoints+1), false);	
			checkCudaErrors(cudaMalloc((void**) &newData_d, sizeOfNewData));
			deleteFromArrayWrapper(ceilf((float)oldNumberOfPoints*dim/dimBlock),  dimBlock, stream1_1, data_d, prefixSum_d, oldNumberOfPoints, dim, newData_d);

			checkCudaErrors(cudaMallocHost((void**) &outputCluster_h, sizeOfOutputCluster));
			cudaMemcpyAsync(outputCluster_h, outputCluster_d, sizeOfOutputCluster, cudaMemcpyDeviceToHost, stream1_1);

			unsigned int* bestCandidate_h;
			checkCudaErrors(cudaMallocHost((void**) &bestCandidate_h, numberOfBlocksPrPoint*sizeof(unsigned int)));
			checkCudaErrors(cudaMemcpyAsync(bestCandidate_h, finalCandidates_d+i*numberOfBlocksPrPoint, numberOfBlocksPrPoint*sizeof(unsigned int), cudaMemcpyDeviceToHost, stream1_1));
	
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
					count++;
					if(count >= dim){
						break;
					}
				}
			}

			result.push_back(std::make_pair(outputCluster,outputDim));
			checkCudaErrors(cudaStreamSynchronize(stream1_2));
			cudaFree(data_d);
			data_d = newData_d;
		}
	}

	cudaStreamDestroy(stream1_1);
	cudaStreamDestroy(stream1_2);


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
