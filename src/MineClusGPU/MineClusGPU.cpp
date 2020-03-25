#include "MineClusGPU.h"
#include <stdexcept>   // for exception, runtime_error, out_of_range
#include "../MineClus/MineClusKernels.h"
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


std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*>> MineClusGPU::findKClusters(int k){

	unsigned int numberOfPoints = this->data->size();
	unsigned int dim = this->data->at(0)->size();
	unsigned int numberOfBlocksPrPoint = ceilf((float)dim/32);
	unsigned int dimBlock = 1024;

	unsigned int minSupp = size*this->alpha;
	
	unsigned int sizeOfData = size*dim*sizeof(float);
	float* data_h = this->transformData();
	float* data_d;
	cudaMalloc((void**) &data_d, sizeOfData);


	size_t sizeOfInitialCandidates = dim*numberOfBlocksPrPoint*sizeof(unsigned int);
	unsigned int* initialCandidates_d;
	cudaMalloc((void**) &initialCandidates_d, sizeOfInitialCandidates);

	size_t sizeOfItemSet = numberOfPoints*numberOfBlocksPrPoint*sizeof(unsigned int);
	unsigned int* itemSet_d;
	cudaMalloc((void**) &itemSet_d, sizeOfItemSet);

	size_t sizeOfInitialSupport = dim*sizeof(unsigned int);
	size_t sizeOfInitialScore = dim*sizeof(float);
	size_t sizeOfInitialToBeDeleted = (dim+1)*sizeof(bool);

	unsigned int* initialSupport_d;
	float* initialScore_d;
	bool* initialToBeDeleted_d;

	cudaMalloc((void**) &initialSupport_d, sizeOfInitialSupport);
	cudaMalloc((void**) &initialScore_d, sizeOfInitialScore);
	cudaMalloc((void**) &initialToBeDeleted_d, sizeOfInitialToBeDeleted);


	size_t sizeOfPrefixSum = (dim+1)*sizeof(unsigned int);
	unsigned int* prefixSum_d;
	cudaMalloc((void**) &prefixSum_d, sizeOfPrefixSum);
	
	
	cudaStream_t stream1_1;
	cudaStreamCreate(&stream1_1);
	cudaStream_t stream1_2;
	cudaStreamCreate(&stream1_2);


	cudaMemcpyAsync(data_d, data_h, sizeOfData, cudaMemcpyHostToDevice, stream1_2);
	
	createItemSetWrapper(ceilf((float)size/dimBlock), dimBlock, stream1_2, data_d, dim, numberOfPoints, 0, this->width, itemSet_d); /// OBS hardcoded to centroid 0
	createInitialCandidatesWrapper(ceilf((float)dim/dimBlock), dimBlock, stream1_1, dim, initialCandidates_d);
	cudaStreamSynchronize(stream1_1);
	cudaStreamSynchronize(stream1_2);

	countSupportWrapper(ceilf((float)dim/dimBlock), dimBlock, stream1_1, initialCandidates_d, itemSet_d, dim, numberOfPoints, dim, minSupp, this->beta, initialSupport_d, initialScore_d, initialToBeDeleted_d);

	sum_scan_blelloch(stream1_1, prefixSum_d,initialToBeDeleted_d,(dim+1), false);

	unsigned int* sum_h;
	cudaMallocHost((void**) &sum_h, sizeof(unsigned int));
	cudaMemcpyAsync(sum_h, prefixSum_d+dim, sizeof(unsigned int), cudaMemcpyDeviceToHost,stream1_1);
	cudaStreamSynchronize(stream1_1);

	unsigned int sizeAfterDelete = dim-sum_h[0];
	std::cout << "sizeAfterDelete: " << sizeAfterDelete << std::endl;

	
	unsigned int* h_block_sums = new unsigned int[dim+1];
	checkCudaErrors(cudaMemcpy(h_block_sums, prefixSum_d, sizeof(unsigned int) * (dim+1), cudaMemcpyDeviceToHost));
	std::cout << "Block sums: ";
	for (int i = 0; i < dim+1; ++i)
	{
		std::cout << h_block_sums[i] << ", ";
	}
	std::cout << std::endl;
	std::cout << "Block sums length: " << dim+1 << std::endl;

			   

			   
	
	

						 
	
	
	
	cudaStreamDestroy(stream1_1);
	cudaStreamDestroy(stream1_2);

	throw std::runtime_error("Not implemented Yet");
	
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
