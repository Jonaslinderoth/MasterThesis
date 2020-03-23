#include "MineClusGPU.h"
#include <stdexcept>   // for exception, runtime_error, out_of_range



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
