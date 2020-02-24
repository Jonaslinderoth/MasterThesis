/*
 * DOCGPU.cpp
 *
 *  Created on: Feb 14, 2020
 *      Author: mikkel
 */

#include "DOCGPU.h"
#include "DOCGPU_Kernels.h"
#include <assert.h>
#include "../randomCudaScripts/DeleteFromArray.h"

# define CUDA_CALL ( x) do { if (( x) != cudaSuccess ) { \
	printf (" Error at % s :% d\ n" , __FILE__ , __LINE__ ) ;\
return EXIT_FAILURE ;}} while (0)
# define CURAND_CALL ( x) do { if (( x) != CURAND_STATUS_SUCCESS ) { \
printf (" Error at % s :% d\ n" , __FILE__ , __LINE__ ) ;\
return EXIT_FAILURE ;}} while (0)


DOCGPU::DOCGPU(std::vector<std::vector<float>*>* input, float alpha, float beta, float width) {
	this->data = input;
	this->alpha = alpha;
	this->width = width;
	this->beta = beta;
	this->size = data->size();
	this->dim = data->at(0)->size();
}


DOCGPU::~DOCGPU() {
	// TODO Auto-generated destructor stub
}


std::vector<std::vector<float>*>* DOCGPU::initDataReader(DataReader* dr){
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
float* DOCGPU::transformData(){
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


std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> DOCGPU::findCluster(){
	auto data = this->data;
	auto alpha = this->alpha;
	auto beta = this->beta;
	auto width = this->width;
	
	unsigned int d = data->at(0)->size();
	unsigned int r = log2(2*d)/log2(1/(2*beta));
	if(r == 0) r = 1;
	unsigned int m = pow((2/alpha),r) * log(4);
	
	unsigned int number_of_ps = 2.0/alpha;
	unsigned int number_of_samples = number_of_ps*m;
	
	unsigned int sample_size = r;
	unsigned int number_of_points = data->size();
	unsigned int point_dim = d;
	
	unsigned int floats_in_data_array = point_dim*number_of_points;
	unsigned int numbers_in_Xs_array = number_of_samples*sample_size;

	unsigned int size_of_data = floats_in_data_array*sizeof(float);
	unsigned int size_of_ps = number_of_ps*sizeof(unsigned int);
	unsigned int size_of_xs = numbers_in_Xs_array*sizeof(unsigned int);
	unsigned int size_of_output_dims = point_dim*sizeof(bool);
	unsigned int size_of_output_cluster = number_of_points*sizeof(bool);



	
	float* data_h = transformData();
	

	float* data_d;
	unsigned int* ps_d;
	unsigned int* xs_d;
	

	cudaMalloc((void **) &data_d, size_of_data);
	cudaMalloc((void **) &ps_d, size_of_ps);
	cudaMalloc((void **) &xs_d, size_of_xs);

	cudaMemcpy(data_d, data_h, size_of_data, cudaMemcpyHostToDevice);


	
	curandState* randomStates_d;
	size_t numberOfRandomStates = 1024*10;// this parameter could be nice to test to find "optimal one"...
	
	cudaMalloc((void**)&randomStates_d, sizeof(curandState) * numberOfRandomStates);
	generateRandomStatesArray(randomStates_d,numberOfRandomStates);
	
	generateRandomIntArrayDevice(xs_d, randomStates_d , numberOfRandomStates,
								 numbers_in_Xs_array , this->data->size()-1 , 0);
	generateRandomIntArrayDevice(ps_d, randomStates_d , numberOfRandomStates, number_of_ps , this->data->size()-1 , 0);

	assert(numbers_in_Xs_array >= number_of_ps);
	

	cudaFreeHost(data_h);
	
	unsigned int findDim_bools = number_of_samples*point_dim;
	unsigned int number_of_points_contained = number_of_samples*number_of_points;

	unsigned int size_of_findDim = findDim_bools*sizeof(bool);
	unsigned int size_of_pointsContained = number_of_points_contained*sizeof(bool);
	
	unsigned int size_of_findDim_count = number_of_samples*sizeof(unsigned int);
	unsigned int size_of_pointsContained_count = number_of_samples*sizeof(unsigned int);
	unsigned int size_of_scores = number_of_samples*sizeof(float);
	unsigned int size_of_scores_index = number_of_samples*sizeof(unsigned int);


	bool* findDim_output_d;
	bool* pointsContained_output_d;
	unsigned int* pointsContained_count_d;
	unsigned int* findDim_count_d;
	float* scores_d;
	unsigned int* scores_index_d;
	
	
	cudaMalloc((void **) &findDim_output_d, size_of_findDim);
	cudaMalloc((void **) &pointsContained_output_d, size_of_pointsContained);
	cudaMalloc((void **) &findDim_count_d, size_of_findDim_count);
	cudaMalloc((void **) &pointsContained_count_d, size_of_pointsContained_count);
	cudaMalloc((void **) &scores_d, size_of_scores);
	cudaMalloc((void **) &scores_index_d, size_of_scores_index );
	

	int smemSize, maxBlock;
	cudaDeviceGetAttribute(&smemSize, 
    cudaDevAttrMaxSharedMemoryPerBlock, 0);
	cudaDeviceGetAttribute(&maxBlock, 
						   cudaDevAttrMaxThreadsPerBlock, 0); 

	// caluclate the maximum thread size based on shared mem requirements and maximum threads
	int dimBlock = smemSize/(sizeof(int)+sizeof(float));
	if(dimBlock > maxBlock) dimBlock = maxBlock;
	int dimGrid = ceil((float)number_of_samples/(float)dimBlock);
	int sharedMemSize = (dimBlock*sizeof(unsigned int) + dimBlock*sizeof(float));

	
	
	findDimmensionsKernel(dimGrid, dimBlock, xs_d, ps_d, data_d,  findDim_output_d,
						   findDim_count_d, point_dim, number_of_samples, sample_size, number_of_ps,
						  m, width);
	cudaFree(xs_d);


	pointsContainedKernel(dimGrid, dimBlock,data_d, ps_d, findDim_output_d,
						  pointsContained_output_d, pointsContained_count_d,
						  width, point_dim, number_of_points, number_of_samples, m);


	cudaFree(ps_d);
	cudaFree(data_d);
	
	scoreKernel(dimGrid, dimBlock,
				pointsContained_count_d,
				findDim_count_d, scores_d,
				number_of_samples,
				alpha, beta, number_of_points);
	

	cudaFree(findDim_count_d);
	cudaFree(pointsContained_count_d);



	createIndicesKernel(dimGrid, dimBlock,scores_index_d, number_of_samples);

	
	argMaxKernel(dimGrid, dimBlock, sharedMemSize, scores_d, scores_index_d, number_of_samples);


	unsigned int size_of_output = sizeof(unsigned int);
	unsigned int* scores_index_h = (unsigned int*) malloc(size_of_output);
	cudaMemcpy(scores_index_h, scores_index_d, size_of_output, cudaMemcpyDeviceToHost);

	unsigned int size_of_score_output = sizeof(float);
	float* scores_out_h = (float*) malloc(size_of_output);
	cudaMemcpy(scores_out_h, scores_d, size_of_score_output, cudaMemcpyDeviceToHost);
	
	cudaFree(scores_d);

	
	bool* output_dims_h = (bool*) malloc(size_of_output_dims);
	bool* output_cluster_h = (bool*) malloc(size_of_output_cluster);


	std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> result;

	// if the score is zero the cluster must be empty
	if(fabs(scores_out_h[0]) < 0.0001){
		result = make_pair(new std::vector<std::vector<float>*>, new std::vector<bool>);
		
	}else{
		cudaMemcpy(output_dims_h, findDim_output_d+(scores_index_h[0]*point_dim),
				   size_of_output_dims, cudaMemcpyDeviceToHost);

		
		cudaMemcpy(output_cluster_h, pointsContained_output_d+(scores_index_h[0]*number_of_points),
				   size_of_output_cluster, cudaMemcpyDeviceToHost);


		std::vector<bool>* outDim = new std::vector<bool>;
		for(int i = 0; i < point_dim; i++){
			outDim->push_back(output_dims_h[i]);
		}
	

		std::vector<std::vector<float>*>* outCluster = new std::vector<std::vector<float>*>;

		for(int i = 0; i < number_of_points; i++){
			if(output_cluster_h[i]){
				outCluster->push_back(data->at(i));
			}
		}	

		result = std::make_pair(outCluster, outDim);

	}
	cudaFree(scores_index_d);
	cudaFree(findDim_output_d);
	cudaFree(pointsContained_output_d);
	free(scores_index_h);
	free(scores_out_h);
	free(output_dims_h);
	free(output_cluster_h);
	
	return result;
};

std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*>> DOCGPU::findKClusters(int k){
	float* data_h = this->transformData();
	uint dim = this->data->at(0)->size();
	auto alpha = this->alpha;
	auto beta = this->beta;
	auto width = this->width;

	// calculating algorithm parameters
	unsigned int d = data->at(0)->size();
	unsigned int r = log2(2*d)/log2(1/(2*beta));
	if(r == 0) r = 1;
	unsigned int m = pow((2/alpha),r) * log(4);
	unsigned int number_of_points = this->data->size();
	unsigned int size_of_data = number_of_points*dim*sizeof(float);


	// Calculating the sizes of the random samples
	unsigned int number_of_centroids = 2.0/alpha;
	unsigned int number_of_samples = number_of_centroids*m;
	unsigned int sample_size = r;
	unsigned int number_of_values_in_samples = number_of_samples*sample_size;
	unsigned int size_of_samples = number_of_values_in_samples*sizeof(unsigned int);
	unsigned int size_of_centroids = number_of_centroids*sizeof(unsigned int);

	// calculating the sizes for findDim 
	unsigned int number_of_bools_for_findDim = number_of_samples*dim;
	// the +1 is to allocate one more for delete, the value do not matter,
	// but is used to avoid reading outside the memory space
	unsigned int size_of_findDim = (number_of_bools_for_findDim+1)*sizeof(bool); 
	unsigned int number_of_values_find_dim_count_output = number_of_samples;
	unsigned int size_of_findDim_count = number_of_values_find_dim_count_output* sizeof(unsigned int);


	// calculating sizes for pointsContained
	unsigned int number_of_values_in_pointsContained = number_of_samples*number_of_points;
	unsigned int size_of_pointsContained = number_of_values_in_pointsContained*sizeof(bool);
	unsigned int number_of_values_in_pointsContained_count = number_of_samples;
	unsigned int size_of_pointsContained_count = number_of_values_in_pointsContained_count*sizeof(unsigned int);

	//calculating sizes for score
	unsigned int number_of_score = number_of_samples;
	unsigned int size_of_score = number_of_score*sizeof(float);

	//calculating sizes for indecies
	unsigned int number_of_indecies = number_of_samples;
	unsigned int size_of_index = number_of_indecies*sizeof(unsigned int);


	//calculating dimensions for threads
	int smemSize, maxBlock;
	cudaDeviceGetAttribute(&smemSize, 
    cudaDevAttrMaxSharedMemoryPerBlock, 0);
	cudaDeviceGetAttribute(&maxBlock, 
						   cudaDevAttrMaxThreadsPerBlock, 0); 

	// caluclate the maximum thread size based on shared mem requirements and maximum threads
	unsigned int dimBlock = smemSize/(sizeof(int)+sizeof(float));
	if(dimBlock > maxBlock) dimBlock = maxBlock;
	unsigned int dimGrid = ceil((float)number_of_samples/(float)dimBlock);
	unsigned int sharedMemSize = (dimBlock*sizeof(unsigned int) + dimBlock*sizeof(float));



	
	
	
	// allocating memory for random samples
	unsigned int* centroids_d;
	unsigned int* samples_d;	
	cudaMalloc((void **) &samples_d, size_of_samples);
	cudaMalloc((void **) &centroids_d, size_of_centroids);

	curandState* randomStates_d;
	size_t numberOfRandomStates = 1024*10;// this parameter could be nice to test to find "optimal one"...
	
	cudaMalloc((void**)&randomStates_d, sizeof(curandState) * numberOfRandomStates);
	generateRandomStatesArray(randomStates_d,numberOfRandomStates);

	float* data_d;
	cudaMalloc((void **) &data_d, size_of_data);
	
	cudaMemcpy(data_d, data_h, size_of_data, cudaMemcpyHostToDevice);
	
	cudaFreeHost(data_h); // the data is not used on the host side any more


	// allocating memory for findDim
	bool* findDim_d;
	unsigned int* findDim_count_d;
	cudaMalloc((void **) &findDim_d, size_of_findDim);
	cudaMalloc((void **) &findDim_count_d, size_of_findDim_count);

	// allocating memory for pointsContained
	bool* pointsContained_d;
	unsigned int* pointsContained_count_d;
	cudaMalloc((void **) &pointsContained_d, size_of_pointsContained);
	cudaMalloc((void **) &pointsContained_count_d, size_of_pointsContained_count);

	//allocating memory for score
	float* score_d;
	cudaMalloc((void **) &score_d, size_of_score);

	//allocating memory for index
	unsigned int* index_d;
	cudaMalloc((void **) &index_d, size_of_index);
	std::cout << "size_of_index: " << size_of_index << std::endl;


	
	
	for(int i = 0; i < k; k++){ // for each of the clusters
		
		generateRandomIntArrayDevice(samples_d, randomStates_d , numberOfRandomStates,
									 number_of_values_in_samples , number_of_points-1, 0);
		
		generateRandomIntArrayDevice(centroids_d, randomStates_d , numberOfRandomStates,
									 number_of_centroids , number_of_points-1 , 0);

		assert(number_of_values_in_samples >= number_of_centroids);


		// Find dimmnsions
		findDimmensionsKernel(dimGrid, dimBlock, samples_d, centroids_d, data_d,  findDim_d,
							  findDim_count_d, dim, number_of_samples, sample_size, number_of_centroids,
							  m, width);
		
		// Find points contained
		pointsContainedKernel(dimGrid, dimBlock, data_d, centroids_d, findDim_d,
							  pointsContained_d, pointsContained_count_d,
							  width, dim, number_of_points, number_of_samples, m);
		
		// Calculate scores
		scoreKernel(dimGrid, dimBlock,
					pointsContained_count_d,
					findDim_count_d, score_d,
					number_of_samples,
					alpha, beta, number_of_points);
	
		// Create indices for the scores
		createIndicesKernel(dimGrid, dimBlock, index_d, number_of_samples);
		
		// Find highest scoring
		argMaxKernel(dimGrid, dimBlock, sharedMemSize, score_d, index_d, number_of_samples);
		
		// Output Points in that cluster, along with the dimensions and centroid
		unsigned int* best_index_h = (unsigned int*) malloc(sizeof(unsigned int));
		cudaMemcpy(best_index_h, index_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);

		unsigned int* cluster_size_h = (unsigned int*) malloc(sizeof(unsigned int));
		cudaMemcpy(cluster_size_h, pointsContained_count_d+(best_index_h[0]*number_of_points),
				   sizeof(unsigned int), cudaMemcpyDeviceToHost);

		unsigned int size_of_cluster = cluster_size_h[0]*dim*sizeof(float);
		float* cluster_d;
		cudaMalloc((void **) &cluster_d, size_of_cluster);

		std::cout<< "best_index: " << best_index_h[0] << ", size_of_best_cluster: " << cluster_size_h[0] << std::endl;
		
		deleteFromArray(cluster_d, findDim_d+(best_index_h[0]*dim), data_d, number_of_points, dim);

		
		float* cluster_h;
		cudaMallocHost((void**) &cluster_h, size_of_cluster);
		cudaMemcpy(cluster_h, cluster_d, size_of_cluster, cudaMemcpyDeviceToHost);
		cudaFree(cluster_d);


	
		for(int a = 0; a < cluster_size_h[0]; a++){
			for(int b = 0; b < dim; b++){
				std::cout << cluster_h[a*dim+b] << ", ";
			}
			std::cout << std::endl;
		}
		
		// Delete the points from the original dataset


		// set number of points to the new data size.

		break;
	}

	
	throw std::runtime_error("Not implemented yet");	
};

