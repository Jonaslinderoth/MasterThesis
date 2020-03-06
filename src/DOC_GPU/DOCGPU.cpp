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
#include "MemSolver.h"

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


/**
 * Used for loading the data from the data reader into main memory.
 */
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


/**
 * Find a single cluster, uses the function to find multiple clusters
 */
std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> DOCGPU::findCluster(){
	auto result = findKClusters(1).at(0);
	return result;
};


std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*>> DOCGPU::findKClusters(int k){
	auto result = std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*>>();
	float* data_h = this->transformData();
	uint dim = this->data->at(0)->size();
	auto alpha = this->alpha;
	auto beta = this->beta;
	auto width = this->width;

	// calculating algorithm parameters
	unsigned int r = log2(2*dim)/log2(1/(2*beta));
	if(r == 0) r = 1;
	unsigned int m = pow((2/alpha),r) * log(4);
	unsigned int number_of_points = this->data->size();
	size_t numberOfRandomStates = 1024*10;

	// Calculating the sizes of the random samples
	unsigned int number_of_centroids = 2.0/alpha;
	unsigned int sample_size = r;

	size_t memoryCap = (size_t)10*(size_t)1000*(size_t)1000*(size_t)1000;

	
	Memory_sizes sizes = MemSolver::computeForAllocations(dim, number_of_points, number_of_centroids, m, sample_size, k, memoryCap);
	Array_sizes arr_sizes = MemSolver::computeArraySizes(sizes.first_number_of_centroids, number_of_points, m, sample_size);

	
	// calculating the sizes for findDim 
	/* size_t number_of_bools_for_findDim = number_of_samples*dim; 
	size_t size_of_findDim = number_of_bools_for_findDim*sizeof(bool);
	this->allocated_size_of_findDim = size_of_findDim;
	size_t number_of_values_find_dim_count_output = number_of_samples; 
	size_t size_of_findDim_count = number_of_values_find_dim_count_output* sizeof(unsigned int);
	this->allocated_size_of_findDim_count = size_of_findDim_count;

	// calculating sizes for pointsContained
	size_t number_of_values_in_pointsContained = (size_t)number_of_samples*(size_t)number_of_points; 
	// the +1 is to allocate one more for delete, the value do not matter,
	// but is used to avoid reading outside the memory space
	size_t size_of_pointsContained = (number_of_values_in_pointsContained+1)*sizeof(bool);
	this->allocated_size_of_pointsContained = size_of_pointsContained;
	size_t number_of_values_in_pointsContained_count = number_of_samples; 
	size_t size_of_pointsContained_count = number_of_values_in_pointsContained_count*sizeof(unsigned int);
	this->allocated_size_of_pointsContained_count = size_of_pointsContained_count;
	
	//calculating sizes for score
	size_t number_of_score = number_of_samples; 
	size_t size_of_score = number_of_score*sizeof(float);
	this->allocated_size_of_score = size_of_score;

	//calculating sizes for indecies
	size_t number_of_indecies = number_of_samples; 
	size_t size_of_index = number_of_indecies*sizeof(unsigned int);
	this->allocated_size_of_index = size_of_index;

	//sizes for random states

	size_t numberOfRandomStates = 1024*10;// this parameter could be nice to test to find "optima one"...
	size_t size_of_randomStates = sizeof(curandState) * numberOfRandomStates;
	this->allocated_size_of_randomStates = size_of_randomStates;

	size_t size_of_bestDims = (number_of_points+1)*sizeof(bool);
	this->allocated_size_of_bestDims = size_of_bestDims;
*/
	
	//calculating dimensions for threads
	
	int smemSize, maxBlock;
	cudaDeviceGetAttribute(&smemSize, 
    cudaDevAttrMaxSharedMemoryPerBlock, 0);
	cudaDeviceGetAttribute(&maxBlock, 
						   cudaDevAttrMaxThreadsPerBlock, 0); 

	// caluclate the maximum thread size based on shared mem requirements and maximum threads
	unsigned int dimBlock = smemSize/(sizeof(int)+sizeof(float));
	if(dimBlock > maxBlock) dimBlock = maxBlock;
	unsigned int dimGrid = ceil((float)arr_sizes.number_of_samples/(float)dimBlock); /**/
	unsigned int sharedMemSize = (dimBlock*sizeof(unsigned int) + dimBlock*sizeof(float)); /**/


	// create a stream
	cudaStream_t stream1;
	checkCudaErrors(cudaStreamCreate(&stream1));
	cudaStream_t stream2;
	checkCudaErrors(cudaStreamCreate(&stream2));

	// allocate memory for storing the current best clusters in one-hot format.
	bool* bestDims_d;
	checkCudaErrors(cudaMalloc((void **) &bestDims_d, sizes.size_of_bestDims));
	
	// allocating memory for random samples
	unsigned int* centroids_d;
	unsigned int* samples_d;	
	checkCudaErrors(cudaMalloc((void **) &samples_d, sizes.size_of_samples));
	checkCudaErrors(cudaMalloc((void **) &centroids_d, sizes.size_of_centroids));

	
	curandState* randomStates_d;
	checkCudaErrors(cudaMalloc((void**)&randomStates_d, sizes.size_of_randomStates));
	
	int randomSeed = this->randInt(0,100000, 1).at(0);
	generateRandomStatesArray(stream1, randomStates_d,numberOfRandomStates,false, randomSeed);

	float* data_d;
	checkCudaErrors(cudaMalloc((void **) &data_d, sizes.size_of_data));
	
	checkCudaErrors(cudaMemcpyAsync(data_d, data_h, sizes.size_of_data, cudaMemcpyHostToDevice, stream2));
	
	checkCudaErrors(cudaFreeHost(data_h)); // the data is not used on the host side any more


	// allocating memory for findDim
	bool* findDim_d;
	unsigned int* findDim_count_d;
	checkCudaErrors(cudaMalloc((void **) &findDim_d, sizes.size_of_findDim));
	checkCudaErrors(cudaMalloc((void **) &findDim_count_d, sizes.size_of_findDim_count));

	// allocating memory for pointsContained
	bool* pointsContained_d;
	unsigned int* pointsContained_count_d;
	checkCudaErrors(cudaMalloc((void **) &pointsContained_d, sizes.size_of_pointsContained));
	checkCudaErrors(cudaMalloc((void **) &pointsContained_count_d, sizes.size_of_pointsContained_count));

	//allocating memory for score
	float* score_d;
	checkCudaErrors(cudaMalloc((void **) &score_d, sizes.size_of_score));

	//allocating memory for index
	unsigned int* index_d;
	checkCudaErrors(cudaMalloc((void **) &index_d, sizes.size_of_index));

	double number_of_centroids_sample;
	double number_of_centroids_max;
	
	for(int i = 0; i < k; i++){ // for each of the clusters
		number_of_centroids_max = MemSolver::computeCentroidSizeForAllocation(sizes, dim, number_of_points,
																   number_of_centroids, m, sample_size);
		std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> res;
		float maxScore = 0;
		float centroids_used = 0; 
		int j = 0; 
		

		while(centroids_used < number_of_centroids){
			// fixes the last offset
			if(centroids_used-ceilf(centroids_used)!=0 &&
			   ceilf(centroids_used)-centroids_used < number_of_centroids_max){
			   
				std::cout << "case 1: ";
				number_of_centroids_sample = ceilf(centroids_used)-centroids_used;
			}else if((centroids_used+number_of_centroids_max) > number_of_centroids){
				std::cout << "case 2: ";
				number_of_centroids_sample = number_of_centroids-centroids_used;
			}else{
				if(number_of_centroids_max < 2){
					std::cout << "case 3a: ";
					number_of_centroids_sample = number_of_centroids_max;					
				}else{
					std::cout << "case 3b: ";
					number_of_centroids_sample = floorf(number_of_centroids_max);
				}

			}
			std::cout << j << " centroid_size: " << number_of_centroids_sample << std::endl;
			centroids_used += number_of_centroids_sample;

			arr_sizes = MemSolver::computeArraySizes(number_of_centroids_sample, number_of_points, m, sample_size);
			


			if(dimBlock > maxBlock) dimBlock = maxBlock;
			unsigned int dimGrid = ceil((float)arr_sizes.number_of_samples/(float)dimBlock); /**/
			unsigned int sharedMemSize = (dimBlock*sizeof(unsigned int) + dimBlock*sizeof(float)); /**/

			
			// generate random indices for samples
			generateRandomIntArrayDevice(stream1, samples_d, randomStates_d , numberOfRandomStates,
										 arr_sizes.number_of_values_in_samples, number_of_points-1, 0);

			if(j == 0 || j%((unsigned int)ceilf(1/number_of_centroids_sample))==0){
				generateRandomIntArrayDevice(stream1, centroids_d, randomStates_d , numberOfRandomStates,
											 arr_sizes.number_of_centroids, number_of_points-1 , 0);
			}

			std::cout << sizes.size_of_data << ", " << number_of_points*dim*sizeof(float) << std::endl;
			assert(sizes.size_of_data >= number_of_points*dim*sizeof(float));
			assert(arr_sizes.number_of_values_in_samples >= arr_sizes.number_of_centroids_f);

			// make sure the data is here before find dimension is called

			
			cudaStreamSynchronize(stream2);
		
			assert(arr_sizes.number_of_samples*sizeof(unsigned int) <= sizes.size_of_findDim_count);
			// Find dimensions
			findDimmensionsKernel(dimGrid, dimBlock, stream1, samples_d, centroids_d, data_d,  findDim_d,
								  findDim_count_d, dim, arr_sizes.number_of_samples, sample_size, arr_sizes.number_of_centroids,
								  m, width, number_of_points);

			
			std::cout << arr_sizes.number_of_values_in_pointsContained*sizeof(bool) << ", " << sizes.size_of_pointsContained << std::endl;
			assert(arr_sizes.number_of_values_in_pointsContained*sizeof(bool) <= sizes.size_of_pointsContained);
			// Find points contained
			pointsContainedKernel(dimGrid, dimBlock, stream1, data_d, centroids_d, findDim_d,
								  pointsContained_d, pointsContained_count_d,
								  width, dim, number_of_points, arr_sizes.number_of_samples, m);
		
			// Calculate scores
			scoreKernel(dimGrid, dimBlock, stream1,
						pointsContained_count_d,
						findDim_count_d, score_d,
						arr_sizes.number_of_samples,
						alpha, beta, number_of_points);
	
			// Create indices for the scores
			createIndicesKernel(dimGrid, dimBlock, stream2, index_d, arr_sizes.number_of_samples);

			// make sure the indexes are created before argmax is called
			cudaStreamSynchronize(stream2);
		
			// Find highest scoring
			argMaxKernel(dimGrid, dimBlock, sharedMemSize, stream1, score_d, index_d, arr_sizes.number_of_samples);
		
			// Output Points in that cluster, along with the dimensions and centroid
			unsigned int* best_index_h = (unsigned int*) malloc(sizeof(unsigned int));
			checkCudaErrors(cudaMemcpyAsync(best_index_h, index_d, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream1));

			
			// Output Points in that cluster, along with the dimensions and centroid
			float* best_score_h = (float*) malloc(sizeof(float));
			checkCudaErrors(cudaMemcpyAsync(best_score_h, score_d, sizeof(float), cudaMemcpyDeviceToHost, stream1));

			cudaStreamSynchronize(stream1);
			if(maxScore < best_score_h[0]){
				maxScore = best_score_h[0];
		
				unsigned int* cluster_size_h = (unsigned int*) malloc(sizeof(unsigned int));
				checkCudaErrors(cudaMemcpyAsync(cluster_size_h, pointsContained_count_d+(best_index_h[0]),
												sizeof(unsigned int), cudaMemcpyDeviceToHost, stream1));

				cudaStreamSynchronize(stream1);

				unsigned int size_of_cluster = cluster_size_h[0]*dim*sizeof(float);
				float* cluster_d;
				checkCudaErrors(cudaMalloc((void **) &cluster_d, size_of_cluster));

				checkCudaErrors(cudaMemcpyAsync(bestDims_d, pointsContained_d+(best_index_h[0]*number_of_points), number_of_points, cudaMemcpyDeviceToDevice, stream1));
				std::cout << "copied to be deleted" << std::endl;
				

				// create output cluster
				notDevice(dimGrid, dimBlock, stream1, pointsContained_d+(best_index_h[0]*number_of_points), number_of_points);
		
				deleteFromArray(stream1, cluster_d, pointsContained_d+(best_index_h[0]*number_of_points), data_d, number_of_points, dim);

			   

				auto cluster = new std::vector<std::vector<float>*>;
				auto dims = new std::vector<bool>;
				if(cluster_size_h[0] < this->alpha*number_of_points){
					// cluster is too small, return empty clusters, and try again....
					// this might cause it to loop for the rest of the iterations and do nothing, maybe break here? 
				}else{

					// set number of points to the new data size.
					

					float* cluster_h;
					checkCudaErrors(cudaMallocHost((void**) &cluster_h, size_of_cluster));
					checkCudaErrors(cudaMemcpyAsync(cluster_h, cluster_d, size_of_cluster, cudaMemcpyDeviceToHost, stream1));

					bool* output_dims_h;
					checkCudaErrors(cudaMallocHost((void**) &output_dims_h, dim*sizeof(bool)));
					checkCudaErrors(cudaMemcpyAsync(output_dims_h, findDim_d+(best_index_h[0]*dim),
													dim, cudaMemcpyDeviceToHost, stream1));

					cudaStreamSynchronize(stream1);

					for(int a = 0; a < cluster_size_h[0]; a++){ 
						auto point = new std::vector<float>;
						for(int b = 0; b < dim; b++){
							point->push_back(cluster_h[a*dim+b]);
						}
						cluster->push_back(point);
					}

					for(int a = 0; a < dim; a++){
						dims->push_back(output_dims_h[a]);
					}


					checkCudaErrors(cudaFreeHost(cluster_h));
					checkCudaErrors(cudaFreeHost(output_dims_h));
					checkCudaErrors(cudaFree(cluster_d));
				} 
				// build result
				res = std::make_pair(cluster, dims);
				// clean up
				free(best_index_h);
				free(cluster_size_h);
			}
			// TODO delete old vectors when new cluster found
			j++;
		
		}
		deleteFromArray(stream2, data_d, bestDims_d, data_d, number_of_points, dim);
		number_of_points -= res.first->size();
		result.push_back(res);
		//break if there is no more points
		if(number_of_points <= 0){
			break;
		}

	}

	checkCudaErrors(cudaStreamDestroy(stream1));
	checkCudaErrors(cudaStreamDestroy(stream2));
	
	// clean up on device
	checkCudaErrors(cudaFree(samples_d));
	checkCudaErrors(cudaFree(centroids_d));
	checkCudaErrors(cudaFree(randomStates_d));
	checkCudaErrors(cudaFree(data_d));
	checkCudaErrors(cudaFree(findDim_d));
	checkCudaErrors(cudaFree(findDim_count_d));
	checkCudaErrors(cudaFree(pointsContained_d));
	checkCudaErrors(cudaFree(pointsContained_count_d));
	checkCudaErrors(cudaFree(score_d));
	checkCudaErrors(cudaFree(index_d));
	

	return result;
};


