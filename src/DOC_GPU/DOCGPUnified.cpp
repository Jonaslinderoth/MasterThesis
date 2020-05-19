/*
 * DOCGPU.cpp
 *
 *  Created on: Feb 14, 2020
 *      Author: mikkel
 */

#include "DOCGPUnified.h"
#include "DOCGPU_Kernels.h"
#include "pointsContainedDevice.h"
#include "ArgMax.h"
#include <assert.h>
#include "../randomCudaScripts/DeleteFromArray.h"
#include "MemSolverUnified.h"
#include "../randomCudaScripts/arrayEqual.h"
#include <algorithm>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

# define CUDA_CALL ( x) do { if (( x) != cudaSuccess ) { \
	printf (" Error at % s :% d\ n" , __FILE__ , __LINE__ ) ;\
return EXIT_FAILURE ;}} while (0)
# define CURAND_CALL ( x) do { if (( x) != CURAND_STATUS_SUCCESS ) { \
printf (" Error at % s :% d\ n" , __FILE__ , __LINE__ ) ;\
return EXIT_FAILURE ;}} while (0)


DOCGPUnified::DOCGPUnified(std::vector<std::vector<float>*>* input, float alpha, float beta, float width) {
	this->data = input;
	this->alpha = alpha;
	this->width = width;
	this->beta = beta;
	this->size = data->size();
	this->dim = data->at(0)->size();
}


DOCGPUnified::~DOCGPUnified() {
	// TODO Auto-generated destructor stub
}


/**
 * Used for loading the data from the data reader into main memory.
 */
std::vector<std::vector<float>*>* DOCGPUnified::initDataReader(DataReader* dr){
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
float* DOCGPUnified::transformData(){
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
std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> DOCGPUnified::findCluster(){
	auto result = findKClusters(1).at(0);
	return result;
};


std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*>> DOCGPUnified::findKClusters(int k){
	auto result = std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*>>();
	if(k == 0){
		return result;
	}
	float* data_d = this->transformData();
	uint dim = this->data->at(0)->size();
	auto alpha = this->alpha;
	auto beta = this->beta;
	auto width = this->width;

	// calculating algorithm parameters
	unsigned int r = log2(2*dim)/log2(1/(2*beta));
	if(r == 0) r = 1;
	unsigned int m = pow((2/alpha),r) * log(4);
	if(this->m != 0){
		if(m > this->m){
			m = this->m;
		}
	}
	unsigned int number_of_points = this->data->size();
	size_t numberOfRandomStates = 1024*10;

	// Calculating the sizes of the random samples
	unsigned int number_of_centroids = 2.0/alpha;
	unsigned int sample_size = r;

	// Calculate the amount of free memory
	size_t memoryCap;
	size_t totalMem;
	cudaMemGetInfo(&memoryCap, &totalMem);

	// Calculate the sizes for array allocation, and the number of elements in each
	Memory_sizes sizes = MemSolverUnified::computeForAllocations(dim, number_of_points, number_of_centroids, m, sample_size, k, memoryCap);
	Array_sizes arr_sizes = MemSolverUnified::computeArraySizes(sizes.first_number_of_centroids, number_of_points, m, sample_size);


	//calculating dimensions for threads
	int device = -1;
	cudaGetDevice(&device);
	int smemSize, maxBlock;
	cudaDeviceGetAttribute(&smemSize, 
						   cudaDevAttrMaxSharedMemoryPerBlock, device);
	cudaDeviceGetAttribute(&maxBlock, 
						   cudaDevAttrMaxThreadsPerBlock, device);

	// caluclate the maximum thread size based on shared mem requirements and maximum threads
	unsigned int dimBlock = smemSize/(sizeof(int)+sizeof(float));
	if(dimBlock > maxBlock) dimBlock = maxBlock;
	unsigned int dimGrid = ceil((float)arr_sizes.number_of_samples/(float)dimBlock); /**/
	unsigned int sharedMemSize = (dimBlock*sizeof(unsigned int) + dimBlock*sizeof(float)); /**/


	// create streams
	cudaStream_t stream1;
	checkCudaErrors(cudaStreamCreate(&stream1));
	cudaStream_t stream2;
	checkCudaErrors(cudaStreamCreate(&stream2));

	// allocate memory for storing the current best clusters in one-hot format.
	bool* bestDims_d;
	checkCudaErrors(cudaMallocManaged((void **) &bestDims_d, sizes.size_of_bestDims));

	
	// allocating memory for random samples
	unsigned int* centroids_d;
	unsigned int* samples_d;
	curandState* randomStates_d;
	checkCudaErrors(cudaMallocManaged((void **) &samples_d, sizes.size_of_samples));
	checkCudaErrors(cudaMallocManaged((void **) &centroids_d, sizes.size_of_centroids));
	checkCudaErrors(cudaMallocManaged((void **) &randomStates_d, sizes.size_of_randomStates));

	// Create the random states using the CPU random generator to create a seed,
	// if the CPU random generator is seeded then the GPU is also
	assert(numberOfRandomStates*sizeof(curandState) == sizes.size_of_randomStates);
  	int randomSeed = this->randInt(0,100000, 1).at(0);
	generateRandomStatesArray(stream1, randomStates_d,numberOfRandomStates,false, randomSeed);

	// allocate space for the Data, transfer it and release it from Host
	float* outputCluster_d;
	checkCudaErrors(cudaMallocManaged((void **) &outputCluster_d, sizes.size_of_data));

	float* dataPointers_d[2];
	dataPointers_d[0] = data_d;
	dataPointers_d[1] = outputCluster_d;
	
	// allocating memory for findDim
	bool* findDim_d;
	unsigned int* findDim_count_d;
	checkCudaErrors(cudaMallocManaged((void **) &findDim_d, sizes.size_of_findDim));
	checkCudaErrors(cudaMallocManaged((void **) &findDim_count_d, sizes.size_of_findDim_count));

	// allocating memory for pointsContained
	bool* pointsContained_d;
	unsigned int* pointsContained_count_d;
	checkCudaErrors(cudaMallocManaged((void **) &pointsContained_d, sizes.size_of_pointsContained));
	checkCudaErrors(cudaMallocManaged((void **) &pointsContained_count_d, sizes.size_of_pointsContained_count));

	//allocating memory for score
	float* score_d;
	checkCudaErrors(cudaMallocManaged((void **) &score_d, sizes.size_of_score));

	
	//allocating memory for index
	unsigned int* index_d;
	checkCudaErrors(cudaMallocManaged((void **) &index_d, sizes.size_of_index));

	// Declaring variables used in the loop
	double number_of_centroids_sample;
	double number_of_centroids_max;
	
	for(int i = 0; i < k; i++){ // for each of the clusters
		number_of_centroids_max = MemSolverUnified::computeCentroidSizeForAllocation(sizes, dim, number_of_points,
																   number_of_centroids, m, sample_size);
		std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> res;
		float maxScore = 0; // used to store the current best score
		float centroids_used = 0; // used to count the amount of centroids that have been processed
		int j = 0; // counting the total number of iterations for this cluster

		while(centroids_used < number_of_centroids){
			// fixes the last offset
			if(centroids_used-ceilf(centroids_used)!=0 &&
			   ceilf(centroids_used)-centroids_used < number_of_centroids_max){
				//std::cout << "case 1: ";
				number_of_centroids_sample = ceilf(centroids_used)-centroids_used;
			}else if((centroids_used+number_of_centroids_max) > number_of_centroids){
				//std::cout << "case 2: ";
				number_of_centroids_sample = number_of_centroids-centroids_used;
			}else{
				if(number_of_centroids_max < 2){
					//std::cout << "case 3a: ";
					number_of_centroids_sample = number_of_centroids_max;
				}else{
					//std::cout << "case 3b: ";
					number_of_centroids_sample = floorf(number_of_centroids_max);
				}
			}
			centroids_used += number_of_centroids_sample;

			assert(sizes.size_of_data >= number_of_points*dim*sizeof(float));
			assert(arr_sizes.number_of_values_in_samples >= arr_sizes.number_of_centroids_f);
			assert(arr_sizes.number_of_samples*sizeof(unsigned int) <= sizes.size_of_findDim_count);
			assert(arr_sizes.number_of_values_in_pointsContained*sizeof(bool) <= sizes.size_of_pointsContained);

			// compute the number of values given the new number of centroids
			arr_sizes = MemSolverUnified::computeArraySizes(number_of_centroids_sample, number_of_points, m, sample_size);

			if(dimBlock > maxBlock) dimBlock = maxBlock;
			unsigned int dimGrid = ceil((float)arr_sizes.number_of_samples/(float)dimBlock);
			unsigned int sharedMemSize = (dimBlock*sizeof(unsigned int) + dimBlock*sizeof(float));


			// generate random indices for samples
			checkCudaErrors(cudaMemPrefetchAsync(samples_d, arr_sizes.number_of_values_in_samples*sizeof(unsigned int), device, stream1));
			generateRandomIntArrayDevice(stream1, samples_d, randomStates_d , numberOfRandomStates,
										 arr_sizes.number_of_values_in_samples, number_of_points-1, 0);

			// only regenerate when new centroids are needed
			if(j == 0 || j%((unsigned int)ceilf(1/number_of_centroids_sample))==0){
				checkCudaErrors(cudaMemPrefetchAsync(centroids_d, arr_sizes.number_of_centroids*sizeof(unsigned int), device, stream1));				
				generateRandomIntArrayDevice(stream1, centroids_d, randomStates_d , numberOfRandomStates,
											 arr_sizes.number_of_centroids, number_of_points-1 , 0);
			}
			
			checkCudaErrors(cudaMemPrefetchAsync(data_d, number_of_points*dim*sizeof(float), device, stream2));
			checkCudaErrors(cudaMemPrefetchAsync(centroids_d, arr_sizes.number_of_centroids*sizeof(unsigned int), device, stream2));
			checkCudaErrors(cudaMemPrefetchAsync(findDim_d, arr_sizes.number_of_samples*dim*sizeof(bool), device, stream2));
			checkCudaErrors(cudaMemPrefetchAsync(findDim_count_d, arr_sizes.number_of_samples*sizeof(unsigned int), device, stream2));

			cudaStreamSynchronize(stream2); // Synchronize stream 2 to make sure that the data has arrived


			// Find dimensions
			findDimmensionsKernel(dimGrid, dimBlock, stream1, samples_d, centroids_d, data_d,  findDim_d,
								  findDim_count_d, dim, arr_sizes.number_of_samples, sample_size, arr_sizes.number_of_centroids,
								  m, width, number_of_points, this->findDimKernelVersion);

			checkCudaErrors(cudaMemPrefetchAsync(data_d, number_of_points*dim*sizeof(float), device, stream2));
			checkCudaErrors(cudaMemPrefetchAsync(centroids_d, arr_sizes.number_of_centroids*sizeof(unsigned int), device, stream2));
			checkCudaErrors(cudaMemPrefetchAsync(findDim_d, arr_sizes.number_of_samples*dim*sizeof(bool), device, stream2));
			checkCudaErrors(cudaMemPrefetchAsync(pointsContained_d, arr_sizes.number_of_samples*number_of_points*sizeof(bool), device, stream2));
			checkCudaErrors(cudaMemPrefetchAsync(pointsContained_count_d, arr_sizes.number_of_samples*sizeof(unsigned int), device, stream2));
			cudaStreamSynchronize(stream2);

			// Find points contained
			pointsContainedKernelNaive(dimGrid, dimBlock, stream1, data_d, centroids_d, findDim_d,
								  pointsContained_d, pointsContained_count_d,
								  width, dim, number_of_points, arr_sizes.number_of_samples, m);


			// Calculate scores
			checkCudaErrors(cudaMemPrefetchAsync(pointsContained_count_d, arr_sizes.number_of_samples*sizeof(unsigned int), device, stream2));
			checkCudaErrors(cudaMemPrefetchAsync(findDim_count_d, arr_sizes.number_of_samples*sizeof(unsigned int), device, stream2));
			checkCudaErrors(cudaMemPrefetchAsync(score_d, arr_sizes.number_of_samples*sizeof(float), device, stream2));
			cudaStreamSynchronize(stream2);
			scoreKernel(dimGrid, dimBlock, stream1,
						pointsContained_count_d,
						findDim_count_d, score_d,
						arr_sizes.number_of_samples,
						alpha, beta, number_of_points);
			// Create indices for the scores
			checkCudaErrors(cudaMemPrefetchAsync(index_d, arr_sizes.number_of_samples*sizeof(unsigned int), device, stream2));
			createIndicesKernel(dimGrid, dimBlock, stream2, index_d, arr_sizes.number_of_samples);

			// make sure the indexes are created before argmax is called
			cudaStreamSynchronize(stream2);
		
			// Find highest scoring
			checkCudaErrors(cudaMemPrefetchAsync(score_d, arr_sizes.number_of_samples*sizeof(float), device, stream2));
			argMaxKernel(dimGrid, dimBlock, sharedMemSize, stream1, score_d, index_d, arr_sizes.number_of_samples);

			// Output Points in that cluster, along with the dimensions and centroid
			checkCudaErrors(cudaMemPrefetchAsync((score_d), sizeof(float), cudaCpuDeviceId, stream1));
			cudaStreamSynchronize(stream1);
			float bestScore = score_d[0];
			if(maxScore < bestScore){ // if the new score is better than the current;
				maxScore = bestScore;

				// Allocate space for the size of the cluster
				checkCudaErrors(cudaMemPrefetchAsync((index_d), sizeof(unsigned int), cudaCpuDeviceId, stream1));
				cudaStreamSynchronize(stream1);
				unsigned int bestIndex = index_d[0];

				checkCudaErrors(cudaMemPrefetchAsync((pointsContained_count_d+bestIndex), sizeof(unsigned int), cudaCpuDeviceId, stream1));
				cudaStreamSynchronize(stream1);
				unsigned int clusterSize = (pointsContained_count_d+bestIndex)[0];
				// make sure that the size have arrived on the host side
				cudaStreamSynchronize(stream1);

				// allocate space for the cluster on host side, and copy it.
				unsigned int size_of_cluster = clusterSize*dim*sizeof(float);

				auto cluster = new std::vector<std::vector<float>*>;
				auto dims = new std::vector<bool>;
				if(clusterSize < this->alpha*number_of_points){
					// cluster is too small, return empty clusters, and try again....
					// this might cause it to loop for the rest of the iterations and do nothing, maybe break here?
				}else{
					checkCudaErrors(cudaMemcpyAsync(bestDims_d, pointsContained_d+(bestIndex*number_of_points),
													number_of_points, cudaMemcpyDeviceToDevice, stream1));

					// create output cluster
					checkCudaErrors(cudaMemPrefetchAsync(pointsContained_d+(bestIndex*number_of_points), sizeof(bool)*(number_of_points+1), device, stream1));			   
					notDevice(dimGrid, dimBlock, stream1, pointsContained_d+(bestIndex*number_of_points), number_of_points);
					
					assert(sizes.size_of_data >= number_of_points*dim*sizeof(float));
					checkCudaErrors(cudaMemPrefetchAsync(outputCluster_d, sizeof(float)*(clusterSize*dim), device, stream1));
					deleteFromArray_managed(stream1, outputCluster_d, pointsContained_d+(bestIndex*number_of_points),
									data_d, number_of_points, dim);
					
					// syncronise before deleting
					checkCudaErrors(cudaMemPrefetchAsync(outputCluster_d, sizeof(unsigned int)*(dim*clusterSize), cudaCpuDeviceId, stream1));
					checkCudaErrors(cudaMemPrefetchAsync(findDim_d+(bestIndex*dim), sizeof(unsigned int)*dim, cudaCpuDeviceId, stream2));
					bool* output_dims_h = findDim_d+(bestIndex*dim);
					cudaStreamSynchronize(stream1);

					// Create the output in vector format
					// IDEA: create a seperate thread for this, because it just needs to be done before return.
					for(int a = 0; a < clusterSize; a++){
						auto point = new std::vector<float>;
						for(int b = 0; b < dim; b++){
							point->push_back(outputCluster_d[a*dim+b]);
						}
						cluster->push_back(point);
					}

					cudaStreamSynchronize(stream2);
					for(int a = 0; a < dim; a++){
						dims->push_back(output_dims_h[a]);
					}

				}
				// build result & and delete old
				delete res.first;
				delete res.second;
				res = std::make_pair(cluster, dims);
				// clean up
			}
			j++;

		}

		float* newData_d = outputCluster_d+res.first->size()*dim;
		// Delete the extracted cluster from the data
		size_t a = res.first->size();
		assert(sizes.size_of_data >= number_of_points*dim*sizeof(float));
		checkCudaErrors(cudaMemPrefetchAsync(data_d, number_of_points*dim*sizeof(float), device, stream2));
		if(a < number_of_points){
			checkCudaErrors(cudaMemPrefetchAsync(newData_d, (number_of_points - a)*dim*sizeof(float), device, stream2));
		}
		checkCudaErrors(cudaMemPrefetchAsync(bestDims_d, number_of_points*sizeof(bool), device, stream2));
		deleteFromArray_managed(stream2, newData_d, bestDims_d, data_d, number_of_points, dim);

		number_of_points -= a;
		result.push_back(res);

		data_d = newData_d;
		assert(outputCluster_d != dataPointers_d[(i)%2]);
		outputCluster_d = dataPointers_d[(i)%2];
		
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
	checkCudaErrors(cudaFree(dataPointers_d[0]));
	checkCudaErrors(cudaFree(dataPointers_d[1]));
	checkCudaErrors(cudaFree(findDim_d));
	checkCudaErrors(cudaFree(findDim_count_d));
	checkCudaErrors(cudaFree(pointsContained_d));
	checkCudaErrors(cudaFree(pointsContained_count_d));
	checkCudaErrors(cudaFree(score_d));
	checkCudaErrors(cudaFree(index_d));
	

	return result;
};

