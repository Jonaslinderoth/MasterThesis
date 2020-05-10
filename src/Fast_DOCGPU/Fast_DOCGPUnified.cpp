#include "Fast_DOCGPUnified.h"
#include "../DOC_GPU/DOCGPU_Kernels.h"
#include "../DOC_GPU/ArgMax.h"
#include <assert.h>
#include "../randomCudaScripts/DeleteFromArray.h"
#include "whatDataInCentroid.h"
#include "MemSolver_Fast_DOCUnified.h"

Fast_DOCGPUnified::Fast_DOCGPUnified(std::vector<std::vector<float>*>* input, float alpha, float beta, float width, unsigned int d0) {
	this->data = input;
	this->alpha = alpha;
	this->width = width;
	this->beta = beta;
	this->size = data->size();
	this->dim = data->at(0)->size();
	if(d0 == 0){
		if(input->size() > 0){
			this->d0 = input->at(0)->size();		
		}else{
			this->d0 = 1;
		}

	}else{
		this->d0 = d0;
	}
}


Fast_DOCGPUnified::~Fast_DOCGPUnified() {
	// TODO Auto-generated destructor stub
}


/**
 * Used for loading the data from the data reader into main memory.
 */
std::vector<std::vector<float>*>* Fast_DOCGPUnified::initDataReader(DataReader* dr){
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
float* Fast_DOCGPUnified::transformData(){
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
std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> Fast_DOCGPUnified::findCluster(){
	auto result = findKClusters(1).at(0);
	return result;
};

std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*>> Fast_DOCGPUnified::findKClusters(int k){
	auto result = std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*>>();
	float* data_d = this->transformData();
	uint dim = this->data->at(0)->size();
	auto alpha = this->alpha;
	auto beta = this->beta;
	auto width = this->width;

	// calculating algorithm parameters
	unsigned int r = log2(2*dim)/log2(1/(2*this->beta));
	if(r == 0) r = 1;
	float MAXITER = fmin(pow(dim,2), pow(10,6));
	float m = fmin(MAXITER, pow((2/this->alpha),r) * log(4));

	
	unsigned int number_of_points = this->data->size();
	size_t numberOfRandomStates = 1024*2;

	// Calculating the sizes of the random samples
	unsigned int number_of_centroids = 2.0/alpha;
	unsigned int sample_size = r;

	// Calculate the amount of free memory
	size_t memoryCap;
	size_t totalMem;
	cudaMemGetInfo(&memoryCap, &totalMem);

	// Calculate the sizes for array allocation, and the number of elements in each
	Memory_sizes sizes = MemSolver_Fast_DOCUnified::computeForAllocations(dim, number_of_points, number_of_centroids, m, sample_size, k, memoryCap);
	Array_sizes arr_sizes = MemSolver_Fast_DOCUnified::computeArraySizes(sizes.first_number_of_centroids, number_of_points, m, sample_size);


	//calculating dimensions for threads
	int device = -1;
	cudaGetDevice(&device);
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
	// std::cout << "sizes.size_of_samples: " << sizes.size_of_samples << std::endl;
	// std::cout << "sizes.size_of_centroids: " << sizes.size_of_centroids << std::endl;
	checkCudaErrors(cudaMallocManaged((void **) &samples_d, sizes.size_of_samples));
	checkCudaErrors(cudaMallocManaged((void **) &centroids_d, sizes.size_of_centroids));
	checkCudaErrors(cudaMallocManaged((void**) &randomStates_d, sizes.size_of_randomStates));

	// Create the random states using the CPU random generator to create a seed,
	// if the CPU random generator is seeded then the GPU is also
	assert(numberOfRandomStates*sizeof(curandState) == sizes.size_of_randomStates);
  	int randomSeed = this->randInt(0,100000, 1).at(0);
	generateRandomStatesArray(stream1, randomStates_d,numberOfRandomStates,false, randomSeed);

	
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
	checkCudaErrors(cudaMallocManaged((void **) &pointsContained_d, sizes.size_of_pointsContained));
	// std::cout << "sizes.size_of_pointsContained: " << sizes.size_of_pointsContained << std::endl;


	//allocating memory for index
	unsigned int* index_d;
	checkCudaErrors(cudaMallocManaged((void **) &index_d, sizes.size_of_index));


	// allocate for deletion
	size_t sizeOfPrefixSum = (number_of_points+1)*sizeof(unsigned int); // +1 because of the prefixSum
	unsigned int* prefixSum_d;
	checkCudaErrors(cudaMallocManaged((void**) &prefixSum_d, sizeOfPrefixSum));
	
	// Declaring variables used in the loop
	double number_of_centroids_sample;
	double number_of_centroids_max;
	for(int i = 0; i < k; i++){ // for each of the clusters
		// std::cout << "finding new cluster" << std::endl;
		number_of_centroids_max = MemSolver_Fast_DOCUnified::computeCentroidSizeForAllocation(sizes, dim, number_of_points,
																   number_of_centroids, m, sample_size);
		std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> res;
		float maxScore = -10; // used to store the current best score
		float centroids_used = 0; // used to count the amount of centroids that have been processed
		int j = 0; // counting the total number of iterations for this cluster
		while(centroids_used < number_of_centroids){
			// fixes the last offset
			if(centroids_used-ceilf(centroids_used)!=0 &&
			   ceilf(centroids_used)-centroids_used < number_of_centroids_max){
				// std::cout << "case 1: ";
				number_of_centroids_sample = ceilf(centroids_used)-centroids_used;
			}else if((centroids_used+number_of_centroids_max) > number_of_centroids){
				// std::cout << "case 2: ";
				number_of_centroids_sample = number_of_centroids-centroids_used;
			}else{
				if(number_of_centroids_max < 2){
					// std::cout << "case 3a: ";
					number_of_centroids_sample = number_of_centroids_max;
				}else{
					// std::cout << "case 3b: ";
					number_of_centroids_sample = floorf(number_of_centroids_max);
				}
			}
			//std::cout << j << " centroid_size: " << number_of_centroids_sample << std::endl;
			centroids_used += number_of_centroids_sample;

			assert(sizes.size_of_data >= number_of_points*dim*sizeof(float));
			assert(arr_sizes.number_of_values_in_samples >= arr_sizes.number_of_centroids_f);
			assert(arr_sizes.number_of_samples*sizeof(unsigned int) <= sizes.size_of_findDim_count);
			assert(arr_sizes.number_of_values_in_pointsContained*sizeof(bool) <= sizes.size_of_pointsContained);

			// compute the number of values given the new number of centroids
			arr_sizes = MemSolver_Fast_DOCUnified::computeArraySizes(number_of_centroids_sample, number_of_points, m, sample_size);

			if(dimBlock > maxBlock) dimBlock = maxBlock;
			unsigned int dimGrid = ceil((float)arr_sizes.number_of_samples/(float)dimBlock);
			unsigned int sharedMemSize = (dimBlock*sizeof(unsigned int) + dimBlock*sizeof(float));


			// generate random indices for samples
			checkCudaErrors(cudaMemPrefetchAsync(samples_d, arr_sizes.number_of_samples*sizeof(unsigned int), device, stream1));
			checkCudaErrors(cudaMemPrefetchAsync(randomStates_d, sizes.size_of_randomStates, device, stream1));
			generateRandomIntArrayDevice(stream1, samples_d, randomStates_d , numberOfRandomStates,
										 arr_sizes.number_of_values_in_samples, number_of_points-1, 0);

			// only regenerate when new centroids are needed
			if(j == 0 || j%((unsigned int)ceilf(1/number_of_centroids_sample))==0){
				generateRandomIntArrayDevice(stream1, centroids_d, randomStates_d , numberOfRandomStates,
											 arr_sizes.number_of_centroids, number_of_points-1 , 0);
			}

			checkCudaErrors(cudaMemPrefetchAsync(data_d, number_of_points*dim*sizeof(float), device, stream2));
			// std::cout << " arr_sizes.number_of_centroids*sizeof(unsigned int): " << arr_sizes.number_of_centroids*sizeof(unsigned int) << std::endl;
			// std::cout << "number_of_centroids_sample: " << number_of_centroids_sample << std::endl;
			checkCudaErrors(cudaMemPrefetchAsync(centroids_d, arr_sizes.number_of_centroids*sizeof(unsigned int), device, stream2));
			checkCudaErrors(cudaMemPrefetchAsync(findDim_d, arr_sizes.number_of_samples*dim*sizeof(bool), device, stream2));
			checkCudaErrors(cudaMemPrefetchAsync(findDim_count_d, arr_sizes.number_of_samples*sizeof(unsigned int), device, stream2));
				
			cudaStreamSynchronize(stream2); // Synchronize stream 2 to make sure that the data has arrived


			// Find dimensions
			findDimmensionsKernel(dimGrid, dimBlock, stream1, samples_d, centroids_d, data_d,  findDim_d,
								  findDim_count_d, dim, arr_sizes.number_of_samples, sample_size, arr_sizes.number_of_centroids,
								  m, width, number_of_points);

			// Create indices for the scores
			checkCudaErrors(cudaMemPrefetchAsync(index_d, arr_sizes.number_of_samples*sizeof(unsigned int), device, stream2));
			createIndicesKernel(dimGrid, dimBlock, stream2, index_d, arr_sizes.number_of_samples);

			// make sure the indexes are created before argmax is called
			checkCudaErrors(cudaMemPrefetchAsync(findDim_count_d, arr_sizes.number_of_samples*sizeof(unsigned int), device, stream1));
			cudaStreamSynchronize(stream2);

			
			// Find highest scoring
			argMaxKernelWidthUpperBound(dimGrid, dimBlock, sharedMemSize, stream1, findDim_count_d, index_d, arr_sizes.number_of_samples, this->d0);

			
			// Output Points in that cluster, along with the dimensions and centroid
			// unsigned int* best_index_h = (unsigned int*) malloc(sizeof(unsigned int));
			// checkCudaErrors(cudaMemcpyAsync(best_index_h, index_d, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream1));

			// // Output Points in that cluster, along with the dimensions and centroid
			// unsigned int* best_score_h = (unsigned int*) malloc(sizeof(unsigned int));
			// checkCudaErrors(cudaMemcpyAsync(best_score_h, findDim_count_d, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream1));
			// Synchronize to make sure that the value have arrived to the host side
			cudaStreamSynchronize(stream1);
			if(maxScore < findDim_count_d[0]){ // if the new score is better than the current;
				maxScore = findDim_count_d[0];
				// Find points contained
				
				/*
				pointsContainedKernelNaive(dimGrid, dimBlock, stream1, data_d,
										   centroids_d+(((size_t)(best_index_h[0]/m))),
										   findDim_d+(best_index_h[0]*dim),
										   pointsContained_d, pointsContained_count_d,
										   width, dim, number_of_points, 1, 1);
				*/
				checkCudaErrors(cudaMemPrefetchAsync(pointsContained_d, sizeof(bool)*number_of_points, device, stream1));
				checkCudaErrors(cudaMemPrefetchAsync(centroids_d+(((size_t)(index_d[0]/m))), sizeof(unsigned int), device, stream1));
				checkCudaErrors(cudaMemPrefetchAsync(findDim_d+(index_d[0]*dim), sizeof(bool)*dim, device, stream1));
				checkCudaErrors(cudaMemPrefetchAsync(data_d, sizeof(float)*number_of_points*dim, device, stream1));
				whatDataIsInCentroid(dimGrid,
									 dimBlock,
									 stream1,
									 pointsContained_d,
									 data_d,
									 centroids_d+(((size_t)(index_d[0]/m))),
									 findDim_d+(index_d[0]*dim),
									 width,
									 dim,
									 number_of_points);

				//prefixsum....
												 

				// Allocate space for the size of the cluster
				checkCudaErrors(cudaMemPrefetchAsync(prefixSum_d, sizeof(unsigned int)*number_of_points, device, stream1));
				// std::cout << "sizeof(bool)*(number_of_points+1): " << sizeof(bool)*(number_of_points+1) << std::endl;
				checkCudaErrors(cudaMemPrefetchAsync(pointsContained_d, sizeof(bool)*(number_of_points+1), device, stream1));
				sum_scan_blelloch_managed(stream1,stream1, prefixSum_d,pointsContained_d,(number_of_points+1), true);
				// unsigned int* cluster_size_h = (unsigned int*) malloc(sizeof(unsigned int));
				// checkCudaErrors(cudaMemcpyAsync(cluster_size_h, prefixSum_d+number_of_points,
				// 								sizeof(unsigned int), cudaMemcpyDeviceToHost, stream1));

				// make sure that the size have arrived on the host side
				checkCudaErrors(cudaMemPrefetchAsync(prefixSum_d+number_of_points, sizeof(unsigned int), cudaCpuDeviceId, stream1));
				cudaStreamSynchronize(stream1);
				unsigned int cluster_size = number_of_points-prefixSum_d[number_of_points];
				// allocate space for the cluster on host side, and copy it.
				unsigned int size_of_cluster = cluster_size*dim*sizeof(float);

				
				
				
				auto cluster = new std::vector<std::vector<float>*>;
				auto dims = new std::vector<bool>;
				if(cluster_size < this->alpha*number_of_points){
					// cluster is too small, return empty clusters, and try again....
					// this might cause it to loop for the rest of the iterations and do nothing, maybe break here?
				}else{
					delete res.first;
					delete res.second;
					//float* cluster_d
					checkCudaErrors(cudaMemPrefetchAsync(pointsContained_d, sizeof(bool)*(number_of_points+1), device, stream1));
					checkCudaErrors(cudaMemPrefetchAsync(bestDims_d, sizeof(bool)*(number_of_points+1), device, stream1));					
					checkCudaErrors(cudaMemcpyAsync(bestDims_d, pointsContained_d,
													number_of_points*sizeof(bool), cudaMemcpyDeviceToDevice, stream1));



					// create output cluster
					//notDevice(dimGrid, dimBlock, stream1, pointsContained_d, number_of_points);
					cudaStreamSynchronize(stream1);
					assert(sizes.size_of_data >= number_of_points*dim*sizeof(float));

					checkCudaErrors(cudaMemPrefetchAsync(outputCluster_d, sizeof(float)*cluster_size*dim, device, stream1));
					checkCudaErrors(cudaMemPrefetchAsync(pointsContained_d, sizeof(bool)*(number_of_points+1), device, stream1));
					
					deleteFromArrayWrapper(ceilf((float)(number_of_points*dim)/dimBlock), dimBlock,
										   stream1, data_d, prefixSum_d, 
										   number_of_points, dim, outputCluster_d);

					cudaStreamSynchronize(stream1);
					checkCudaErrors(cudaMemPrefetchAsync(findDim_d+(index_d[0]*dim), sizeof(unsigned int)*dim, cudaCpuDeviceId, stream1));
					
					bool* output_dims_h = findDim_d+(index_d[0]*dim);
					// checkCudaErrors(cudaMallocManagedHost((void**) &output_dims_h, dim*sizeof(bool)));
					// checkCudaErrors(cudaMemcpyAsync(output_dims_h, findDim_d+(best_index_h[0]*dim),
					// 								dim, cudaMemcpyDeviceToHost, stream1));

					// syncronise before deleting
					checkCudaErrors(cudaMemPrefetchAsync(outputCluster_d, sizeof(unsigned int)*(dim*cluster_size), cudaCpuDeviceId, stream1));
					cudaStreamSynchronize(stream1);

					// Create the output in vector format
					// IDEA: create a seperate thread for this, because it just needs to be done before return.
					// std::cout << "outputting" << std::endl;
					for(size_t a = 0; a < cluster_size; a++){
						auto point = new std::vector<float>;
						for(size_t b = 0; b < dim; b++){
							point->push_back(outputCluster_d[a*dim+b]);
						}
						// if(a % 1000 == 0) std::cout << "returning point " << a << std::endl;	
						cluster->push_back(point);
					}
					// std::cout << "cluster outputted" << std::endl;					

					for(size_t a = 0; a < dim; a++){
						dims->push_back(output_dims_h[a]);
					}

				}
				// build result & and delete old
				res = std::make_pair(cluster, dims);
				// clean up
			}
			j++;
		}
		float* newData_d = outputCluster_d+res.first->size()*dim;
		// Delete the extracted cluster from the data
		assert(sizes.size_of_data >= number_of_points*dim*sizeof(float));
		size_t a = res.first->size();

		// std::cout << "(number_of_points - a)*dim*sizeof(float): " << (number_of_points - a)*dim*sizeof(float) << std::endl;
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
	checkCudaErrors(cudaFree(index_d));
	

	return result;	
}
