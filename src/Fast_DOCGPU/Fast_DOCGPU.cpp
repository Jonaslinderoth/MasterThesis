#include "Fast_DOCGPU.h"
#include "../DOC_GPU/DOCGPU_Kernels.h"
#include "../DOC_GPU/ArgMax.h"
#include <assert.h>
#include "../randomCudaScripts/DeleteFromArray.h"
#include "whatDataInCentroid.h"
#include "MemSolver_Fast_DOC.h"

Fast_DOCGPU::Fast_DOCGPU(std::vector<std::vector<float>*>* input, float alpha, float beta, float width, unsigned int d0) {
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


Fast_DOCGPU::~Fast_DOCGPU() {
	// TODO Auto-generated destructor stub
}


/**
 * Used for loading the data from the data reader into main memory.
 */
std::vector<std::vector<float>*>* Fast_DOCGPU::initDataReader(DataReader* dr){
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
float* Fast_DOCGPU::transformData(){
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
std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> Fast_DOCGPU::findCluster(){
	auto result = findKClusters(1).at(0);
	return result;
};


std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*>> Fast_DOCGPU::findKClusters(int k){
	auto result = std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*>>();
	float* data_h = this->transformData();
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
	Memory_sizes sizes = MemSolver_Fast_DOC::computeForAllocations(dim, number_of_points, number_of_centroids, m, sample_size, k, memoryCap);
	Array_sizes arr_sizes = MemSolver_Fast_DOC::computeArraySizes(sizes.first_number_of_centroids, number_of_points, m, sample_size);


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


	// create streams
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
	curandState* randomStates_d;
	checkCudaErrors(cudaMalloc((void **) &samples_d, sizes.size_of_samples));
	checkCudaErrors(cudaMalloc((void **) &centroids_d, sizes.size_of_centroids));
	checkCudaErrors(cudaMalloc((void**) &randomStates_d, sizes.size_of_randomStates));

	// Create the random states using the CPU random generator to create a seed,
	// if the CPU random generator is seeded then the GPU is also
	assert(numberOfRandomStates*sizeof(curandState) == sizes.size_of_randomStates);
  	int randomSeed = this->randInt(0,100000, 1).at(0);
	generateRandomStatesArray(stream1, randomStates_d,numberOfRandomStates,false, randomSeed);

	// allocate space for the Data, transfer it and release it from Host
	float* data_d;
	checkCudaErrors(cudaMalloc((void **) &data_d, sizes.size_of_data));
	checkCudaErrors(cudaMemcpyAsync(data_d, data_h, sizes.size_of_data, cudaMemcpyHostToDevice, stream2));
	cudaStreamSynchronize(stream2);
	
	float* outputCluster_d;
	checkCudaErrors(cudaMalloc((void **) &outputCluster_d, sizes.size_of_data));

	float* dataPointers_d[2];
	dataPointers_d[0] = data_d;
	dataPointers_d[1] = outputCluster_d;
	
	checkCudaErrors(cudaFreeHost(data_h)); // the data is not used on the host side any more


	// allocating memory for findDim
	bool* findDim_d;
	unsigned int* findDim_count_d;
	checkCudaErrors(cudaMalloc((void **) &findDim_d, sizes.size_of_findDim));
	checkCudaErrors(cudaMalloc((void **) &findDim_count_d, sizes.size_of_findDim_count));

	// allocating memory for pointsContained
	bool* pointsContained_d;
	checkCudaErrors(cudaMalloc((void **) &pointsContained_d, sizes.size_of_pointsContained));


	//allocating memory for index
	unsigned int* index_d;
	checkCudaErrors(cudaMalloc((void **) &index_d, sizes.size_of_index));


	// allocate for deletion
	size_t sizeOfPrefixSum = (number_of_points+1)*sizeof(unsigned int); // +1 because of the prefixSum
	unsigned int* prefixSum_d;
	checkCudaErrors(cudaMalloc((void**) &prefixSum_d,   sizeOfPrefixSum));
	
	// Declaring variables used in the loop
	double number_of_centroids_sample;
	double number_of_centroids_max;
	for(int i = 0; i < k; i++){ // for each of the clusters
		number_of_centroids_max = MemSolver_Fast_DOC::computeCentroidSizeForAllocation(sizes, dim, number_of_points,
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
			arr_sizes = MemSolver_Fast_DOC::computeArraySizes(number_of_centroids_sample, number_of_points, m, sample_size);

			if(dimBlock > maxBlock) dimBlock = maxBlock;
			unsigned int dimGrid = ceil((float)arr_sizes.number_of_samples/(float)dimBlock);
			unsigned int sharedMemSize = (dimBlock*sizeof(unsigned int) + dimBlock*sizeof(float));


			// generate random indices for samples
			generateRandomIntArrayDevice(stream1, samples_d, randomStates_d , numberOfRandomStates,
										 arr_sizes.number_of_values_in_samples, number_of_points-1, 0);

			// only regenerate when new centroids are needed
			if(j == 0 || j%((unsigned int)ceilf(1/number_of_centroids_sample))==0){
				generateRandomIntArrayDevice(stream1, centroids_d, randomStates_d , numberOfRandomStates,
											 arr_sizes.number_of_centroids, number_of_points-1 , 0);
			}

			cudaStreamSynchronize(stream2); // Synchronize stream 2 to make sure that the data has arrived


			// Find dimensions
			findDimmensionsKernel(dimGrid, dimBlock, stream1, samples_d, centroids_d, data_d,  findDim_d,
								  findDim_count_d, dim, arr_sizes.number_of_samples, sample_size, arr_sizes.number_of_centroids,
								  m, width, number_of_points, this->findDimKernelVersion);


			// Create indices for the scores
			createIndicesKernel(dimGrid, dimBlock, stream2, index_d, arr_sizes.number_of_samples);

			// make sure the indexes are created before argmax is called
			cudaStreamSynchronize(stream2);

			
			// Find highest scoring
			argMaxKernelWidthUpperBound(dimGrid, dimBlock, sharedMemSize, stream1, findDim_count_d, index_d, arr_sizes.number_of_samples, this->d0);

			
			// Output Points in that cluster, along with the dimensions and centroid
			unsigned int* best_index_h = (unsigned int*) malloc(sizeof(unsigned int));
			checkCudaErrors(cudaMemcpyAsync(best_index_h, index_d, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream1));

			// Output Points in that cluster, along with the dimensions and centroid
			unsigned int* best_score_h = (unsigned int*) malloc(sizeof(unsigned int));
			checkCudaErrors(cudaMemcpyAsync(best_score_h, findDim_count_d, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream1));
			// Synchronize to make sure that the value have arrived to the host side
			cudaStreamSynchronize(stream1);
			// std::cout << "best score: " << best_score_h[0] << std::endl;
			if(maxScore < best_score_h[0]){ // if the new score is better than the current;
				maxScore = best_score_h[0];
				// Find points contained
				
				/*
				pointsContainedKernelNaive(dimGrid, dimBlock, stream1, data_d,
										   centroids_d+(((size_t)(best_index_h[0]/m))),
										   findDim_d+(best_index_h[0]*dim),
										   pointsContained_d, pointsContained_count_d,
										   width, dim, number_of_points, 1, 1);
				*/
				assert(dimGrid*dimBlock >= number_of_centroids);
				whatDataIsInCentroid(ceilf((float)number_of_points/dimBlock),
									 dimBlock,
									 stream1,
									 pointsContained_d,
									 data_d,
									 centroids_d+(((size_t)(best_index_h[0]/m))),
									 findDim_d+(best_index_h[0]*dim),
									 width,
									 dim,
									 number_of_points);


				bool* b_h = (bool*) malloc((number_of_points+1)*sizeof(bool));
				checkCudaErrors(cudaMemcpyAsync(b_h, pointsContained_d,
												(number_of_points+1)*sizeof(bool), cudaMemcpyDeviceToHost, stream1));
				// cudaStreamSynchronize(stream1);
				// for(unsigned int g = 0; g < number_of_points+1; g++){
				// 	std::cout << b_h[g] << " ";
				// }
				// std::cout << std::endl;
				
				//prefixsum....
												 

				// Allocate space for the size of the cluster
				sum_scan_blelloch(stream1, prefixSum_d,pointsContained_d,(number_of_points+1), true);
				unsigned int* cluster_size_h = (unsigned int*) malloc(sizeof(unsigned int));
				checkCudaErrors(cudaMemcpyAsync(cluster_size_h, prefixSum_d+number_of_points,
												sizeof(unsigned int), cudaMemcpyDeviceToHost, stream1));


				// cudaStreamSynchronize(stream1);
				// unsigned int* prefix_h = (unsigned int*) malloc((number_of_points+1)*sizeof(unsigned int));
				// checkCudaErrors(cudaMemcpyAsync(prefix_h, prefixSum_d,
				// 								(number_of_points+1)*sizeof(unsigned int), cudaMemcpyDeviceToHost, stream1));
				// cudaStreamSynchronize(stream1);
				// for(unsigned int g = 0; g < number_of_points+1; g++){
				// 	std::cout << prefix_h[g] << " ";
				// }
				// std::cout << std::endl;

				
				// make sure that the size have arrived on the host side
				cudaStreamSynchronize(stream1);
				cluster_size_h[0] = number_of_points-cluster_size_h[0];
				// std::cout << "cluster_size: " << cluster_size_h[0] << std::endl;
				// allocate space for the cluster on host side, and copy it.
				unsigned int size_of_cluster = cluster_size_h[0]*dim*sizeof(float);

				
				
				
				auto cluster = new std::vector<std::vector<float>*>;
				auto dims = new std::vector<bool>;
				// std::cout << "prefixsum: " << cluster_size_h[0] << std::endl;
				if(cluster_size_h[0] < this->alpha*number_of_points){
					// cluster is too small, return empty clusters, and try again....
					// this might cause it to loop for the rest of the iterations and do nothing, maybe break here?
				}else{
					//float* cluster_d;
					//checkCudaErrors(cudaMalloc((void **) &cluster_d, size_of_cluster));
					checkCudaErrors(cudaMemcpyAsync(bestDims_d, pointsContained_d,
													number_of_points, cudaMemcpyDeviceToDevice, stream1));



					// create output cluster
					//notDevice(dimGrid, dimBlock, stream1, pointsContained_d, number_of_points);
					cudaStreamSynchronize(stream1);
					assert(sizes.size_of_data >= number_of_points*dim*sizeof(float));

					// deleteFromArray(stream1, outputCluster_d, pointsContained_d,
					// 				data_d, number_of_points, dim, true);
					// std::cout << "number_of_points: " << number_of_points << std::endl;
					deleteFromArrayWrapper(ceilf((float)(number_of_points*dim)/dimBlock), dimBlock,
										   stream1, data_d, prefixSum_d, 
										   number_of_points, dim, outputCluster_d);

					
					// copy the cluster to Host
					float* cluster_h;
					checkCudaErrors(cudaMallocHost((void**) &cluster_h, size_of_cluster));
					checkCudaErrors(cudaMemcpyAsync(cluster_h, outputCluster_d, size_of_cluster, cudaMemcpyDeviceToHost, stream1));

					// std::cout << "best index: " << best_index_h[0] << std::endl;
					// copy the used dimmensions to host.
					bool* output_dims_h;
					checkCudaErrors(cudaMallocHost((void**) &output_dims_h, dim*sizeof(bool)));
					checkCudaErrors(cudaMemcpyAsync(output_dims_h, findDim_d+(best_index_h[0]*dim),
													dim, cudaMemcpyDeviceToHost, stream1));

					// syncronise before deleting
					cudaStreamSynchronize(stream1);

					// Create the output in vector format
					// IDEA: create a seperate thread for this, because it just needs to be done before return.
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

					// deleting the variables used for cluster
					checkCudaErrors(cudaFreeHost(cluster_h));
					checkCudaErrors(cudaFreeHost(output_dims_h));
				}
				// build result & and delete old
				delete res.first;
				delete res.second;
				res = std::make_pair(cluster, dims);
				// clean up
				free(best_index_h);
				free(cluster_size_h);
			}
			j++;

		}
		float* newData_d = outputCluster_d+res.first->size()*dim;
		// Delete the extracted cluster from the data
		assert(sizes.size_of_data >= number_of_points*dim*sizeof(float));
		deleteFromArray(stream2, newData_d, bestDims_d, data_d, number_of_points, dim);
		size_t a = res.first->size();
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
