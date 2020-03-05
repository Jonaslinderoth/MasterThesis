#include "DOCGPU_Kernels.h"
#include <iostream>
#include <vector>
#include <math.h>
#include <random>
#include <assert.h>
#include <unistd.h>
#include <stdio.h>

__global__ void findDimmensionsDevice(unsigned int* Xs_d, unsigned int* ps_d, float* data, bool* res_d, unsigned int* Dsum_out,
									  unsigned int point_dim, unsigned int no_of_samples, unsigned int no_in_sample, unsigned int no_of_ps, unsigned int m, float width, unsigned int no_data){
	int entry = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int pNo = entry/m;
	if(entry < no_of_samples){
		if(!(pNo < no_of_ps)){
			printf("%u ", no_of_ps);
		}
		assert(pNo < no_of_ps);
		unsigned int Dsum = 0;
		// for each dimension
		for(int i = 0; i < point_dim; i++){
			bool d = true;
			unsigned int tmp = ps_d[pNo];
			assert(tmp < no_data);
			float p_tmp = data[tmp*point_dim+i];
			// for each point in sample
			for(unsigned j = 0; j < no_in_sample; j++){
				assert(entry*no_in_sample+j < no_of_samples*no_in_sample);
				unsigned int sampleNo = Xs_d[entry*no_in_sample+j];
				assert(entry*no_in_sample+j < no_of_samples*no_in_sample);
				if(!(sampleNo < no_data)){
					printf("entry %u, no_in_sample %u, j %u, sampleNo %u, entry*no_in_sample+j %u, no_data %u \n", entry, no_in_sample, j, sampleNo, entry*no_in_sample+j, no_data);
				}
				//assert(sampleNo < no_data);
				float point = data[sampleNo*point_dim+i];
				d &= abs(p_tmp-point) < width;
			}
			res_d[entry*point_dim+i] = d;
			Dsum += d;

		}
		Dsum_out[entry] = Dsum;
	}
}

__global__ void pointsContainedDevice(float* data, unsigned int* centroids, bool* dims, bool* output, unsigned int* Csum_out,
									  float width, unsigned int point_dim, unsigned int no_data, unsigned int no_dims, unsigned int m){
	// one kernel for each hypercube
	unsigned int entry = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int currentCentroid = entry/m;
	if(entry < no_dims){
		//assert(currentCentroid < no_of_ps);
		// for each data point
		unsigned int Csum = 0;
		for(unsigned int j = 0; j < no_data; j++){
			// for all dimmensions in each hypercube / point
			bool d = true;
			for(unsigned int i = 0; i < point_dim; i++){
				//(not (dims[entry*point_dim+i])) ||
				unsigned int centroid_index = centroids[currentCentroid];
				d &= (not (dims[entry*point_dim+i])) || (abs(data[centroid_index*point_dim+i] - data[j*point_dim+i]) < width);
			}

			if(!((size_t)entry*(size_t)no_data+(size_t)j < 2026532205)){
				//printf("entry %u, no_data %u, j %u, no_dims %u \n", entry, no_data, j, no_dims);
			}
			assert(entry < no_dims);
			assert((size_t)entry*(size_t)no_data+(size_t)j < (size_t)no_dims*(size_t)no_data+(size_t)j);
			//assert((size_t)entry*(size_t)no_data+(size_t)j < 2026532205);
			
			output[(size_t)entry*(size_t)no_data+(size_t)j] = d;
			Csum += d;
		}
		Csum_out[entry] = Csum;
	}
}

__global__ void score(unsigned int* Cluster_size, unsigned int* Dim_count, float* score_output, unsigned int len, float alpha, float beta, unsigned int num_points){
	int entry = blockIdx.x*blockDim.x+threadIdx.x;
	if(entry < len){
		score_output[entry] = ((Cluster_size[entry])* powf(1.0/beta, (Dim_count[entry])))*(Cluster_size[entry] >= (alpha*num_points));	
	}

}



float* scoreHost(unsigned int* Cluster_size, unsigned int* Dim_count, float* score_output, int len, float alpha, float beta, unsigned int number_of_points){
	unsigned int* Cluster_size_d;
	unsigned int* Dim_count_d;
	float* score_output_d;
	cudaMalloc((void **) &Cluster_size_d, len*sizeof(unsigned int));
	cudaMalloc((void **) &Dim_count_d, len*sizeof(unsigned int));
	cudaMalloc((void **) &score_output_d, len*sizeof(float));
	
	cudaMemcpy(Cluster_size_d, Cluster_size, len*sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(Dim_count_d, Dim_count, len*sizeof(unsigned int), cudaMemcpyHostToDevice);

	score<<<ceil((len)/256.0), 256>>>(Cluster_size_d, Dim_count_d, score_output_d, len, alpha, beta, number_of_points);


	cudaMemcpy(score_output, score_output_d, len*sizeof(float), cudaMemcpyDeviceToHost);
	return score_output;
	
}


__global__ void createIndices(unsigned int* index, unsigned int length){
	unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
	if(i < length){
		index[i] = i;	
	}

};

__global__ void argMaxDevice(float* scores, unsigned int* scores_index, unsigned int input_size){
	extern __shared__ int array[];
	int* argData = (int*)array;
	float* scoreData = (float*) &argData[blockDim.x];
	
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;

	argData[tid] = 0;
	scoreData[tid] = 0;

	if(i < input_size){
	argData[tid] = scores_index[i];
	scoreData[tid] = scores[i];
	
	__syncthreads();		
		for(unsigned int s=(blockDim.x/2); s > 0; s/=2) {
			if(tid < s){
				assert(tid+s < blockDim.x);
				if(scoreData[tid] < scoreData[tid+s]){
					scoreData[tid] = scoreData[tid+s];
					argData[tid] = argData[tid+s];
				}
			}
			__syncthreads();
		}
	
		if(tid == 0){
			scores_index[blockIdx.x] = argData[0];
			scores[blockIdx.x] = scoreData[0];
		}; 
		
	}
	
}
__global__ void randIntArrayInit(curandState_t* states , unsigned int seed, unsigned int size){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx < size)
		{	curand_init(seed,   /* the seed controls the sequence of random values that are produced */
			idx, /* the sequence number is only important with multiple cores */
			0,/* the offset is how much extra we advance in the sequence for each call, can be 0 */
			&states[idx]);
		}
}



/**
   Number of threads hsould be the same as the number of random states
 */
__global__ void randIntArray(unsigned int *result , curandState_t* states , const unsigned int number_of_states,
							 const unsigned int size , const unsigned int max, const unsigned min){
	unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int numberPrThread = ceilf((float)size/(float)number_of_states); // rounded down

	/*if(idx == 0){
		printf("number pr thread %u \n", numberPrThread);
		printf("size of number of states %u \n", number_of_states);
		printf("size of array %u \n", size);
		}*/
	
	if(idx < number_of_states){
		for(int i = 0; i < numberPrThread; i++){
			if(i*number_of_states+idx < size){
				float myrandf = curand_uniform(&states[idx]);
				myrandf *= (max - min + 0.9999);
				unsigned int res = (unsigned int)truncf(myrandf);
				res %= max;
				res += min;
				assert(res >= min);
				assert(res <= max);
				result[i*number_of_states+idx] = res;
			}
		}		
	}

}


__global__ void notKernel(bool* array, unsigned int length){
	unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int numThreads = blockDim.x*gridDim.x;
	unsigned int numberPrThread = ceilf((float)length/(float)numThreads); // rounded down
	for(int i = 0; i < numberPrThread; i++){
		if(numberPrThread*i+idx < length){
			array[numberPrThread*i+idx] = not array[numberPrThread*i+idx];
		}
	}	
}

void notDevice(unsigned int dimGrid, unsigned int dimBlock, cudaStream_t stream, bool* array, unsigned int length){
	notKernel<<<dimGrid, dimBlock, 0, stream>>>(array, length);
}

void findDimmensionsKernel(unsigned int dimGrid, unsigned int dimBlock, cudaStream_t stream,
						   unsigned int* Xs_d, unsigned int* ps_d, float* data, bool* res_d,
						   unsigned int* Dsum_out, unsigned int point_dim,
						   unsigned int no_of_samples, unsigned int sample_size,
						   unsigned int no_of_ps,
						   unsigned int m, float width, unsigned int no_data){

    findDimmensionsDevice<<<dimGrid, dimBlock, 0, stream>>>(Xs_d, ps_d, data, res_d, Dsum_out,
												 point_dim, no_of_samples, sample_size,
												 no_of_ps, m, width, no_data);
	
};

void pointsContainedKernel(unsigned int dimGrid, unsigned int dimBlock, cudaStream_t stream,
						   float* data, unsigned int* centroids, bool* dims, bool* output, unsigned int* Csum_out,
						   float width, unsigned int point_dim, unsigned int no_data, unsigned int number_of_samples,
						   unsigned int m){

	pointsContainedDevice<<<dimGrid, dimBlock, 0, stream>>>(data, centroids, dims,
												 output, Csum_out,
												 width, point_dim, no_data, number_of_samples, m);
	
};


void scoreKernel(unsigned int dimGrid, unsigned int dimBlock, cudaStream_t stream,
				 unsigned int* cluster_size, unsigned int* dim_count, float* score_output,
				 unsigned int len, float alpha, float beta, unsigned int num_points){

	score<<<dimGrid, dimBlock, 0, stream>>>(cluster_size, dim_count, score_output,
								 len, alpha, beta, num_points);

	

	
};

void createIndicesKernel(unsigned int dimGrid, unsigned int dimBlock, cudaStream_t stream, unsigned int* index, unsigned int length){
	createIndices<<<dimGrid, dimBlock, 0, stream>>>(index, length);

};

void argMaxKernel(unsigned int dimGrid, unsigned int dimBlock, unsigned int sharedMemorySize,
				  cudaStream_t stream,
				  float* scores, unsigned int* scores_index, unsigned int input_size){

	unsigned int* out = (unsigned int*) malloc(sizeof(unsigned int)*input_size);
	float* outScores = (float*) malloc(sizeof(float)*input_size);
	
	
	
	unsigned int out_size = input_size;
	while(out_size > 1){
	   
		argMaxDevice<<<dimGrid, dimBlock, sharedMemorySize, stream>>>(scores, scores_index, out_size);
		out_size = dimGrid;
		dimGrid = ceil((float)out_size/(float)dimBlock);
	}
	
};


std::pair<std::vector<std::vector<bool>*>*,std::vector<unsigned int>*> pointsContained(std::vector<std::vector<bool>*>* dims,
																					   std::vector<std::vector<float>*>* data,
																					   std::vector<unsigned int>* centroids,
																					   int m, float width){

	// Calculaating sizes
	int point_dim = data->at(0)->size();
	int no_of_points = data->size();
	int no_of_dims = dims->size();
	int no_of_centroids = centroids->size();

	int floats_in_data = point_dim * no_of_points;
	int bools_in_dims = no_of_dims * point_dim;
	int bools_in_output = no_of_points * no_of_dims;
	int ints_in_output_count = no_of_dims;
	
	int size_of_data = floats_in_data*sizeof(float);
	int size_of_dims = bools_in_dims*sizeof(bool);
	int size_of_centroids = no_of_centroids*sizeof(unsigned int);
	int size_of_output = bools_in_output*sizeof(bool);
	int size_of_output_count = ints_in_output_count*sizeof(unsigned int);

	// allocating on the host
	float* data_h = (float*) malloc(size_of_data);
	bool* dims_h = (bool*) malloc(size_of_dims);
	unsigned int* centroids_h = (unsigned int*) malloc(size_of_centroids);
	bool* output_h = (bool*) malloc(size_of_output);
	unsigned int* output_count_h = (unsigned int*) malloc(size_of_output_count);

	// filling data array
	for(int i= 0; i < no_of_points; i++){
		for(int j = 0; j < point_dim; j++){
			data_h[i*point_dim+j] = data->at(i)->at(j);
		}
	}

	// filling dims array
	for(int i= 0; i < no_of_dims; i++){
		for(int j = 0; j < point_dim; j++){
			dims_h[i*point_dim+j] = dims->at(i)->at(j);
		}
	}

	// filling centroid array
	for(int i= 0; i < no_of_centroids; i++){
		centroids_h[i] = centroids->at(i);
	}

	// allocating on device
	float* data_d;
	bool* dims_d;
	unsigned int* centroids_d;
	bool* output_d;
	unsigned int* output_count_d;
	
	cudaMalloc((void **) &data_d, size_of_data);
	cudaMalloc((void **) &dims_d, size_of_dims);
	cudaMalloc((void **) &centroids_d, size_of_centroids);
	cudaMalloc((void **) &output_d, size_of_output);
	cudaMalloc((void **) &output_count_d, size_of_output_count);

	//Copy from host to device
				
	cudaMemcpy(data_d, data_h, size_of_data, cudaMemcpyHostToDevice);
	cudaMemcpy(dims_d, dims_h, size_of_dims, cudaMemcpyHostToDevice);
	cudaMemcpy(centroids_d, centroids_h, size_of_centroids, cudaMemcpyHostToDevice);


	// Call kernel
	pointsContainedDevice<<<ceil((no_of_dims)/256.0), 256>>>(data_d, centroids_d, dims_d, output_d, output_count_d, 
															 width, point_dim, no_of_points, no_of_dims, m);

	
	// copy from device
	cudaMemcpy(output_h, output_d, size_of_output, cudaMemcpyDeviceToHost);
	cudaMemcpy(output_count_h, output_count_d, size_of_output_count, cudaMemcpyDeviceToHost);

	// construnct output
	auto output =  new std::vector<std::vector<bool>*>;
	auto output_count =  new std::vector<unsigned int>;
   	
	
	for(int i = 0; i < no_of_dims; i++){
		auto a =  new std::vector<bool>;
		for(int j = 0; j < no_of_points; j++){
			a->push_back(output_h[i*no_of_points+j]);
		}
		output->push_back(a);
	}

	for(int i = 0; i < no_of_dims; i++){
		output_count->push_back(output_count_h[i]);
	}


	cudaFree(data_d);
	cudaFree(dims_d);
	cudaFree(centroids_d);
	cudaFree(output_d);
	cudaFree(output_count_d);
	free(data_h);
	free(dims_h);
	free(centroids_h);
	free(output_h);
	free(output_count_h);

	
	return std::make_pair(output,output_count);
};


std::pair<std::vector<std::vector<bool>*>*,std::vector<unsigned int>*> findDimmensions(std::vector<std::vector<float>*>* data,
																					   std::vector<unsigned int>* centroids,
																					   std::vector<std::vector<unsigned int>*>* samples,
																					   int m, float width){

	int no_of_samples = samples->size();	
	int no_in_sample = samples->at(0)->size();
	int no_of_centroids = centroids->size();

	int no_of_points = data->size();
	int point_dim = data->at(0)->size();
   
	int sizeOfData = no_of_points*point_dim*sizeof(unsigned int);
	int sizeOfSamples = no_of_samples*no_in_sample*sizeof(unsigned int);
	int sizeOfCentroids = point_dim*no_of_centroids*sizeof(unsigned int);
	
	
	unsigned int* centroids_h = (unsigned int*) malloc(sizeOfCentroids);
	unsigned int* samples_h = (unsigned int*) malloc(sizeOfSamples);
	float* data_h = (float*) malloc(sizeOfData);
	

	for(int i = 0; i < no_of_points; i++){
		for(int j = 0; j < point_dim; j++){
			data_h[i*point_dim+j] = data->at(i)->at(j);
		}
	}

	for(int i = 0; i < no_of_samples; i++){
		for(int j = 0;  j < no_in_sample; j++){
			samples_h[i*no_in_sample+j] = samples->at(i)->at(j);
		}
	}

	for(int i = 0; i < no_of_centroids; i++){
		centroids_h[i] = centroids->at(i);
	}

	unsigned int size_of_count = (no_of_samples)*sizeof(unsigned int);
	
	int outputDim = no_of_samples*point_dim;		
	int outputSize = outputDim*sizeof(bool);
	bool* result_h = (bool*) malloc(outputSize);
	unsigned int* count_h = (unsigned int*) malloc(size_of_count);


	unsigned int* samples_d;
	unsigned int* centroids_d;
	float* data_d;
	bool* result_d;
	unsigned int* count_d;
	
	cudaMalloc((void **) &samples_d, sizeOfSamples);
	cudaMalloc((void **) &centroids_d, sizeOfCentroids);
	cudaMalloc((void **) &data_d, sizeOfData);
	cudaMalloc((void **) &result_d, outputSize);
	cudaMalloc((void **) &count_d, size_of_count);

	cudaMemcpy( samples_d, samples_h, sizeOfSamples, cudaMemcpyHostToDevice);
    cudaMemcpy( centroids_d, centroids_h, sizeOfCentroids, cudaMemcpyHostToDevice);
	cudaMemcpy( data_d, data_h, sizeOfData, cudaMemcpyHostToDevice);


	findDimmensionsDevice<<<ceil((no_of_samples)/256.0), 256>>>(samples_d, centroids_d,
																data_d, result_d, count_d,
																point_dim, no_of_samples,
																no_in_sample, no_of_centroids,
																m, width,no_of_points );

   
	cudaMemcpy(result_h, result_d, outputSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(count_h, count_d, size_of_count, cudaMemcpyDeviceToHost);


	

	auto output =  new std::vector<std::vector<bool>*>;
	
	for(int i = 0; i < no_of_samples; i++){
		auto a =  new std::vector<bool>;
		for(int j = 0; j < point_dim; j++){
			a->push_back(result_h[i*point_dim+j]);
		}
		output->push_back(a);
	}


	auto count = new std::vector<unsigned int>;
	for(int i = 0; i < (no_of_samples); i++){
		count->push_back(count_h[i]);
	}

	cudaFree(samples_d);
	cudaFree(centroids_d);
	cudaFree(result_d);
	cudaFree(count_d);
	free(result_h);
	free(count_h);
	free(centroids_h);
	free(samples_h);
	
	return std::make_pair(output, count);
}



int argMax(std::vector<float>* scores){
	//Calculate size of shared Memory, block and thread dim
	//fetch device info
	// TODO: hardcoded device 0
	int smemSize, maxBlock;
	cudaDeviceGetAttribute(&smemSize, 
    cudaDevAttrMaxSharedMemoryPerBlock, 0);
	cudaDeviceGetAttribute(&maxBlock, 
						   cudaDevAttrMaxThreadsPerBlock, 0); 

	// caluclate the maximum thread size based on shared mem requirements and maximum threads
	int dimBlock = smemSize/(sizeof(int)+sizeof(float));
	if(dimBlock > maxBlock) dimBlock = maxBlock;
	int dimGrid = ceil((float)scores->size()/(float)dimBlock);
	int sharedMemSize = (dimBlock*sizeof(unsigned int) + dimBlock*sizeof(float));

	int size_of_score = scores->size()*sizeof(float);
	int size_of_score_index = scores->size()*sizeof(unsigned int);

	
	float* scores_h = (float*) malloc(size_of_score);
	float* scores_d;
	unsigned int* scores_index_d;

	for(int i = 0; i < scores->size(); i++){
		scores_h[i] = scores->at(i);
	}
	

	cudaMalloc((void **) &scores_d, size_of_score);
	cudaMalloc((void **) &scores_index_d, size_of_score_index);

	cudaMemcpy(scores_d, scores_h, size_of_score, cudaMemcpyHostToDevice);
	
	//Call kernel
	int out_size = scores->size();
   
	createIndices<<<dimGrid, dimBlock>>>(scores_index_d, out_size);
   
	
	while(out_size > 1){
		//std::cout << "called" << std::endl;
		//std::cout << "dimGrid:" << dimGrid << " dimBlock: " << dimBlock << " out_size: " << out_size << std::endl;
		argMaxDevice<<<dimGrid, dimBlock, sharedMemSize>>>(scores_d, scores_index_d, out_size);				
		out_size = dimGrid;
		dimGrid = ceil((float)out_size/(float)dimBlock);
	}
	
	//argMaxDevice<<<dimGrid, dimBlock, sharedMemSize>>>(scores_d, scores_index_d, out_size);		

	unsigned int size_of_output = sizeof(unsigned int);
	
	unsigned int* scores_index_h = (unsigned int*) malloc(size_of_output);

	cudaMemcpy(scores_index_h, scores_index_d, size_of_output, cudaMemcpyDeviceToHost);
	
	int result = scores_index_h[0];
	cudaFree(scores_d);
	cudaFree(scores_index_d);
	free(scores_index_h);
	free(scores_h);

	return result;
	
	
}

/*
 * this function makes the states of the random number generator
 * this needs to be called before generateRandomIntArrayDevice.
 * "save" the states to save on compiutational time.
 */
bool generateRandomStatesArray(cudaStream_t stream,
							   curandState* d_randomStates,
							   const size_t size,
							   const bool randomSeed,
							   unsigned int seed,
							   unsigned int dimBlock){
	//set the seed
	if(randomSeed){
		std::random_device rd;
		seed = rd();
		//std::cout << "seed: " << seed << std::endl;
	}

#ifdef NDEBUG
	// nondebug
#else
	// debug code
	dimBlock = 512;
#endif
	//calculate the ammount of blocks
	int ammountOfBlocks = size/dimBlock;
	if(size%dimBlock != 0){
		ammountOfBlocks++;
	}
	randIntArrayInit<<<ammountOfBlocks,dimBlock, 0, stream>>>(d_randomStates ,seed, size);

	return true;
}


/* This fuction makes an array of random numbers between min and max in the gpu , given the allocation
 * and the states.
 * to get the states generate random states array call generateRandomStatesArray.
 */
bool generateRandomIntArrayDevice(cudaStream_t stream,
								  unsigned int* randomIndexes_d,
								  curandState* randomStates_d,
								  const size_t size_of_randomStates,
								  const size_t size,
								  const unsigned int max,
								  const unsigned int min,
								  unsigned int dimBlock){

	if(max<min){
		return false;
	}
	
	// if there is more random states than what we need, dont spawn too many threads
	size_t accual_size = size_of_randomStates;
	if(accual_size > size) accual_size = size;



	//calculate the ammount of blocks
	int ammountOfBlocks = accual_size/dimBlock;
	if(accual_size%dimBlock != 0){
		ammountOfBlocks++;
	}
	//std::cout << "number of blocks: " << ammountOfBlocks << " number of threads: " << dimBlock << " size: " << size << " number of states: " << size_of_randomStates <<std::endl; 

	//std::cout << "max: " << max << std::endl;
	//call the generation of random numbers
	randIntArray<<<ammountOfBlocks,dimBlock, 0, stream>>>(randomIndexes_d, randomStates_d, size_of_randomStates, size , max , min);

	return true;
}


