#include "DOCGPU_Kernels.h"
#include <iostream>
#include <vector>
#include <math.h>
#include <random>
#include <assert.h>
#include <unistd.h>
#include <stdio.h>

/*
 * Finds the hypercube the for each medoid - discriminating set combination
*/
__global__ void findDimmensionsDevice(unsigned int* Xs_d, unsigned int* ps_d, float* data, bool* res_d, unsigned int* Dsum_out,
									  unsigned int point_dim, unsigned int no_of_samples, unsigned int no_in_sample, unsigned int no_of_ps, unsigned int m, float width, unsigned int no_data){
	unsigned int entry = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int pNo = entry/m;
	if(entry < no_of_samples){
		assert(pNo < no_of_ps);
		unsigned int Dsum = 0;
		// for each dimension
		unsigned int tmp = ps_d[pNo]; // tmp is the index of the medoid in the dataset
		for(int i = 0; i < point_dim; i++){
			bool d = true;
			assert(tmp < no_data);
			float p_tmp = data[tmp*point_dim+i];
			// for each point in sample
			for(unsigned j = 0; j < no_in_sample; j++){
				assert(entry*no_in_sample+j < no_of_samples*no_in_sample);
				unsigned int sampleNo = Xs_d[entry*no_in_sample+j];
				assert(entry*no_in_sample+j < no_of_samples*no_in_sample);
				assert(sampleNo < no_data);
				float point = data[sampleNo*point_dim+i];
				d &= abs(p_tmp-point) < width;
			}
			res_d[entry*point_dim+i] = d;
			Dsum += d;
		}
		Dsum_out[entry] = Dsum;
	}
}



// Hack
struct floatArray{
	float f0;
	float f1;
	float f2;
	float f3;
	float f4;
	float f5;
	float f6;
	float f7;
};

struct boolArray{
	bool b0;
	bool b1;
	bool b2;
	bool b3;
	bool b4;
	bool b5;
	bool b6;
	bool b7;
};

/*
 * Finds the hypercube the for each medoid - discriminating set combination
 * From simple testing and nvprof, this kernel gets around twice the memory throughput as the naive version. 
 * 6 in a chunk allows for oly use 32 registers, 8 will give somewhat better memory perfomance
*/
__global__ void findDimmensionsLoadChunks(unsigned int* Xs_d, unsigned int* ps_d, float* data, bool* res_d, unsigned int* Dsum_out,
									  unsigned int point_dim, unsigned int no_of_samples, unsigned int no_in_sample, unsigned int no_of_ps, unsigned int m, float width, unsigned int no_data){
	const unsigned int entry = blockIdx.x*blockDim.x+threadIdx.x;
	floatArray p_tmp;
	floatArray x_tmp;
	unsigned int d;
	if(entry < no_of_samples){
		assert(entry/m < no_of_ps);
		unsigned int Dsum = 0;
		// for each dimension
		unsigned int tmp = ps_d[entry/m]; // tmp is the index of the medoid in the dataset
		for(int i = 0; i < point_dim; i +=8){
			assert(tmp < no_data);
			if(i+8 > point_dim){
				for(; i < point_dim; i++){
					// if(threadIdx.x + blockIdx.x*blockDim.x == 0) printf("i %u  point_dim %u\n", i, point_dim);
					d = 1;
					assert(tmp < no_data);
					p_tmp.f0 = data[tmp*point_dim+i];
					// for each point in sample
					for(unsigned j = 0; j < no_in_sample; j++){
						assert(entry*no_in_sample+j < no_of_samples*no_in_sample);
						assert(entry*no_in_sample+j < no_of_samples*no_in_sample);
						unsigned int sampleNo = Xs_d[entry*no_in_sample+j];
						assert(sampleNo < no_data);
						x_tmp.f0 = data[sampleNo*point_dim+i];
						d &= abs(p_tmp.f0-x_tmp.f0) < width;
					}
					res_d[entry*point_dim+i] = d;
					Dsum += d;
				}
				break;
			}else{
				p_tmp = *((floatArray*) (data+tmp*point_dim+i));// data[tmp*point_dim+i];
				// for each point in sample
				d = 0xff;
				for(unsigned j = 0; j < no_in_sample; j++){
					unsigned int sampleNo = Xs_d[entry*no_in_sample+j];
					x_tmp = *((floatArray*)(data+sampleNo*point_dim+i));// data[tmp*point_dim+i];

   


					if (!(abs(p_tmp.f0-x_tmp.f0) < width)){d &= ~(1 << 0);}
					if (!(abs(p_tmp.f1-x_tmp.f1) < width)){d &= ~(1 << 1);}
					if (!(abs(p_tmp.f2-x_tmp.f2) < width)){d &= ~(1 << 2);}
					if (!(abs(p_tmp.f3-x_tmp.f3) < width)){d &= ~(1 << 3);}
					if (!(abs(p_tmp.f4-x_tmp.f4) < width)){d &= ~(1 << 4);}
					if (!(abs(p_tmp.f5-x_tmp.f5) < width)){d &= ~(1 << 5);}
					if (!(abs(p_tmp.f6-x_tmp.f6) < width)){d &= ~(1 << 6);}
					if (!(abs(p_tmp.f7-x_tmp.f7) < width)){d &= ~(1 << 7);}
					
					// d &= (abs(p_tmp.f0-x_tmp.f0) < width) ? ((0xFF) & (1 << 0)): ;
					// d &= (0xFF) | ((abs(p_tmp.f1-x_tmp.f1) < width) << 1);
					// d &= (0xFF) | ((abs(p_tmp.f2-x_tmp.f2) < width) << 2);
					// d &= (0xFF) | ((abs(p_tmp.f3-x_tmp.f3) < width) << 3);
					// d &= (0xFF) | ((abs(p_tmp.f4-x_tmp.f4) < width) << 4);
					// d &= (0xFF) | ((abs(p_tmp.f5-x_tmp.f5) < width) << 5);
					// d &= (0xFF) | ((abs(p_tmp.f6-x_tmp.f6) < width) << 6);
					// d &= (0xFF) | ((abs(p_tmp.f7-x_tmp.f7) < width) << 7);
					// if(threadIdx.x == 0){
					// 	if((d >> 0) & 1 == 1){printf("1");}else{printf("0");};
					// 	if((d >> 1) & 1 == 1){printf("1");}else{printf("0");};
					// 	if((d >> 2) & 1 == 1){printf("1");}else{printf("0");};
					// 	if((d >> 3) & 1 == 1){printf("1");}else{printf("0");};
					// 	if((d >> 4) & 1 == 1){printf("1");}else{printf("0");};
					// 	if((d >> 5) & 1 == 1){printf("1");}else{printf("0");};
					// 	if((d >> 6) & 1 == 1){printf("1");}else{printf("0");};
					// 	if((d >> 7) & 1 == 1){printf("1");}else{printf("0");};
					// }
					
					// if(threadIdx.x == 0) printf("i %u j %u p %f x %f from %u\n", i+0, j, p_tmp.f0, x_tmp.f0, data+sampleNo*point_dim+i);
					// if(threadIdx.x == 0) printf("i %u j %u p %f x %f from %u\n", i+1, j, p_tmp.f1, x_tmp.f1, data+sampleNo*point_dim+i+1);
					// if(threadIdx.x == 0) printf("i %u j %u p %f x %f from %u\n", i+2, j, p_tmp.f2, x_tmp.f2, data+sampleNo*point_dim+i+2);
					// if(threadIdx.x == 0) printf("i %u j %u p %f x %f from %u\n", i+3, j, p_tmp.f3, x_tmp.f3, data+sampleNo*point_dim+i+3);

					
				}
				//*((boolArray*)res_d[entry*point_dim+i]) = d;
				
				res_d[entry*point_dim+i  ] = (d >> 0) & 1;
				res_d[entry*point_dim+i+1] = (d >> 1) & 1;
				res_d[entry*point_dim+i+2] = (d >> 2) & 1;
				res_d[entry*point_dim+i+3] = (d >> 3) & 1;
				res_d[entry*point_dim+i+4] = (d >> 4) & 1;
				res_d[entry*point_dim+i+5] = (d >> 5) & 1;
				res_d[entry*point_dim+i+6] = (d >> 6) & 1;
				res_d[entry*point_dim+i+7] = (d >> 7) & 1;
				Dsum += __popc(d);//d.b0 + d.b1 + d.b2 +d.b3 +d.b4 +d.b5 +d.b6 +d.b7;
			}
		}
		Dsum_out[entry] = Dsum;
	}
}




__global__ void score(unsigned int* Cluster_size, unsigned int* Dim_count, float* score_output, unsigned int len, float alpha, float beta, unsigned int num_points){
	int entry = blockIdx.x*blockDim.x+threadIdx.x;
	if(entry < len){
		// score_output[entry] = ((Cluster_size[entry])* powf(1.0/beta, (Dim_count[entry])))*(Cluster_size[entry] >= (alpha*num_points));
		score_output[entry] = (logf(Cluster_size[entry])+ logf(1.0/beta)*((Dim_count[entry])))*(Cluster_size[entry] >= (alpha*num_points));
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
				if(min == max){
					result[i*number_of_states+idx] = min;
				}else{
					float myrandf = curand_uniform(&states[idx]);
					myrandf *= (max - min + 0.9999);
					unsigned int res = (unsigned int)truncf(myrandf);
					res %= max;
					res += min;
					assert(res >= min);
					if(!(res <= max)){
						printf("res: %u, max: %u \n", res, max);
					}
					assert(res <= max);
					result[i*number_of_states+idx] = res;	
				}
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
						   unsigned int m, float width, unsigned int no_data,
						   findDimVersion version){

	if(version == naiveFindDim){
	    findDimmensionsDevice<<<dimGrid, dimBlock, 0, stream>>>(Xs_d, ps_d, data, res_d, Dsum_out,
												 point_dim, no_of_samples, sample_size,
												 no_of_ps, m, width, no_data);	
	}else{
		findDimmensionsLoadChunks<<<dimGrid, dimBlock, 0, stream>>>(Xs_d, ps_d, data, res_d, Dsum_out,
												 point_dim, no_of_samples, sample_size,
												 no_of_ps, m, width, no_data);	
	}
	
};



void scoreKernel(unsigned int dimGrid, unsigned int dimBlock, cudaStream_t stream,
				 unsigned int* cluster_size, unsigned int* dim_count, float* score_output,
				 unsigned int len, float alpha, float beta, unsigned int num_points){

	score<<<dimGrid, dimBlock, 0, stream>>>(cluster_size, dim_count, score_output,
								 len, alpha, beta, num_points);

	

	
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




std::pair<std::vector<std::vector<bool>*>*,std::vector<unsigned int>*> findDimmensionsChunk(std::vector<std::vector<float>*>* data,
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

	findDimmensionsLoadChunks<<<ceil((no_of_samples)/256.0), 256>>>(samples_d, centroids_d,
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











