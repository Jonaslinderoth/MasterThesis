#include "ArgMax.h"
#include <assert.h>





__global__ void argMaxDevice(float* scores, unsigned int* scores_index, unsigned int input_size){
	extern __shared__ int array[];
	int* argData = (int*)array;
	float* scoreData = (float*) &argData[blockDim.x];

	unsigned int tid = threadIdx.x;
	unsigned int i = 2*blockIdx.x*blockDim.x+threadIdx.x;

	argData[tid] = 0;
	scoreData[tid] = 0;

	if(i < input_size){
		if(i+blockDim.x < input_size && scores[i] < scores[i+blockDim.x]){
			scoreData[tid] = scores[i+blockDim.x];
			argData[tid] = scores_index[i+blockDim.x];
		}else{
			scoreData[tid] = scores[i];
			argData[tid] = scores_index[i];			
		}

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


__global__ void argMaxDevice(unsigned int* scores, unsigned int* scores_index, unsigned int input_size){
	extern __shared__ int array[];
	int* argData = (int*)array;
	unsigned int* scoreData = (unsigned int*) &argData[blockDim.x];

	unsigned int tid = threadIdx.x;
	unsigned int i = 2*blockIdx.x*blockDim.x+threadIdx.x;

	argData[tid] = 0;
	scoreData[tid] = 0;

	if(i < input_size){
		if(i+blockDim.x < input_size && scores[i] < scores[i+blockDim.x]){
			scoreData[tid] = scores[i+blockDim.x];
			argData[tid] = scores_index[i+blockDim.x];
		}else{
			scoreData[tid] = scores[i];
			argData[tid] = scores_index[i];			
		}

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



__global__ void argMaxWithEarlyStop(unsigned int* scores, unsigned int* scores_index, unsigned int input_size, unsigned int bound){
	extern __shared__ int array[];
	int* argData = (int*)array;
	unsigned int* scoreData = (unsigned int*) &argData[blockDim.x];

	unsigned int tid = threadIdx.x;
	unsigned int i = 2*blockIdx.x*blockDim.x+threadIdx.x;

	argData[tid] = 0;
	scoreData[tid] = 0;

	if(i < input_size){
		if(i+blockDim.x < input_size && scores[i] < bound && scores[i] < scores[i+blockDim.x]){
			scoreData[tid] = scores[i+blockDim.x];
			argData[tid] = scores_index[i+blockDim.x];
		}else{
			scoreData[tid] = scores[i];
			argData[tid] = scores_index[i];			
		}

	__syncthreads();
		for(unsigned int s=(blockDim.x/2); s > 0; s/=2) {
			if(tid < s){
				assert(tid+s < blockDim.x);
				if(scoreData[tid] < bound && scoreData[tid] < scoreData[tid+s]){
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


__global__ void argMaxWithEarlyStop(float* scores, unsigned int* scores_index, unsigned int input_size, unsigned int bound){
	extern __shared__ int array[];
	unsigned int* argData = (unsigned int*)array;
	float* scoreData = (float*) &argData[blockDim.x];

	unsigned int tid = threadIdx.x;
	unsigned int i = 2*blockIdx.x*blockDim.x+threadIdx.x;

	argData[tid] = 0;
	scoreData[tid] = 0;

	if(i < input_size){
		if(i+blockDim.x < input_size && scores[i] < bound && scores[i] < scores[i+blockDim.x]){
			scoreData[tid] = scores[i+blockDim.x];
			argData[tid] = scores_index[i+blockDim.x];
		}else{
			scoreData[tid] = scores[i];
			argData[tid] = scores_index[i];			
		}

	__syncthreads();
		for(unsigned int s=(blockDim.x/2); s > 0; s/=2) {
			if(tid < s){
				assert(tid+s < blockDim.x);
				if(scoreData[tid] > bound && scoreData[tid+s] >= bound && argData[tid] > argData[tid+s]){
					scoreData[tid] = scoreData[tid+s];
					argData[tid] = argData[tid+s];					
				}
				if(scoreData[tid] < bound && scoreData[tid] < scoreData[tid+s]){
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



__global__ void createIndices(unsigned int* index, unsigned int length){
	unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
	if(i < length){
		index[i] = i;
	}

};


void createIndicesKernel(unsigned int dimGrid, unsigned int dimBlock, cudaStream_t stream, unsigned int* index, unsigned int length){
	createIndices<<<dimGrid, dimBlock, 0, stream>>>(index, length);

};





void argMaxKernel(unsigned int dimGrid, unsigned int dimBlock, unsigned int sharedMemorySize,
				  cudaStream_t stream,
				  float* scores, unsigned int* scores_index, unsigned int input_size){


	unsigned int out_size = input_size;
	while(out_size > 1){

		argMaxDevice<<<ceilf((float)dimGrid/2), dimBlock, sharedMemorySize, stream>>>(scores, scores_index, out_size);
		out_size = ceilf((float)dimGrid/2);
		dimGrid = ceil((float)out_size/(float)dimBlock);
	}

};


void argMaxKernel(unsigned int dimGrid, unsigned int dimBlock, unsigned int sharedMemorySize,
				  cudaStream_t stream,
				  unsigned int* scores, unsigned int* scores_index, unsigned int input_size){


	unsigned int out_size = input_size;
	while(out_size > 1){

		argMaxDevice<<<ceilf((float)dimGrid/2), dimBlock, sharedMemorySize, stream>>>(scores, scores_index, out_size);
		out_size = ceilf((float)dimGrid/2);
		dimGrid = ceil((float)out_size/(float)dimBlock);
	}

};


void argMaxKernelWidthUpperBound(unsigned int dimGrid, unsigned int dimBlock, unsigned int sharedMemorySize,
				  cudaStream_t stream,
								 unsigned int* scores, unsigned int* scores_index, unsigned int input_size, unsigned int bound){


	unsigned int out_size = input_size;
	while(out_size > 1){

		argMaxWithEarlyStop<<<ceilf((float)dimGrid/2), dimBlock, sharedMemorySize, stream>>>(scores, scores_index, out_size, bound);
		out_size = ceilf((float)dimGrid/2);
		dimGrid = ceil((float)out_size/(float)dimBlock);
	}

};

void argMaxKernelWidthUpperBound(unsigned int dimGrid, unsigned int dimBlock, unsigned int sharedMemorySize,
				  cudaStream_t stream,
								 float* scores, unsigned int* scores_index, unsigned int input_size, unsigned int bound){


	unsigned int out_size = input_size;
	while(out_size > 1){

		argMaxWithEarlyStop<<<ceilf((float)dimGrid/2), dimBlock, sharedMemorySize, stream>>>(scores, scores_index, out_size, bound);
		out_size = ceilf((float)dimGrid/2);
		dimGrid = ceil((float)out_size/(float)dimBlock);
	}

};


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
		argMaxDevice<<<ceilf((float)dimGrid/2), dimBlock, sharedMemSize>>>(scores_d, scores_index_d, out_size);
		out_size = ceilf((float)dimGrid/2);
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



int argMaxBound(std::vector<float>* scores, unsigned int bound){
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
		argMaxWithEarlyStop<<<ceilf((float)dimGrid/2),
			dimBlock, sharedMemSize>>>(
									   scores_d,
									   scores_index_d,
									   out_size,
									   bound);
		out_size = ceilf((float)dimGrid/2);
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
