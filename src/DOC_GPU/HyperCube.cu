#include "HyperCube.h"
#include <iostream>
#include <vector>
#include <math.h>

__global__ void findDimmensionsDevice(float* Xs_d, float* ps_d, bool* res_d,
									  int point_dim, int no_of_samples, int no_in_sample, int no_of_ps, float width){
	int entry = blockIdx.x*blockDim.x+threadIdx.x;
	int pNo = entry/no_of_samples;
	int sampleNo = entry%no_of_samples;
	
	if(entry < no_of_samples*no_of_ps){
		for(int i = 0; i < point_dim; i++){
			bool d = true;
			float p_tmp = ps_d[pNo*point_dim+i];
			for(int j = 0; j < no_in_sample; j++){
				d &= abs(p_tmp-Xs_d[sampleNo*no_in_sample*point_dim+j*point_dim+i]) < width;
			}
			res_d[entry*point_dim+i] = d;
		}
	}
}

__global__ void pointsContainedDevice(float* data, float* centroids, bool* dims, bool* output,
									  float width, int point_dim, int no_data, int no_dims){
	// one kernel for each hypercube
	int entry = blockIdx.x*blockDim.x+threadIdx.x;
	if(entry < no_dims){
		// for each data point 
		for(int j = 0; j < no_data; j++){
			// for all dimmensions in each hypercube / point
			bool d = true;
			for(int i = 0; i < point_dim; i++){
				//(not (dims[entry*point_dim+i])) ||
				d &= (not (dims[entry*point_dim+i])) || (abs(centroids[entry*point_dim+i] - data[j*point_dim+i]) < width);
			}
			output[entry*no_data+j] = d;
		}
	}
}

__global__ void score(float* Cluster_size, float* Dim_count, float* score_output, int len, float beta){
	int entry = blockIdx.x*blockDim.x+threadIdx.x;
	if(entry < len){
		score_output[entry] = Cluster_size[entry]*powf(1.0/beta, Dim_count[entry]);	
	}

}

__global__ void argMaxDevice(float* scores, int* scores_index, float* output ,int* output_index, int input_size){
	extern __shared__ int array[];
	int* argData = (int*)array;
	float* scoreData = (float*) &argData[blockDim.x];

	
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
	argData[tid] = scores_index[i];
	scoreData[tid] = scores[i];
	
	__syncthreads();

	if(i < input_size){
		
		for(unsigned int s=blockDim.x; s > 0; s/=2) {

			if(tid < s){
				if(scoreData[tid] < scoreData[tid+s]){
					scoreData[tid] = scoreData[tid+s];
					argData[tid] = argData[tid+s];
				}
			}
			__syncthreads();
		}
	
		if(tid == 0){
			output_index[blockIdx.x] = argData[0];
			output[blockIdx.x] = scoreData[0];
		}; 
		
	}
	
}







std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> findCluster(std::vector<std::vector<float>*>* data, float alpha, float beta, float width){
	float d = data->at(0)->size();
	float r = log2(2*d)/log2(1/(2*beta));
	float m = pow((2/alpha),2) * log(4);

	unsigned int number_of_ps = 2.0/alpha;
	unsigned int number_of_samples = number_of_ps*m;
	unsigned int sample_size = r;
	unsigned int number_of_points = data->size();
	unsigned int point_dim = d;
	unsigned int floats_in_data_array = point_dim*number_of_points;
	unsigned int floats_in_ps_array = point_dim*number_of_ps; 
	
};






std::vector<std::vector<bool>*>* pointsContained(std::vector<std::vector<bool>*>* dims,
												 std::vector<std::vector<float>*>* data,
												 std::vector<std::vector<float>*>* centroids, float width){

	// Calculaating sizes
	int point_dim = centroids->at(0)->size();
	int no_of_points = data->size();
	int no_of_dims = dims->size();
	int no_of_centroids = centroids->size();

	int floats_in_data = point_dim * no_of_points;
	int bools_in_dims = no_of_dims * point_dim;
	int floats_in_centorids = no_of_centroids * point_dim;
	int bools_in_output = no_of_points * no_of_dims;
	
	int size_of_data = floats_in_data*sizeof(float);
	int size_of_dims = bools_in_dims*sizeof(bool);
	int size_of_centroids = floats_in_centorids*sizeof(float);
	int size_of_output = bools_in_output*sizeof(bool);

	// allocating on the host
	float* data_h = (float*) malloc(size_of_data);
	bool* dims_h = (bool*) malloc(size_of_dims);
	float* centroids_h = (float*) malloc(size_of_centroids);
	bool* output_h = (bool*) malloc(size_of_output);

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
		for(int j = 0; j < point_dim; j++){
			centroids_h[i*point_dim+j] = centroids->at(i)->at(j);
		}
	}

	// allocating on device
	float* data_d;
	bool* dims_d;
	float* centroids_d;
	bool* output_d;
	
	cudaMalloc((void **) &data_d, size_of_data);
	cudaMalloc((void **) &dims_d, size_of_dims);
	cudaMalloc((void **) &centroids_d, size_of_centroids);
	cudaMalloc((void **) &output_d, size_of_output);

	//Copy from host to device
				
	cudaMemcpy(data_d, data_h, size_of_data, cudaMemcpyHostToDevice);
	cudaMemcpy(dims_d, dims_h, size_of_dims, cudaMemcpyHostToDevice);
	cudaMemcpy(centroids_d, centroids_h, size_of_centroids, cudaMemcpyHostToDevice);


	// Call kernel
	pointsContainedDevice<<<ceil((no_of_dims)/256.0), 256>>>(data_d, centroids_d, dims_d, output_d,
						  width, point_dim, no_of_points, no_of_dims);

	// copy from device
	cudaMemcpy(output_h, output_d, size_of_output, cudaMemcpyDeviceToHost);

	// construnct output
	auto output =  new std::vector<std::vector<bool>*>;
   	
	
	for(int i = 0; i < no_of_dims; i++){
		auto a =  new std::vector<bool>;
		for(int j = 0; j < no_of_points; j++){
			a->push_back(output_h[i*no_of_points+j]);
		}
		output->push_back(a);
	}


	cudaFree(data_d);
	cudaFree(dims_d);
	cudaFree(centroids_d);
	cudaFree(output_d);

	
	return output;
};


std::vector<std::vector<bool>*>* findDimmensions(std::vector<std::vector<float>*>* ps,
												 std::vector<std::vector<std::vector<float>*>*> Xs, float width){
	
	int point_dim = Xs.at(0)->at(0)->size();
	int no_in_sample = Xs.at(0)->size();
	int no_of_samples = Xs.size();

	
	int sizeOfXs = no_of_samples*no_in_sample*point_dim*sizeof(float);
	float* xs_h = (float*) malloc(sizeOfXs);
	for(int i = 0; i < no_of_samples; i++){
		for(int j = 0; j < no_in_sample; j++){
			for(int k = 0; k < point_dim; k++){
				xs_h[i*no_in_sample*point_dim+j*point_dim+k] = Xs.at(i)->at(j)->at(k);
			}
		}
	}

	int no_of_ps = ps->size();
	int sizeOfps = point_dim*no_of_ps*sizeof(float);
	float* ps_h = (float*) malloc(sizeOfps);
	for(int i = 0; i < no_of_ps; i++){
		for(int j = 0; j < point_dim; j++){
			ps_h[i*point_dim+j] = ps->at(i)->at(j);
		}
	}
	/*
	std::cout << "xs: " << std::endl;
	for(int i = 0; i < no_of_samples*no_in_sample*point_dim; i++){
		std::cout << xs_h[i] << ", ";
		if((i+1)% point_dim == 0){
			std::cout << std::endl;
		}
	}
	std::cout << std::endl;
	

	std::cout << "ps: " << std::endl;

	for(int i = 0; i < no_of_ps*point_dim; i++){
		std::cout << ps_h[i] << ", ";
	}
	std::cout << std::endl;
	*/
	int outputDim = no_of_ps*no_of_samples*point_dim;		
	int outputSize = outputDim*sizeof(bool);
	bool* result_h = (bool*) malloc(outputSize);


	float* Xs_d;
	float* ps_d;
	bool* result_d;
	
	cudaMalloc((void **) &Xs_d, sizeOfXs);
	cudaMalloc((void **) &ps_d, sizeOfps);
	cudaMalloc((void **) &result_d, outputSize);

	cudaMemcpy( Xs_d, xs_h, sizeOfXs, cudaMemcpyHostToDevice);
    cudaMemcpy( ps_d, ps_h, sizeOfps, cudaMemcpyHostToDevice);

	findDimmensionsDevice<<<ceil((no_of_ps*no_of_samples)/256.0), 256>>>(Xs_d, ps_d, result_d, point_dim, no_of_samples, no_in_sample, no_of_ps, width);

   
	cudaMemcpy(result_h, result_d, outputSize, cudaMemcpyDeviceToHost);

	auto output =  new std::vector<std::vector<bool>*>;
	
	for(int i = 0; i < no_of_ps*no_of_samples; i++){
		auto a =  new std::vector<bool>;
		for(int j = 0; j < point_dim; j++){
			a->push_back(result_h[i*point_dim+j]);
		}
		output->push_back(a);
	}

	cudaFree(Xs_d);
	cudaFree(ps_d);
	cudaFree(result_d);
	
	return output;
}



int argMax(std::vector<float>* scores){
	//Calculate size of shared Memory, block and thread dim
	//fetch device info
	// TODO: hardcoded device 0
	int smemSize, maxBlock;
	cudaDeviceGetAttribute(&smemSize, 
    cudaDevAttrMaxSharedMemoryPerBlock, 0);
	cudaDeviceGetAttribute(&maxBlock, 
						   cudaDevAttrMaxThreadsPerBlock, 0); // TODO::: not working::....

	maxBlock = 64;//512; // TODO::: findout why values larger than 64 wont work for all samples larger than 64...
	//std::cout << smemSize << std::endl;

	// caluclate the maximum thread size based on shared mem requirements and maximum threads
	int dimBlock = smemSize/(sizeof(int)+sizeof(float));
	if(dimBlock > maxBlock) dimBlock = maxBlock;
	int dimGrid = ceil((float)scores->size()/(float)dimBlock);
	int sharedMemSize = ((1+dimBlock)*sizeof(int) + (1+dimBlock)*sizeof(float));


	int size_of_score = scores->size()*sizeof(float);
	int size_of_score_index = scores->size()*sizeof(int);
	int size_of_output = sizeof(float)*dimGrid;
	int size_of_output_index = sizeof(int)*dimGrid;
	
	float* scores_h = (float*) malloc(size_of_score);
	int* scores_index_h = (int*) malloc(size_of_score_index);
	float* output_h = (float*) malloc(size_of_output);
	int* output_index_h = (int*) malloc(size_of_output_index);

	//std::cout << "creating data..." << scores->size() << std::endl;
	for(int i = 0; i < scores->size(); i++){
		scores_h[i] = scores->at(i);
		scores_index_h[i] = i;
	}

	//std::cout << "data created" << std::endl;
	float* scores_d;
	int* scores_index_d;
	float* output_d;
	int* output_index_d;

	cudaMalloc((void **) &scores_d, size_of_score);
	cudaMalloc((void **) &scores_index_d, size_of_score_index);
	cudaMalloc((void **) &output_d, size_of_output);
	cudaMalloc((void **) &output_index_d, size_of_output_index);
	//std::cout << "cuda malloc" << std::endl;
	
	cudaMemcpy(scores_d, scores_h, size_of_score, cudaMemcpyHostToDevice);
	cudaMemcpy(scores_index_d, scores_index_h, size_of_score, cudaMemcpyHostToDevice);
	//std::cout << "data copied" << std::endl;

	//std::cout << dimBlock << ", " << dimGrid << ", " << sharedMemSize << std::endl;

	
	//Call kernel
	int out_size = scores->size();
	argMaxDevice<<<dimGrid, dimBlock, sharedMemSize>>>(scores_d, scores_index_d, output_d, output_index_d, out_size);

	/*
	cudaMemcpy(output_h, output_d, size_of_output, cudaMemcpyDeviceToHost);
	cudaMemcpy(output_index_h, output_index_d, size_of_output_index, cudaMemcpyDeviceToHost);	
	for(int i=0; i < dimGrid; i++){
		std::cout << "max value: " << output_h[i] << ", maxIndex: " << output_index_h[i] << std::endl;
	}
	*/
	//std::cout << "first pass" << std::endl;
	//int i = 1;
	while(out_size > 1){
		//std::cout << i << "th pass" << std::endl;
		//i++;
		auto temp = output_d;
		auto temp_index = output_index_d;
		output_index_d = scores_index_d;
		output_d = scores_d;
		scores_index_d = temp_index;
		scores_d = temp;
		
		out_size = dimGrid;
		dimGrid = ceil((float)out_size/(float)dimBlock);
		//std::cout << dimBlock << ", " << dimGrid << ", " << sharedMemSize << std::endl;
		argMaxDevice<<<dimGrid, dimBlock, sharedMemSize>>>(scores_d, scores_index_d, output_d, output_index_d, out_size);		
		}

	cudaMemcpy(output_h, output_d, size_of_output, cudaMemcpyDeviceToHost);
	cudaMemcpy(output_index_h, output_index_d, size_of_output_index, cudaMemcpyDeviceToHost);	
	
	

	cudaFree(scores_d);
	cudaFree(scores_index_d);
	cudaFree(output_index_d);	
	cudaFree(output_d);

	return output_index_h[0] ;
	
	
}