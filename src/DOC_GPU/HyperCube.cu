#include "HyperCube.h"
#include <iostream>
#include <vector>
#include <math.h>
#include <random>

__global__ void findDimmensionsDevice(float* Xs_d, float* ps_d, bool* res_d, unsigned int* Dsum_out,
									  int point_dim, int no_of_samples, int no_in_sample, int no_of_ps, int m, float width){
	int entry = blockIdx.x*blockDim.x+threadIdx.x;
	int pNo = entry/m;
	
	if(entry < no_of_samples){
		unsigned int Dsum = 0;
		// for each dimension
		for(int i = 0; i < point_dim; i++){
			bool d = true;
			float p_tmp = ps_d[pNo*point_dim+i];
			// for each point in sample
			for(int j = 0; j < no_in_sample; j++){
				d &= abs(p_tmp-Xs_d[entry*no_in_sample*point_dim+j*point_dim+i]) < width;
			}
			res_d[entry*point_dim+i] = d;
			Dsum += d;

		}
		Dsum_out[entry] = Dsum;
	}
}

__global__ void pointsContainedDevice(float* data, float* centroids, bool* dims, bool* output, unsigned int* Csum_out,
									  float width, int point_dim, int no_data, int no_dims, int m){
	// one kernel for each hypercube
	int entry = blockIdx.x*blockDim.x+threadIdx.x;
	int currentCentroid = entry/m;
	if(entry < no_dims){
		// for each data point
		unsigned int Csum = 0;
		for(int j = 0; j < no_data; j++){
			// for all dimmensions in each hypercube / point
			bool d = true;
			for(int i = 0; i < point_dim; i++){
				//(not (dims[entry*point_dim+i])) ||
				d &= (not (dims[entry*point_dim+i])) || (abs(centroids[currentCentroid*point_dim+i] - data[j*point_dim+i]) < width);
			}
			output[entry*no_data+j] = d;
			Csum += d;
		}
		Csum_out[entry] = Csum;
	}
}

__global__ void score(unsigned int* Cluster_size, unsigned int* Dim_count, float* score_output, int len, float alpha, float beta, unsigned int num_points){
	int entry = blockIdx.x*blockDim.x+threadIdx.x;
	if(entry < len){
		score_output[entry] = ((Cluster_size[entry])* powf(1.0/beta, (Dim_count[entry])))*(Dim_count[entry] >= (alpha*num_points));	
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

__global__ void argMaxDevice(float* scores, unsigned int* scores_index, int input_size){
	extern __shared__ int array[];
	int* argData = (int*)array;
	float* scoreData = (float*) &argData[blockDim.x];

	
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
	argData[tid] = scores_index[i];
	scoreData[tid] = scores[i];
	
	__syncthreads();

	if(i < input_size){
		
		for(unsigned int s=(blockDim.x/2); s > 0; s/=2) {

			if(tid < s){
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
	unsigned int floats_in_Xs_array = point_dim*number_of_samples*sample_size;

	unsigned int size_of_data = floats_in_data_array*sizeof(float);
	unsigned int size_of_ps = floats_in_ps_array*sizeof(float);
	unsigned int size_of_xs = floats_in_Xs_array*sizeof(float);
	unsigned int size_of_output_dims = point_dim*sizeof(bool);
	unsigned int size_of_output_cluster = number_of_points*sizeof(bool);


	float* data_h = (float*) malloc(size_of_data);
	float* ps_h = (float*) malloc(size_of_ps);
	float* xs_h = (float*) malloc(size_of_xs);
	
	for(int i = 0; i < number_of_points; i++){
		for(int j = 0; j < point_dim; j++){
			data_h[i*point_dim+j] = data->at(i)->at(j);
		}
	}


	std::mt19937 gen;
	gen.seed(1);
	
	std::uniform_int_distribution<> dis(0, number_of_points-1); //inclusive

	for(int i = 0; i < number_of_ps; i++){
		unsigned int id = dis(gen);
		for(int j = 0; j < point_dim; j++){
			ps_h[i*point_dim+j] = data->at(id)->at(j);
		}
	}

	for(int i = 0; i < number_of_samples*sample_size; i++){
		unsigned int id = dis(gen);
		for(int j = 0; j < point_dim; j++){
			xs_h[i*point_dim+j] = data->at(id)->at(j);
		}
	}
	

	float* data_d;
	float* ps_d;
	float* xs_d;
	

	cudaMalloc((void **) &data_d, size_of_data);
	cudaMalloc((void **) &ps_d, size_of_ps);
	cudaMalloc((void **) &xs_d, size_of_xs);

	cudaMemcpy(data_d, data_h, size_of_data, cudaMemcpyHostToDevice);
	cudaMemcpy(xs_d, xs_h, size_of_xs, cudaMemcpyHostToDevice);
	cudaMemcpy(ps_d, ps_h, size_of_ps, cudaMemcpyHostToDevice);


	
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

	
    findDimmensionsDevice<<<dimGrid, dimBlock>>>(xs_d, ps_d, findDim_output_d, findDim_count_d,
												 point_dim, number_of_samples, sample_size,
												 number_of_ps, m, width);
	cudaFree(xs_d);


	pointsContainedDevice<<<dimGrid, dimBlock>>>(data_d, ps_d, findDim_output_d,
												 pointsContained_output_d, pointsContained_count_d,
												 width, point_dim, number_of_points, number_of_samples, m);
	cudaFree(ps_d);
	cudaFree(data_d);
	
	score<<<dimGrid, dimBlock>>>(pointsContained_count_d, findDim_count_d, scores_d, number_of_samples, alpha, beta, number_of_points);

	cudaFree(findDim_count_d);
	cudaFree(pointsContained_count_d);



	createIndices<<<dimGrid, dimBlock>>>(scores_index_d, number_of_samples);	
	unsigned int out_size = number_of_samples;
	while(out_size > 1){
		argMaxDevice<<<dimGrid, dimBlock, sharedMemSize>>>(scores_d, scores_index_d, out_size);				
		out_size = dimGrid;
		dimGrid = ceil((float)out_size/(float)dimBlock);
	}
	
	argMaxDevice<<<dimGrid, dimBlock, sharedMemSize>>>(scores_d, scores_index_d, out_size);


	cudaFree(scores_d);




	unsigned int size_of_output = sizeof(unsigned int);
	unsigned int* scores_index_h = (unsigned int*) malloc(size_of_output);
	cudaMemcpy(scores_index_h, scores_index_d, size_of_output, cudaMemcpyDeviceToHost);


	
	bool* output_dims_h = (bool*) malloc(size_of_output_dims);
	bool* output_cluster_h = (bool*) malloc(size_of_output_cluster);
	float* scores_h = (float*) malloc(size_of_scores);
	unsigned int* findDim_count_h = (unsigned int*) malloc(size_of_findDim_count);



	cudaMemcpy(output_dims_h, findDim_output_d+(scores_index_h[0]*point_dim), size_of_output_dims, cudaMemcpyDeviceToHost);
	cudaMemcpy(output_cluster_h, pointsContained_output_d+(scores_index_h[0]*number_of_points), size_of_output_cluster, cudaMemcpyDeviceToHost);

	cudaFree(scores_index_d);
	cudaFree(findDim_output_d);
	cudaFree(pointsContained_output_d);


	
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
	
	return std::make_pair(outCluster, outDim);
};






std::pair<std::vector<std::vector<bool>*>*,std::vector<unsigned int>*> pointsContained(std::vector<std::vector<bool>*>* dims,
																					   std::vector<std::vector<float>*>* data,
																					   std::vector<std::vector<float>*>* centroids,
																					   int m, float width){

	// Calculaating sizes
	int point_dim = centroids->at(0)->size();
	int no_of_points = data->size();
	int no_of_dims = dims->size();
	int no_of_centroids = centroids->size();

	int floats_in_data = point_dim * no_of_points;
	int bools_in_dims = no_of_dims * point_dim;
	int floats_in_centorids = no_of_centroids * point_dim;
	int bools_in_output = no_of_points * no_of_dims;
	int ints_in_output_count = no_of_dims;
	
	int size_of_data = floats_in_data*sizeof(float);
	int size_of_dims = bools_in_dims*sizeof(bool);
	int size_of_centroids = floats_in_centorids*sizeof(float);
	int size_of_output = bools_in_output*sizeof(bool);
	int size_of_output_count = ints_in_output_count*sizeof(unsigned int);

	// allocating on the host
	float* data_h = (float*) malloc(size_of_data);
	bool* dims_h = (bool*) malloc(size_of_dims);
	float* centroids_h = (float*) malloc(size_of_centroids);
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
		for(int j = 0; j < point_dim; j++){
			centroids_h[i*point_dim+j] = centroids->at(i)->at(j);
		}
	}

	// allocating on device
	float* data_d;
	bool* dims_d;
	float* centroids_d;
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

	
	return std::make_pair(output,output_count);
};


std::pair<std::vector<std::vector<bool>*>*,std::vector<unsigned int>*> findDimmensions(std::vector<std::vector<float>*>* ps,
																					   std::vector<std::vector<std::vector<float>*>*> Xs, int m, float width){
	
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
	unsigned int size_of_count = (no_of_samples)*sizeof(unsigned int);
	
	int outputDim = no_of_samples*point_dim;		
	int outputSize = outputDim*sizeof(bool);
	bool* result_h = (bool*) malloc(outputSize);
	unsigned int* count_h = (unsigned int*) malloc(size_of_count);


	float* Xs_d;
	float* ps_d;
	bool* result_d;
	unsigned int* count_d;
	
	cudaMalloc((void **) &Xs_d, sizeOfXs);
	cudaMalloc((void **) &ps_d, sizeOfps);
	cudaMalloc((void **) &result_d, outputSize);
	cudaMalloc((void **) &count_d, size_of_count);

	cudaMemcpy( Xs_d, xs_h, sizeOfXs, cudaMemcpyHostToDevice);
    cudaMemcpy( ps_d, ps_h, sizeOfps, cudaMemcpyHostToDevice);

	findDimmensionsDevice<<<ceil((no_of_samples)/256.0), 256>>>(Xs_d, ps_d, result_d, count_d, point_dim, no_of_samples, no_in_sample, no_of_ps, m, width);

   
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

	cudaFree(Xs_d);
	cudaFree(ps_d);
	cudaFree(result_d);
	cudaFree(count_d);
	
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
		argMaxDevice<<<dimGrid, dimBlock, sharedMemSize>>>(scores_d, scores_index_d, out_size);				
		out_size = dimGrid;
		dimGrid = ceil((float)out_size/(float)dimBlock);
	}
	
	argMaxDevice<<<dimGrid, dimBlock, sharedMemSize>>>(scores_d, scores_index_d, out_size);		

	unsigned int size_of_output = sizeof(unsigned int);
	
	unsigned int* scores_index_h = (unsigned int*) malloc(size_of_output);

	cudaMemcpy(scores_index_h, scores_index_d, size_of_output, cudaMemcpyDeviceToHost);
	
 
	cudaFree(scores_d);
	cudaFree(scores_index_d);

	return scores_index_h[0];
	
	
}



