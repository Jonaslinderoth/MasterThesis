/*
 * DOCGPU.cpp
 *
 *  Created on: Feb 14, 2020
 *      Author: mikkel
 */

#include "DOCGPU.h"
#include "DOCGPU_Kernels.h"

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
}

unsigned int* DOCGPU::cudaRandomNumberArray(const size_t lenght ,const curandGenerator_t* gen, unsigned int* array) {
	//if the array was not given
	if(array == nullptr){
		/*allocate that array.*/
		cudaMalloc(( void **) & array , lenght * sizeof ( unsigned int )) ;
	}

	/* Generate n floats on device */
	curandGenerate( *gen , array , lenght);
	return array;
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


std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> DOCGPU::findCluster(){
	auto data = this->data;
	auto alpha = this->alpha;
	auto beta = this->beta;
	auto width = this->width;
	
	unsigned int d = data->at(0)->size();
	unsigned int r = log2(2*d)/log2(1/(2*beta));
	unsigned int m = pow((2/alpha),r) * log(4);
	
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

	
	std::vector<int> id = randInt(0,number_of_points-1, number_of_ps);

	for(int i = 0; i < number_of_ps; i++){
		for(int j = 0; j < point_dim; j++){
			ps_h[i*point_dim+j] = data->at(id.at(i))->at(j);
		}
	}
	id= randInt(0,number_of_points-1, number_of_samples*sample_size);
	for(int i = 0; i < number_of_samples*sample_size; i++){
		for(int j = 0; j < point_dim; j++){
			xs_h[i*point_dim+j] = data->at(id.at(i))->at(j);
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


	free(data_h);
	free(ps_h);
	free(xs_h);


	
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

	
	
	findDimmensionsKernel(dimGrid, dimBlock, xs_d, ps_d, findDim_output_d,
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
	//unsigned int* findDim_count_h = (unsigned int*) malloc(size_of_findDim_count);



	std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> result;

	// if the score is zero the cluster must be empty
	if(fabs(scores_out_h[0]) < 0.0001){
		result = make_pair(new std::vector<std::vector<float>*>, new std::vector<bool>);
		
	}else{
		cudaMemcpy(output_dims_h, findDim_output_d+(scores_index_h[0]*point_dim), size_of_output_dims, cudaMemcpyDeviceToHost);
		cudaMemcpy(output_cluster_h, pointsContained_output_d+(scores_index_h[0]*number_of_points), size_of_output_cluster, cudaMemcpyDeviceToHost);


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
	
};


