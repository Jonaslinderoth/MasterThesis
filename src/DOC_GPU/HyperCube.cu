#include "HyperCube.h"
#include <iostream>
#include <vector>

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
	
	
	return output;
}
