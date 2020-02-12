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

std::vector<std::vector<bool>*>* findDimmensions(std::vector<std::vector<float>*>* ps,
												 std::vector<std::vector<std::vector<float>*>*> Xs){
	
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

	findDimmensionsDevice<<<ceil((no_of_ps*no_of_samples)/256.0), 256>>>(Xs_d, ps_d, result_d, point_dim, no_of_samples, no_in_sample, no_of_ps, 10);

   
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
