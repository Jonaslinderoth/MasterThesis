#include "MineClusKernels.h"
#include <assert.h>
#include <iostream>

/*
  Naive kernel for creating the itemSet. 
  
*/
__global__ void createItemSet(float* data, unsigned int dim, unsigned int numberOfPoints, unsigned int centroidId, float width, unsigned int* output){
	unsigned int point = blockIdx.x*blockDim.x+threadIdx.x;
	if(point < numberOfPoints){
		unsigned int numberOfOutputs = ceilf((float)dim/32);
		for(unsigned int i = 0; i < numberOfOutputs; i++){
			unsigned int output_block = 0;
			for(unsigned int j = 0; j < 32; j++){
				if(i == numberOfOutputs-1 && j >= dim%32){
					break;
				}else{
					assert(dim*centroidId+i*32+j < numberOfpoints*dim);
					assert(point*dim+i*32+j < numberOfpoints*dim);
					output_block |= ((abs(data[dim*centroidId+i*32+j] - data[point*dim+i*32+j]) < width) << j);
				}
			}
			assert(numberOfPoints*i+point < numberOfPoints*ceilf(dim/32));
			output[numberOfPoints*i+point] = output_block;
		}	
	}
}


std::vector<unsigned int> createItemSetTester(std::vector<std::vector<float>*>* data, unsigned int centroid, float width){
	uint size = data->size();
	uint dim = data->at(0)->size();
	uint size_of_data = size*dim*sizeof(float);
	size_t size_of_output = size*ceilf((float)dim/32)*sizeof(unsigned int);
	float* data_h;
	unsigned int* output_h;
	cudaMallocHost((void**) &data_h, size_of_data);
	cudaMallocHost((void**) &output_h, size_of_output);
	for(int i = 0; i < size; i++){
		for(int j = 0; j < dim; j++){
			data_h[i*dim+j] = data->at(i)->at(j);
		}
	}
	float* data_d;
	cudaMalloc((void**) &data_d, size_of_data);
	cudaMemcpy(data_d, data_h, size_of_data, cudaMemcpyHostToDevice);
		
	unsigned int* output_d;
	cudaMalloc((void**) &output_d, size_of_output);

	unsigned int dimBlock = 1024;
	unsigned int dimGrid = ceilf((float)size/dimBlock);
	createItemSet<<<dimGrid, dimBlock>>>(data_d, dim, size, centroid, width, output_d);

	
	
	cudaMemcpy(output_h, output_d, size_of_output, cudaMemcpyDeviceToHost);
	std::vector<unsigned int> res;
	for(int i = 0; i < ceilf((float)dim/32)*size;i++){
		res.push_back(output_h[i]);
	}
	return res;
}


