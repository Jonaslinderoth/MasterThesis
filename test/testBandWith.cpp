#include <gtest/gtest.h>
#include "../src/randomCudaScripts/bandwithTesting.h"
#include <cuda.h>
#include <iostream>
#include "../src/DOC_GPU/DOCGPU_Kernels.h"
#include "../src/randomCudaScripts/DeleteFromArray.h"

/*
TEST(testBandWith, DISABLE_thisShoudNotRunAsATest){
	size_t dataSize = 250000000;

	const size_t size_of_data =  dataSize*sizeof(unsigned int);
	unsigned int* h_data = (unsigned int*)malloc(size_of_data);
	unsigned int* d_data;
	cudaMalloc((void **) &d_data, size_of_data);
	//set all entries to non zero
	for(std::size_t i = 0; i < dataSize ; ++i){
		h_data[i] = 1;
	}

	cudaMemcpy(d_data, h_data, size_of_data, cudaMemcpyHostToDevice);

	testBandWith(d_data,dataSize,true);
	EXPECT_TRUE(true);
}*/
