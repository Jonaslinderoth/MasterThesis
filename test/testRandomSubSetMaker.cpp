#include <gtest/gtest.h>
#include <iostream>
#include <cuda.h>
#include "../src/DOC_GPU/DOCGPU_Kernels.h"
#include "../src/randomCudaScripts/DeleteFromArray.h"


TEST(testRandomSubSetMaker, testOne){
	bool print = false;
	//test size
	const unsigned int size = 1000;
	//allocate memory
	unsigned int* d_randomIndexs;
	cudaMalloc(&d_randomIndexs, sizeof(unsigned int) * size);
	unsigned int* h_randomIndexs = new unsigned int[size];
	curandState* d_randomStates;
	cudaMalloc((void**)&d_randomStates, sizeof(curandState) * size);

	//generate the states
	bool res1 = generateRandomStatesArray(d_randomStates,size);

	//generate the random numbers
	bool res2 = generateRandomIntArrayDevice(d_randomIndexs, d_randomStates , size);

	cudaMemcpy(h_randomIndexs, d_randomIndexs, sizeof(unsigned int) * size, cudaMemcpyDeviceToHost);

	if(print){
		std::cout << "output: ";
		for(int i = 0 ; i < size ; ++i){
			std::cout << h_randomIndexs[i] << " ";
		}
	}
	unsigned int maxChecked = 10;
	if(size<maxChecked){
		maxChecked = size;
	}
	for(int i = 0 ; i < maxChecked ; ++i){
		EXPECT_NE(h_randomIndexs[i],h_randomIndexs[i+1]);
	}
	std::cout << std::endl;
	SUCCEED();
}



TEST(testRandomSubSetMaker, testTwo){
	bool print = false;
	//test size
	const unsigned int size = 1000;
	//allocate memory
	unsigned int* d_randomIndexs;
	cudaMalloc(&d_randomIndexs, sizeof(unsigned int) * size);
	unsigned int* h_randomIndexs = new unsigned int[size];
	curandState* d_randomStates;
	cudaMalloc((void**)&d_randomStates, sizeof(curandState) * size);

	//generate the states
	bool res1 = generateRandomStatesArray(d_randomStates,size);

	//generate the random numbers
	bool res = generateRandomIntArrayDevice(d_randomIndexs, d_randomStates , size , 100 , 50);

	cudaMemcpy(h_randomIndexs, d_randomIndexs, sizeof(unsigned int) * size, cudaMemcpyDeviceToHost);

	if(print){
		std::cout << "output: ";
		for(int i = 0 ; i < size ; ++i){
			std::cout << h_randomIndexs[i] << " ";
		}
		std::cout << std::endl;
	}
	for(int i = 0 ; i < size ; ++i){
		ASSERT_LT(h_randomIndexs[i],100.1);
		ASSERT_LT(49.99,h_randomIndexs[i]);
	}


}


