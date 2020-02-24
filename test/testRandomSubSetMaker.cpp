#include <gtest/gtest.h>
#include <iostream>
#include <cuda.h>
#include "../src/DOC_GPU/DOCGPU_Kernels.h"
#include "../src/randomCudaScripts/DeleteFromArray.h"


TEST(testRandomSubSetMaker, _SLOW_testOne){
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
	bool res1 = generateRandomStatesArray(d_randomStates,size, false, 10);

	//generate the random numbers
	bool res2 = generateRandomIntArrayDevice(d_randomIndexs, d_randomStates , size, size);

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
	bool res = generateRandomIntArrayDevice(d_randomIndexs, d_randomStates , size,size , 100 , 50);

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


TEST(testRandomSubSetMaker, testMultible){
	unsigned int n = 100;
	unsigned int* ids_d;
	cudaMalloc(&ids_d, sizeof(unsigned int) * n);
	unsigned int* ids_h = (unsigned int*) malloc(sizeof(unsigned int) * n);
	unsigned int* ids2_h = (unsigned int*) malloc(sizeof(unsigned int) * n);
	

	curandState* d_randomStates;
	cudaMalloc((void**)&d_randomStates, sizeof(curandState) * n);

	generateRandomStatesArray(d_randomStates,n,false, 1);
	
	generateRandomIntArrayDevice(ids_d, d_randomStates , n,n , 10000 , 50);
	cudaMemcpy(ids_h, ids_d, sizeof(unsigned int) * n, cudaMemcpyDeviceToHost);
	
	generateRandomIntArrayDevice(ids_d, d_randomStates , n,n , 10000 , 50);
	cudaMemcpy(ids2_h, ids_d, sizeof(unsigned int) * n, cudaMemcpyDeviceToHost);


	for(int i = 0; i < n; i++){
		EXPECT_NE(ids_h[i], ids2_h[i])  << "iteration: " << i;
		EXPECT_NE(ids_h[i], ids_h[(i+1)%n])  << "iteration: " << i;
		EXPECT_NE(ids2_h[i], ids2_h[(i+1)%n])  << "iteration: " << i;
	}
}

TEST(testRandomSubSetMaker, testMultibleEqualSeed){
	unsigned int n = 100;
	unsigned int* ids_d;
	cudaMalloc(&ids_d, sizeof(unsigned int) * n);
	unsigned int* ids_h = (unsigned int*) malloc(sizeof(unsigned int) * n);
	unsigned int* ids2_h = (unsigned int*) malloc(sizeof(unsigned int) * n);
	

	curandState* d_randomStates;
	cudaMalloc((void**)&d_randomStates, sizeof(curandState) * n);

	generateRandomStatesArray(d_randomStates,n,false, 1);

	generateRandomIntArrayDevice(ids_d, d_randomStates , n,n , 1000 , 50);
	cudaMemcpy(ids_h, ids_d, sizeof(unsigned int) * n, cudaMemcpyDeviceToHost);

	curandState* d_randomStates2;
	cudaMalloc((void**)&d_randomStates2, sizeof(curandState) * n);
	generateRandomStatesArray(d_randomStates2,n,false, 1);
	
	generateRandomIntArrayDevice(ids_d, d_randomStates2 , n,n , 1000 , 50);
	cudaMemcpy(ids2_h, ids_d, sizeof(unsigned int) * n, cudaMemcpyDeviceToHost);


	for(int i = 0; i < n; i++){
		EXPECT_EQ(ids_h[i], ids2_h[i]);
	}
}


TEST(testRandomSubSetMaker, testMultibleDifferentSizes){
	unsigned int n = 100;
	unsigned int* ids_d;
	cudaMalloc(&ids_d, sizeof(unsigned int) * n);
	unsigned int* ids_h = (unsigned int*) malloc(sizeof(unsigned int) * n*2);
	unsigned int* ids2_h = (unsigned int*) malloc(sizeof(unsigned int) * n);
	

	curandState* d_randomStates;
	cudaMalloc((void**)&d_randomStates, sizeof(curandState) * n*2);

	generateRandomStatesArray(d_randomStates,n*2, false, 4);
	
	generateRandomIntArrayDevice(ids_d, d_randomStates , n*2, n*2 , 100000 , 50);
	cudaMemcpy(ids_h, ids_d, sizeof(unsigned int) * n, cudaMemcpyDeviceToHost);
	
	generateRandomIntArrayDevice(ids_d, d_randomStates , n*2, n , 100000 , 50);
	cudaMemcpy(ids2_h, ids_d, sizeof(unsigned int) * n, cudaMemcpyDeviceToHost);


	for(int i = 0; i < n; i++){
		EXPECT_NE(ids_h[i], ids2_h[i]);
		EXPECT_NE(ids_h[i], ids_h[(i+1)%n]);
		EXPECT_NE(ids2_h[i], ids2_h[(i+1)%n]);
		//std::cout << ids2_h[i] << ", ";
	}
}

TEST(testRandomSubSetMaker, testMultibleDifferentSizes2){
	unsigned int n = 10000;
	unsigned int n2 = 1000;
	EXPECT_LT(n2,n);
	
	unsigned int* ids_d;
	unsigned int* ids2_d;
	cudaMalloc(&ids_d, sizeof(unsigned int) * n);
	cudaMalloc(&ids2_d, sizeof(unsigned int) * n2);
	
	unsigned int* ids_h = (unsigned int*) malloc(sizeof(unsigned int) * n);
	unsigned int* ids2_h = (unsigned int*) malloc(sizeof(unsigned int) * n2);
	

	curandState* d_randomStates;
	cudaMalloc((void**)&d_randomStates, sizeof(curandState) * n);

	generateRandomStatesArray(d_randomStates,n, false, 4);
	
	
	generateRandomIntArrayDevice(ids_d, d_randomStates, n, n , 100000 , 50);
	generateRandomIntArrayDevice(ids2_d, d_randomStates, n , n2 , 100000 , 50);
	
	cudaMemcpy(ids_h, ids_d, sizeof(unsigned int) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(ids2_h, ids2_d, sizeof(unsigned int) * n2, cudaMemcpyDeviceToHost);


	for(int i = 0; i < n2; i++){
		EXPECT_NE(ids_h[i], ids2_h[i]);
		EXPECT_LE(ids_h[i], 100000);
		EXPECT_GE(ids_h[i], 50);

		EXPECT_LE(ids2_h[i], 100000);
		EXPECT_GE(ids2_h[i], 50);

		
		EXPECT_NE(ids_h[i], ids_h[(i+1)%n2]);
		EXPECT_NE(ids2_h[i], ids2_h[(i+1)%n2]);
		//std::cout << ids2_h[i] << ", ";
	}
}

TEST(testRandomSubSetMaker, testMultibleDifferentSizes3){
	unsigned int n = 10000;
	unsigned int n2 = 1000;
	EXPECT_LT(n2,n);
	
	unsigned int* ids_d;
	unsigned int* ids2_d;
	cudaMalloc(&ids_d, sizeof(unsigned int) * n);
	cudaMalloc(&ids2_d, sizeof(unsigned int) * n2);
	
	unsigned int* ids_h = (unsigned int*) malloc(sizeof(unsigned int) * n);
	unsigned int* ids2_h = (unsigned int*) malloc(sizeof(unsigned int) * n2);
	

	curandState* d_randomStates;
	cudaMalloc((void**)&d_randomStates, sizeof(curandState) * n);

	generateRandomStatesArray(d_randomStates,n, false, 4);
	
	
	generateRandomIntArrayDevice(ids_d, d_randomStates , n, n , 100000 , 50);
	generateRandomIntArrayDevice(ids2_d, d_randomStates , n, n2 , 100000 , 50);
	
	cudaMemcpy(ids_h, ids_d, sizeof(unsigned int) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(ids2_h, ids2_d, sizeof(unsigned int) * n2, cudaMemcpyDeviceToHost);


	for(int i = 0; i < n2; i++){
		EXPECT_NE(ids_h[i], ids2_h[i]);
		EXPECT_LE(ids_h[i], 100000);
		EXPECT_GE(ids_h[i], 50);

		EXPECT_LE(ids2_h[i], 100000);
		EXPECT_GE(ids2_h[i], 50);

		
		EXPECT_NE(ids_h[i], ids_h[(i+1)%n2]);
		EXPECT_NE(ids2_h[i], ids2_h[(i+1)%n2]);
		//std::cout << ids2_h[i] << ", ";
	}
}



TEST(testRandomSubSetMaker, testMultibleDifferentSizes4){
	unsigned int n = 10000;
	unsigned int n2 = 1000;
	EXPECT_LT(n2,n);
	
	unsigned int* ids_d;
	unsigned int* ids2_d;
	cudaMalloc(&ids_d, sizeof(unsigned int) * n);
	cudaMalloc(&ids2_d, sizeof(unsigned int) * n2);
	
	unsigned int* ids_h = (unsigned int*) malloc(sizeof(unsigned int) * n);
	unsigned int* ids2_h = (unsigned int*) malloc(sizeof(unsigned int) * n2);
	

	curandState* d_randomStates;
	cudaMalloc((void**)&d_randomStates, sizeof(curandState) * n2);

	generateRandomStatesArray(d_randomStates,n2, false, 4);
	
	
	generateRandomIntArrayDevice(ids_d, d_randomStates , n2, n , 100000 , 50);
	generateRandomIntArrayDevice(ids2_d, d_randomStates , n2, n2 , 100000 , 50);
	
	cudaMemcpy(ids_h, ids_d, sizeof(unsigned int) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(ids2_h, ids2_d, sizeof(unsigned int) * n2, cudaMemcpyDeviceToHost);


	for(int i = 0; i < n2; i++){
		EXPECT_NE(ids_h[i], ids2_h[i]);
		EXPECT_LE(ids_h[i], 100000);
		EXPECT_GE(ids_h[i], 50);

		EXPECT_LE(ids2_h[i], 100000);
		EXPECT_GE(ids2_h[i], 50);

		
		EXPECT_NE(ids_h[i], ids_h[(i+1)%n2]);
		EXPECT_NE(ids2_h[i], ids2_h[(i+1)%n2]);
		//std::cout << ids2_h[i] << ", ";
	}
	for(int i = n2; i < n; i++){
		ASSERT_LE(ids_h[i], 100000);
		ASSERT_GE(ids_h[i], 50) << "index: " << i ;
		ASSERT_NE(ids_h[i], ids_h[(i+1)%n]);
	}
}
