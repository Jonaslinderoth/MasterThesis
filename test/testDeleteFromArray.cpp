#include <gtest/gtest.h>
#include <random>
#include <iostream>
#include "../src/randomCudaScripts/DeleteFromArray.h"


TEST(testDeleteFromArray, SLOW_tryForceDeleteTheBug){
	std::size_t ammoutOfPoints = 1000000;
	std::size_t dimensions = 100;
	//unsigned long deleteEvery = 2;

	float timeUsed = 0.0;
	float timeForThisRun = 0.0;

	for(unsigned long deleteEvery = 2 ; deleteEvery < 10 ; ++deleteEvery)
	{
		cudaStream_t stream;
		checkCudaErrors(cudaStreamCreate(&stream));


		std::size_t size_of_data = ammoutOfPoints*dimensions*sizeof(float);
		std::size_t size_of_delete_array = (ammoutOfPoints+1)*sizeof(bool);

		float* h_data = (float*)malloc(size_of_data);
		float* h_out_data = (float*)malloc(size_of_data);
		bool* h_delete_array = (bool*)malloc(size_of_delete_array);
		float* d_data;
		float* d_out_data;
		bool* d_delete_array;
		cudaMalloc((void **) &d_data, size_of_data);
		cudaMalloc((void **) &d_out_data, size_of_data);
		cudaMalloc((void **) &d_delete_array, size_of_delete_array);

		//every entry is its identity
		for(std::size_t i = 0 ; i < ammoutOfPoints*dimensions ; ++i){
			h_data[i] = i;
		}
		//this set the output
		for(std::size_t i = 0 ; i < ammoutOfPoints*dimensions ; ++i){
			h_out_data[i] = 0.0;
		}
		//t f t f t f ... <- that is the bool array
		for(std::size_t i = 0 ; i < ammoutOfPoints+1 ; ++i){
			if(i%deleteEvery==0){
				h_delete_array[i] = true;
			}else{
				h_delete_array[i] = false;
			}
		}



		cudaMemcpy(d_data, h_data, size_of_data, cudaMemcpyHostToDevice);
		cudaMemcpy(d_out_data, h_out_data, size_of_data, cudaMemcpyHostToDevice);
		cudaMemcpy(d_delete_array, h_delete_array, size_of_delete_array, cudaMemcpyHostToDevice);

		//deleteFromArrayOld(stream,d_out_data,d_delete_array,d_data,ammoutOfPoints,dimensions,true,&timeForThisRun);
		deleteFromArray(stream,d_out_data,d_delete_array,d_data,ammoutOfPoints,dimensions,true,&timeForThisRun);

		timeUsed += timeForThisRun;
		cudaMemcpy(h_out_data,d_out_data, size_of_data, cudaMemcpyDeviceToHost);

		for(unsigned long i = 0 ; i < ammoutOfPoints/deleteEvery ; ++i){
			unsigned long indexData = i*dimensions;
			for(int indexDim = 0 ; indexDim < dimensions ; ++indexDim){
				EXPECT_EQ(h_out_data[indexData+indexDim],h_data[deleteEvery*indexData+indexDim]);
				/*
				if(h_out_data[indexData+indexDim] != h_data[deleteEvery*indexData+indexDim]){
					std::cout << "not equal at indexData: " << indexData << " indexDim : " << indexDim << " in is: " << h_data[deleteEvery*indexData+indexDim] << " out is: " << h_out_data[indexData+indexDim] << std::endl;
				}*/
			}
		}
		/*
		//this to print the different arrays
		std::cout << "delete array" << std::endl;
		for(std::size_t i = 0 ; i < ammoutOfPoints ; ++i){
			std::cout << h_delete_array[i] << std::endl;
		}
		std::cout << "data" << std::endl;
		for(std::size_t i = 0 ; i < ammoutOfPoints*dimensions ; ++i){
			std::cout << h_data[i] << std::endl;
		}
		std::cout << "output" << std::endl;
		for(std::size_t i = 0 ; i < ammoutOfPoints*dimensions ; ++i){

			std::cout << h_out_data[i] << std::endl;
		}*/

		free(h_data);
		free(h_out_data);
		free(h_delete_array);
		cudaFree(d_data);
		cudaFree(d_out_data);
		cudaFree(d_delete_array);

	}

}


TEST(testDeleteFromArray, SLOW_testTransform){
	std::size_t ammoutOfPoints = 10000;
	std::size_t dimensions = 200;
	//unsigned long deleteEvery = 3;

	float timeUsed = 0.0;
	float timeForThisRun = 0.0;

	for(unsigned long deleteEvery = 2 ; deleteEvery < 100 ; ++deleteEvery)
	{
		cudaStream_t stream;
		checkCudaErrors(cudaStreamCreate(&stream));


		std::size_t size_of_data = ammoutOfPoints*dimensions*sizeof(float);
		std::size_t size_of_delete_array = (ammoutOfPoints+1)*sizeof(bool);

		float* h_data = (float*)malloc(size_of_data);
		float* h_out_data = (float*)malloc(size_of_data);
		bool* h_delete_array = (bool*)malloc(size_of_delete_array);
		float* d_data;
		float* d_out_data;
		bool* d_delete_array;
		cudaMalloc((void **) &d_data, size_of_data);
		cudaMalloc((void **) &d_out_data, size_of_data);
		cudaMalloc((void **) &d_delete_array, size_of_delete_array);

		//every entry is its identity
		for(std::size_t i = 0 ; i < ammoutOfPoints ; ++i){
			for(std::size_t k = 0 ; k < dimensions ; ++ k){
				h_data[k*ammoutOfPoints+i] = i*dimensions+k;
			}

		}
		//this set the output
		for(std::size_t i = 0 ; i < ammoutOfPoints*dimensions ; ++i){
			h_out_data[i] = 0.0;
		}
		//t f t f t f ... <- that is the bool array
		for(std::size_t i = 0 ; i < ammoutOfPoints+1 ; ++i){
			if(i%deleteEvery==0){
				h_delete_array[i] = true;
			}else{
				h_delete_array[i] = false;
			}
		}



		cudaMemcpy(d_data, h_data, size_of_data, cudaMemcpyHostToDevice);
		cudaMemcpy(d_out_data, h_out_data, size_of_data, cudaMemcpyHostToDevice);
		cudaMemcpy(d_delete_array, h_delete_array, size_of_delete_array, cudaMemcpyHostToDevice);

		//deleteFromArrayOld(stream,d_out_data,d_delete_array,d_data,ammoutOfPoints,dimensions,true,&timeForThisRun);
		deleteFromArrayTrasformedData(stream,d_out_data,d_delete_array,d_data,ammoutOfPoints,dimensions,true,&timeForThisRun);

		timeUsed += timeForThisRun;
		cudaMemcpy(h_out_data,d_out_data, size_of_data, cudaMemcpyDeviceToHost);
		unsigned foo = ammoutOfPoints/deleteEvery;
		if(ammoutOfPoints%deleteEvery!=0){
			foo++;
		}
		for(unsigned long i = 0 ; i < foo ; ++i){
			for(int indexDim = 0 ; indexDim < dimensions ; ++indexDim){
				const unsigned int newIndex = indexDim*foo+i;
				const unsigned int oldIndex = indexDim*ammoutOfPoints+deleteEvery*i;
				EXPECT_EQ(h_out_data[newIndex] , h_data[oldIndex]);
				/*
				if(h_out_data[newIndex] != h_data[oldIndex]){
					std::cout << h_out_data[newIndex] << " != " << h_data[oldIndex] << std::endl;
				}*/
			}
		}
		/*
		//this to print the different arrays
		std::cout << "delete array" << std::endl;
		for(std::size_t i = 0 ; i < ammoutOfPoints ; ++i){
			std::cout << h_delete_array[i] << std::endl;
		}
		std::cout << "data" << std::endl;
		for(std::size_t i = 0 ; i < ammoutOfPoints*dimensions ; ++i){
			std::cout << h_data[i] << std::endl;
		}
		std::cout << "output" << std::endl;
		for(std::size_t i = 0 ; i < ammoutOfPoints*dimensions ; ++i){

			std::cout << h_out_data[i] << std::endl;
		}*/

		free(h_data);
		free(h_out_data);
		free(h_delete_array);
		cudaFree(d_data);
		cudaFree(d_out_data);
		cudaFree(d_delete_array);

	}

}





 TEST(testDeleteFromArray, oneDim){
	 cudaStream_t stream;
	 checkCudaErrors(cudaStreamCreate(&stream));

	 bool* toBeDeleted_d;
	 unsigned int* prefixSum_d;
	 unsigned int* candidates_d;
	 unsigned int* candidatesOut_d;

	 bool* toBeDeleted_h;
	 unsigned int* prefixSum_h;
	 unsigned int* candidates_h;
	 unsigned int* candidatesOut_h;

	 unsigned int numberOfPoints = 100;
	 checkCudaErrors(cudaMalloc((void**) &toBeDeleted_d, (numberOfPoints+1)*sizeof(bool)));
	 checkCudaErrors(cudaMalloc((void**) &prefixSum_d, (numberOfPoints+1)*sizeof(unsigned int)));
	 checkCudaErrors(cudaMalloc((void**) &candidates_d, numberOfPoints*sizeof(unsigned int)));
	 checkCudaErrors(cudaMalloc((void**) &candidatesOut_d, 2*sizeof(unsigned int)));
	 
	 checkCudaErrors(cudaMallocHost((void**) &toBeDeleted_h, (numberOfPoints+1)*sizeof(bool)));
	 checkCudaErrors(cudaMallocHost((void**) &prefixSum_h, (numberOfPoints+1)*sizeof(unsigned int)));
	 checkCudaErrors(cudaMallocHost((void**) &candidates_h, numberOfPoints*sizeof(unsigned int)));
	 checkCudaErrors(cudaMallocHost((void**) &candidatesOut_h, 2*sizeof(unsigned int)));
	 
	 int i = 0;
	 for(; i < 10; i++){
		 toBeDeleted_h[i] = true;
	 }
	 toBeDeleted_h[10] = false;
	 i++;
	 for(; i < 20; i++){
		 toBeDeleted_h[i] = true;
	 }
	 toBeDeleted_h[20] = false;
	 i++;
	 for(; i < numberOfPoints; i++){
		 toBeDeleted_h[i] = true;
	 }
	 

	 for(int j = 0; j < numberOfPoints; j++){
		 candidates_h[j] = j;
	 }
	 
	 
	 checkCudaErrors(cudaMemcpyAsync(toBeDeleted_d, toBeDeleted_h, (numberOfPoints+1)*sizeof(bool), cudaMemcpyHostToDevice,stream));
	 checkCudaErrors(cudaMemcpyAsync(candidates_d, candidates_h, (numberOfPoints)*sizeof(unsigned int), cudaMemcpyHostToDevice,stream));	 
	 
	 sum_scan_blelloch(stream, prefixSum_d,toBeDeleted_d,(numberOfPoints+1), false);

	 
	 checkCudaErrors(cudaMemcpyAsync(prefixSum_h, prefixSum_d, (numberOfPoints+1)*sizeof(unsigned int), cudaMemcpyDeviceToHost,stream));
	 checkCudaErrors(cudaStreamSynchronize(stream));

	 int count = 0;
	 i = 0;
	 for(; i < 10; i++){
		 EXPECT_EQ(prefixSum_h[i], count);
		 count++;
	 }
	 i++;
	 EXPECT_EQ(prefixSum_h[i], count);
	 for(; i < 20; i++){
		 EXPECT_EQ(prefixSum_h[i], count);
		 count++;
	 }
	 i++;
	 EXPECT_EQ(prefixSum_h[i], count);
	 for(; i < numberOfPoints; i++){
		 EXPECT_EQ(prefixSum_h[i], count);
		 count++;
	 }
	 deleteFromArrayTransfomedDataWrapper(ceilf((float)numberOfPoints/1024), 1024, stream,
													 candidates_d, prefixSum_d, numberOfPoints, 1, candidatesOut_d);


	 
	 checkCudaErrors(cudaMemcpyAsync(candidatesOut_h, candidatesOut_d, (2)*sizeof(unsigned int), cudaMemcpyDeviceToHost,stream));

	 checkCudaErrors(cudaStreamSynchronize(stream));


	 EXPECT_EQ(candidatesOut_h[0], 10);
	 EXPECT_EQ(candidatesOut_h[1], 20);

 }
