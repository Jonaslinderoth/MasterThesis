#include <gtest/gtest.h>
#include <iostream>
#include "../src/DOC/DOC.h"
#include <random>
#include "../src/randomCudaScripts/DeleteFromArray.h"
#include <cuda.h>


TEST(testPrefixSum, testOne){

	// Set up clock for timing comparisons
	srand(time(NULL));
	std::clock_t start;
	double duration;

	unsigned int h_in_len = 0;
	for (int k = 1; k < 13; ++k)
	{
		h_in_len = (1 << k) + 3;
		//std::cout << "h_in size: " << h_in_len << std::endl;

		// Generate input
		unsigned int* h_in = new unsigned int[h_in_len];
		for (int i = 0; i < h_in_len; ++i)
		{
			//h_in[i] = rand() % 2;
			h_in[i] = i;
		}

		// Set up host-side memory for output
		unsigned int* h_out_naive = new unsigned int[h_in_len];
		unsigned int* h_out_blelloch = new unsigned int[h_in_len];

		// Set up device-side memory for input
		unsigned int* d_in;
		checkCudaErrors(cudaMalloc(&d_in, sizeof(unsigned int) * h_in_len));
		checkCudaErrors(cudaMemcpy(d_in, h_in, sizeof(unsigned int) * h_in_len, cudaMemcpyHostToDevice));

		// Set up device-side memory for output
		unsigned int* d_out_blelloch;
		checkCudaErrors(cudaMalloc(&d_out_blelloch, sizeof(unsigned int) * h_in_len));

		// Do CPU scan for reference
		start = std::clock();
		cpu_sum_scan(h_out_naive, h_in, h_in_len);
		duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
		//std::cout << "CPU time: " << duration << std::endl;

		cudaStream_t stream;
		checkCudaErrors(cudaStreamCreate(&stream));
		// Do GPU scan
		start = std::clock();
		sum_scan_blelloch(stream, d_out_blelloch, d_in, h_in_len);
		duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
		checkCudaErrors(cudaStreamDestroy(stream));
		//std::cout << "GPU time: " << duration << std::endl;

		// Copy device output array to host output array
		// And free device-side memory
		checkCudaErrors(cudaMemcpy(h_out_blelloch, d_out_blelloch, sizeof(unsigned int) * h_in_len, cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaFree(d_out_blelloch));
		checkCudaErrors(cudaFree(d_in));

		// Check for any mismatches between outputs of CPU and GPU
		bool match = true;
		int index_diff = 0;
		for (int i = 0; i < h_in_len; ++i)
		{
			if (h_out_naive[i] != h_out_blelloch[i])
			{
				match = false;
				index_diff = i;
				break;
			}
		}
		//std::cout << "Match: " << match << std::endl;

		// Detail the mismatch if any
		if (!match)
		{
			//std::cout << "Difference in index: " << index_diff << std::endl;
			//std::cout << "CPU: " << h_out_naive[index_diff] << std::endl;
			//std::cout << "Blelloch: " << h_out_blelloch[index_diff] << std::endl;
			int window_sz = 10;

			//std::cout << "Contents: " << std::endl;
			//std::cout << "CPU: ";
			for (int i = -(window_sz / 2); i < (window_sz / 2); ++i)
			{
				///	std::cout << h_out_naive[index_diff + i] << ", ";
			}
			//std::cout << std::endl;
			//std::cout << "Blelloch: ";
			for (int i = -(window_sz / 2); i < (window_sz / 2); ++i)
			{
				//	std::cout << h_out_blelloch[index_diff + i] << ", ";
			}
			//std::cout << std::endl;
			EXPECT_TRUE(false);
		}

		// Free host-side memory
		delete[] h_in;
		delete[] h_out_naive;
		delete[] h_out_blelloch;

		//std::cout << std::endl;


	}

	EXPECT_TRUE(true);
}


TEST(testPrefixSum, testTwo){
	// Set up clock for timing comparisons
	srand(time(NULL));
	std::clock_t start;
	double duration;

	unsigned int h_in_len = 0;
	for (int k = 1; k < 8; ++k)
	{


		//h_dataLenght = (1 << k) + 3;
		unsigned int h_dataLenght = 10;
		//std::cout << "h_in size: " << h_dataLenght << std::endl;


		// Generate input data
		float* h_data = new float[h_dataLenght];
		for (int i = 0; i <h_dataLenght; ++i)
		{
			h_data[i] = (float)(rand() % 100);
			//h_data[i] = i;
		}
		// Generate bool data
		bool* h_deleteArray = new bool[h_dataLenght+1];
		for (int i = 0; i < h_dataLenght; ++i)
		{
			if(rand() % 2){
				h_deleteArray[i] = true;
			}else{
				h_deleteArray[i] = false;
			}
		}
		h_deleteArray[h_dataLenght] = false;

		//extra stuff
		if(k==1){
			for (int i = 0; i < h_dataLenght; ++i)
			{
				h_deleteArray[i] = false;
			}
			h_deleteArray[h_dataLenght] = true;
		}
		if(k==2){
			for (int i = 0; i < h_dataLenght; ++i)
			{
				h_deleteArray[i] = true;
			}
			h_deleteArray[h_dataLenght] = false;
		}
		if(k == 3){
			h_deleteArray[h_dataLenght-1] = true;
			h_deleteArray[h_dataLenght] = false;
		}
		if(k == 4){
			h_deleteArray[h_dataLenght-1] = false;
			h_deleteArray[h_dataLenght] = false;
		}
		if(k == 5){
			h_deleteArray[h_dataLenght-1] = true;
			h_deleteArray[h_dataLenght] = true;
		}
		if(k == 6){
			h_deleteArray[h_dataLenght-1] = false;
			h_deleteArray[h_dataLenght] = true;
		}


		unsigned int* h_indexes = new unsigned int[h_dataLenght];
		for(int i = 0; i < h_dataLenght; ++i){
			h_indexes[i] = i;
		}
		// Set up host-side memory for output data
		float* h_outData_naive = new float[h_dataLenght];
		float* h_outData_gpu = new float[h_dataLenght];

		// Set up device-side memory for bool data
		bool* d_deleteArray;
		checkCudaErrors(cudaMalloc(&d_deleteArray, sizeof(bool) * (h_dataLenght+1)));
		checkCudaErrors(cudaMemcpy(d_deleteArray, h_deleteArray, sizeof(bool) * (h_dataLenght+1), cudaMemcpyHostToDevice));

		// Set up device-side memory for input data
		float* d_inputData;
		checkCudaErrors(cudaMalloc(&d_inputData, sizeof(float) * h_dataLenght));
		checkCudaErrors(cudaMemcpy(d_inputData, h_data, sizeof(float) * h_dataLenght, cudaMemcpyHostToDevice));

		// Set up device-side memory for input indexes
		unsigned int* d_inputIndexes;
		checkCudaErrors(cudaMalloc(&d_inputIndexes, sizeof(unsigned int) * h_dataLenght));
		checkCudaErrors(cudaMemcpy(d_inputIndexes, h_indexes, sizeof(unsigned int) * h_dataLenght, cudaMemcpyHostToDevice));


		// Set up device-side memory for output data
		float* d_outData;
		checkCudaErrors(cudaMalloc(&d_outData, sizeof(float) * h_dataLenght));
		//to test , i fill out the imput
		float* h_garbageArray = new float[h_dataLenght];
		for(int i = 0; i < h_dataLenght; ++i)
		{
			h_garbageArray[i] = -1.0;
		}
		checkCudaErrors(cudaMemcpy(d_outData, h_garbageArray, sizeof(float) * h_dataLenght, cudaMemcpyHostToDevice));

		bool print = false;
		if(print){
			//to test the cpu thing actualy works (test of test :( )
			std::cout << std::endl << "bool array: ";
			for(int i = 0; i < (h_dataLenght+1); ++i){
				std::cout << std::to_string(h_deleteArray[i])<< "         ";
			}
			std::cout << std::endl;

		}

		// Do CPU for reference
		start = std::clock();
		cpuDeleteFromArray(h_outData_naive, h_deleteArray, h_data, h_dataLenght);
		duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
		//std::cout << "CPU time: " << duration << std::endl;


		if(print){
			std::cout << "input:     ";
			for(int i = 0; i < h_dataLenght; ++i){
				std::cout << std::to_string(h_data[i])<< " ";
			}
			std::cout << std::endl;
		}






		// Do GPU run
		start = std::clock();
		deleteFromArray(d_outData, d_deleteArray, d_inputData ,(long unsigned int) h_dataLenght);
		duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;


		//std::cout << "GPU time: " << duration << std::endl;

		// Copy device output array to host output array
		// And free device-side memory
		checkCudaErrors(cudaMemcpy(h_outData_gpu, d_outData, sizeof(float) * h_dataLenght, cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaFree(d_outData));
		checkCudaErrors(cudaFree(d_inputData));
		checkCudaErrors(cudaFree(d_deleteArray));

		//need to get how many were deleted
		unsigned int howManyDeleted = 0;
		for(unsigned int i = 0; i < h_dataLenght; ++i){
			howManyDeleted += h_deleteArray[i];
		}

		if(print){
			std::cout << "cpu result:";
			for(int i = 0; i < (h_dataLenght-howManyDeleted); ++i){
				std::cout << std::to_string(h_outData_naive[i])<< " ";
			}
			std::cout << std::endl;


			std::cout << "gpu result: ";
			for(int i = 0; i < (h_dataLenght-howManyDeleted); ++i){
				std::cout << std::to_string(h_outData_gpu[i]) << " ";
			}
			std::cout << std::endl;


		}

		// Check for any mismatches between outputs of CPU and GPU
		bool match = true;
		int index_diff = 0;
		for (int i = 0; i < (h_dataLenght-howManyDeleted); ++i)
		{
			if (h_outData_naive[i] != h_outData_gpu[i])
			{
				match = false;
				index_diff = i;
				break;
			}
		}
		//std::cout << "Match: " << match << std::endl;

		// Detail the mismatch if any
		if (!match)
		{
			//std::cout << "Difference in index: " << index_diff << std::endl;
			//std::cout << "CPU: " << h_outData_naive[index_diff] << std::endl;
			//std::cout << "GPU: " << h_outData_gpu[index_diff] << std::endl;
			/*
			int window_sz = 4;


			std::cout << "Contents: " << std::endl;
			std::cout << "CPU: ";
			for (int i = -(window_sz / 2); i < (window_sz / 2); ++i)
			{
				std::cout << std::to_string(h_outData_naive[index_diff + i]) << ", ";
			}
			std::cout << std::endl;
			std::cout << "GPU: ";
			for (int i = -(window_sz / 2); i < (window_sz / 2); ++i)
			{
				std::cout << std::to_string(h_outData_gpu[index_diff + i]) << ", ";
			}
			std::cout << std::endl;
			*/


			EXPECT_TRUE(false);

		}

		// Free host-side memory
		//delete[] h_in;
		//delete[] h_out_naive;
		//delete[] h_out_blelloch;

		//std::cout << std::endl;


	}

	EXPECT_TRUE(true);



}

TEST(testPrefixSum, testThree){
	// Set up clock for timing comparisons
	srand(time(NULL));
	std::clock_t start;
	double duration;
	unsigned int dimensions = 3;
	unsigned int h_in_len = 0;
	for (int k = 1; k < 8; ++k)
	{


		//h_dataLenght = (1 << k) + 3;
		unsigned int h_dataLenght = 10;
		//std::cout << "h_in size: " << h_dataLenght << std::endl;


		// Generate input data
		float* h_data = new float[h_dataLenght*dimensions];
		for (int i = 0; i <h_dataLenght*dimensions; ++i)
		{
			h_data[i] = (float)(rand() % 100);
			//h_data[i] = i;
		}
		// Generate bool data
		bool* h_deleteArray = new bool[h_dataLenght+1];
		for (int i = 0; i < h_dataLenght; ++i)
		{
			if(rand() % 2){
				h_deleteArray[i] = true;
			}else{
				h_deleteArray[i] = false;
			}
		}
		h_deleteArray[h_dataLenght] = false;

		//extra stuff
		if(k==1){
			for (int i = 0; i < h_dataLenght; ++i)
			{
				h_deleteArray[i] = false;
			}
			h_deleteArray[h_dataLenght] = true;
		}
		if(k==2){
			for (int i = 0; i < h_dataLenght; ++i)
			{
				h_deleteArray[i] = true;
			}
			h_deleteArray[h_dataLenght] = false;
		}
		if(k == 3){
			h_deleteArray[h_dataLenght-1] = true;
			h_deleteArray[h_dataLenght] = false;
		}
		if(k == 4){
			h_deleteArray[h_dataLenght-1] = false;
			h_deleteArray[h_dataLenght] = false;
		}
		if(k == 5){
			h_deleteArray[h_dataLenght-1] = true;
			h_deleteArray[h_dataLenght] = true;
		}
		if(k == 6){
			h_deleteArray[h_dataLenght-1] = false;
			h_deleteArray[h_dataLenght] = true;
		}



		// Set up host-side memory for output data
		float* h_outData_naive = new float[h_dataLenght * dimensions];
		float* h_outData_gpu = new float[h_dataLenght * dimensions];

		// Set up device-side memory for bool data
		bool* d_deleteArray;
		checkCudaErrors(cudaMalloc(&d_deleteArray, sizeof(bool) * (h_dataLenght+1)));
		checkCudaErrors(cudaMemcpy(d_deleteArray, h_deleteArray, sizeof(bool) * (h_dataLenght+1), cudaMemcpyHostToDevice));

		// Set up device-side memory for input data
		float* d_inputData;
		checkCudaErrors(cudaMalloc(&d_inputData, sizeof(float) * h_dataLenght * dimensions));
		checkCudaErrors(cudaMemcpy(d_inputData, h_data, sizeof(float) * h_dataLenght * dimensions, cudaMemcpyHostToDevice));


		// Set up device-side memory for output data
		float* d_outData;
		checkCudaErrors(cudaMalloc(&d_outData, sizeof(float) * h_dataLenght * dimensions));

		bool print = false;
		if(print){
			//to test the cpu thing actualy works (test of test :( )
			std::cout << std::endl << "bool array: ";
			for(int i = 0; i < (h_dataLenght+1); ++i){
				std::cout << std::to_string(h_deleteArray[i])<< "         ";
			}
			std::cout << std::endl;

		}

		// Do CPU for reference
		start = std::clock();
		cpuDeleteFromArray(h_outData_naive, h_deleteArray, h_data, h_dataLenght,dimensions);
		duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
		//std::cout << "CPU time: " << duration << std::endl;


		if(print){
			std::cout << "input:     ";
			for(int i = 0; i < h_dataLenght*dimensions; ++i){
				std::cout << std::to_string(h_data[i])<< " ";
			}
			std::cout << std::endl;
		}


		// Do GPU run
		start = std::clock();
		deleteFromArray(d_outData, d_deleteArray, d_inputData ,(long unsigned int) h_dataLenght,dimensions);
		duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;


		//std::cout << "GPU time: " << duration << std::endl;

		// Copy device output array to host output array
		// And free device-side memory
		checkCudaErrors(cudaMemcpy(h_outData_gpu, d_outData, sizeof(float) * h_dataLenght*dimensions, cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaFree(d_outData));
		checkCudaErrors(cudaFree(d_inputData));
		checkCudaErrors(cudaFree(d_deleteArray));

		//need to get how many were deleted
		unsigned int howManyDeleted = 0;
		for(unsigned int i = 0; i < h_dataLenght; ++i){
			howManyDeleted += h_deleteArray[i];
		}

		if(print){
			std::cout << "cpu result:";
			for(int i = 0; i < (h_dataLenght-howManyDeleted)*dimensions; ++i){
				std::cout << std::to_string(h_outData_naive[i])<< " ";
			}
			std::cout << std::endl;


			std::cout << "gpu result: ";
			for(int i = 0; i < (h_dataLenght-howManyDeleted)*dimensions; ++i){
				std::cout << std::to_string(h_outData_gpu[i]) << " ";
			}
			std::cout << std::endl;


		}

		// Check for any mismatches between outputs of CPU and GPU
		bool match = true;
		int index_diff = 0;
		for (int i = 0; i < (h_dataLenght-howManyDeleted)*dimensions; ++i)
		{
			if (h_outData_naive[i] != h_outData_gpu[i])
			{
				match = false;
				index_diff = i;
				break;
			}
		}
		//std::cout << "Match: " << match << std::endl;

		// Detail the mismatch if any
		if (!match)
		{
			std::cout << "Difference in index: " << index_diff << std::endl;
			//std::cout << "CPU: " << h_outData_naive[index_diff] << std::endl;
			//std::cout << "GPU: " << h_outData_gpu[index_diff] << std::endl;
			/*
			int window_sz = 4;

			std::cout << "Contents: " << std::endl;
			std::cout << "CPU: ";
			for (int i = -(window_sz / 2); i < (window_sz / 2); ++i)
			{
				std::cout << std::to_string(h_outData_naive[index_diff + i]) << ", ";
			}
			std::cout << std::endl;
			std::cout << "GPU: ";
			for (int i = -(window_sz / 2); i < (window_sz / 2); ++i)
			{
				std::cout << std::to_string(h_outData_gpu[index_diff + i]) << ", ";
			}
			std::cout << std::endl;
			*/
			EXPECT_TRUE(false);

		}

		// Free host-side memory
		//delete[] h_in;
		//delete[] h_out_naive;
		//delete[] h_out_blelloch;

		//std::cout << std::endl;


	}

	EXPECT_TRUE(true);



}


TEST(testPrefixSum, testDeleteSimple){
	float a_h[4] = {1.0, 2.0, 3.0, 4.0};
	bool b_h[5] = {true, false, true, false, false};
	float* a_d;
	bool* b_d;
	float* out_d;
	cudaMalloc((void **) &a_d, 4*sizeof(float));
	cudaMalloc((void **) &b_d, 4*sizeof(bool));
	cudaMalloc((void **) &out_d, 2*sizeof(float));

	cudaMemcpy(a_d, a_h, 4*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(b_d, b_h, 4*sizeof(bool), cudaMemcpyHostToDevice);

	deleteFromArray(out_d, b_d, a_d, 4, 1);

	float* out_h = (float*) malloc(2*sizeof(float));
	cudaMemcpy(out_h, out_d, 2*sizeof(float), cudaMemcpyDeviceToHost);

	EXPECT_EQ(out_h[0], 2.0);
	EXPECT_EQ(out_h[1], 4.0);
}


TEST(testPrefixSum, testDeleteSimple2){
	float a_h[4] = {1.0, 2.0, 3.0, 4.0};
	bool b_h[5] = {true, true, true, false, false};
	float* a_d;
	bool* b_d;
	float* out_d;
	cudaMalloc((void **) &a_d, 4*sizeof(float));
	cudaMalloc((void **) &b_d, 4*sizeof(bool));
	cudaMalloc((void **) &out_d, 1*sizeof(float));

	cudaMemcpy(a_d, a_h, 4*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(b_d, b_h, 4*sizeof(bool), cudaMemcpyHostToDevice);

	deleteFromArray(out_d, b_d, a_d, 4, 1);

	float* out_h = (float*) malloc(1*sizeof(float));
	cudaMemcpy(out_h, out_d, 1*sizeof(float), cudaMemcpyDeviceToHost);

	EXPECT_EQ(out_h[0], 4.0);
}


TEST(testPrefixSum, testDeleteSimple3){
	float a_h[4] = {1.0, 2.0, 3.0, 4.0};
	bool b_h[5] = {true, true, true, true, false};
	float* a_d;
	bool* b_d;
	float* out_d;
	cudaMalloc((void **) &a_d, 4*sizeof(float));
	cudaMalloc((void **) &b_d, 4*sizeof(bool));
	cudaMalloc((void **) &out_d, 1*sizeof(float));

	cudaMemcpy(a_d, a_h, 4*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(b_d, b_h, 4*sizeof(bool), cudaMemcpyHostToDevice);

	deleteFromArray(out_d, b_d, a_d, 4, 1);

	float* out_h = (float*) malloc(1*sizeof(float));
	cudaMemcpy(out_h, out_d, 1*sizeof(float), cudaMemcpyDeviceToHost);

	EXPECT_NE(out_h[0], 4.0);
}



TEST(testPrefixSum, testDeleteSimple4){
	float a_h[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
	bool b_h[5] = {true, false, true, false, false};
	float* a_d;
	bool* b_d;
	float* out_d;
	cudaMalloc((void **) &a_d, 8*sizeof(float));
	cudaMalloc((void **) &b_d, 4*sizeof(bool));
	cudaMalloc((void **) &out_d, 4*sizeof(float));

	cudaMemcpy(a_d, a_h, 8*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(b_d, b_h, 4*sizeof(bool), cudaMemcpyHostToDevice);

	deleteFromArray(out_d, b_d, a_d, 4, 2);

	float* out_h = (float*) malloc(4*sizeof(float));
	cudaMemcpy(out_h, out_d, 4*sizeof(float), cudaMemcpyDeviceToHost);

	EXPECT_EQ(out_h[0], 3.0);
	EXPECT_EQ(out_h[1], 4.0);

	EXPECT_EQ(out_h[2], 7.0);
	EXPECT_EQ(out_h[3], 8.0);
}


TEST(testPrefixSum, testDeleteSimple5){
	float a_h[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
	bool b_h[5] = {true, false, true, false, false};
	float* a_d;
	bool* b_d;
	float* out_d;
	cudaMalloc((void **) &a_d, 8*sizeof(float));
	cudaMalloc((void **) &b_d, 4*sizeof(bool));
	cudaMalloc((void **) &out_d, 4*sizeof(float));

	cudaMemcpy(a_d, a_h, 8*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(b_d, b_h, 4*sizeof(bool), cudaMemcpyHostToDevice);

	deleteFromArray(out_d, b_d, a_d, 4, 2);

	float* out_h = (float*) malloc(4*sizeof(float));
	cudaMemcpy(out_h, out_d, 4*sizeof(float), cudaMemcpyDeviceToHost);

	EXPECT_EQ(out_h[0], 3.0);
	EXPECT_EQ(out_h[1], 4.0);

	EXPECT_EQ(out_h[2], 7.0);
	EXPECT_EQ(out_h[3], 8.0);

	float* out2_h = (float*) malloc(8*sizeof(float));
	cudaMemcpy(out2_h, a_d, 8*sizeof(float), cudaMemcpyDeviceToHost);
	EXPECT_EQ(out2_h[0], 1.0);
	EXPECT_EQ(out2_h[1], 2.0);
	EXPECT_EQ(out2_h[2], 3.0);
	EXPECT_EQ(out2_h[3], 4.0);
	EXPECT_EQ(out2_h[4], 5.0);
	EXPECT_EQ(out2_h[5], 6.0);
	EXPECT_EQ(out2_h[6], 7.0);
	EXPECT_EQ(out2_h[7], 8.0);
}


TEST(testPrefixSum, testDeleteInputOutput){
	float a_h[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
	bool b_h[5] = {true,      false,    true,     false,   false};
	float* a_d;
	bool* b_d;
	float* out_d;
	cudaMalloc((void **) &a_d, 8*sizeof(float));
	cudaMalloc((void **) &b_d, 4*sizeof(bool));
	cudaMalloc((void **) &out_d, 4*sizeof(float));

	cudaMemcpy(a_d, a_h, 8*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(b_d, b_h, 4*sizeof(bool), cudaMemcpyHostToDevice);

	deleteFromArray(a_d, b_d, a_d, 4, 2);

	float* out_h = (float*) malloc(4*sizeof(float));
	cudaMemcpy(out_h, a_d, 4*sizeof(float), cudaMemcpyDeviceToHost);

	EXPECT_EQ(out_h[0], 3.0);
	EXPECT_EQ(out_h[1], 4.0);

	EXPECT_EQ(out_h[2], 7.0);
	EXPECT_EQ(out_h[3], 8.0);
}



TEST(testPrefixSum, _SLOW_testDeleteInputOutput2){
	unsigned int dim = 301;
	unsigned int n = 51200*dim;
	float* data_h = (float*) malloc(n*sizeof(float));
	float* data_out_h = (float*) malloc(n*sizeof(float));
	bool* mask_h = (bool*) malloc(((n/dim)+1)*sizeof(bool));
	for(int i = 0; i < n; i++){
		data_h[i] = i;
	}

	unsigned int output_points = 0;
	for(int i = 0; i < (n/dim)+1; i++){
		mask_h[i] = (i%2)==0;
		if((not mask_h[i]) && i < (n/dim)){
			output_points++;
		}
	}

	unsigned int output_numbers = output_points*dim;

	float* out_d;
	bool* mask_d;
	float* data_d;
	float* data_out_d;

	
	cudaMalloc((void **) &data_d, n*sizeof(float));
	cudaMalloc((void **) &data_out_d, n*sizeof(float));
	cudaMalloc((void **) &mask_d, ((n/dim)+1)*sizeof(bool));

  
	cudaMemcpy(mask_d, mask_h, ((n/dim)+1)*sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(data_d, data_h, n*sizeof(float), cudaMemcpyHostToDevice);

	
	deleteFromArray(data_d, mask_d, data_d, n/dim, dim);

	cudaMemcpy(data_out_h, data_d, output_numbers*sizeof(float), cudaMemcpyDeviceToHost);

	int a = 0;
	for(int i = 0; i < n/dim; i++){
		if(not mask_h[i]){
			for(int j = 0; j < dim; j++){
				//std::cout << data_out_h[a*dim+j] << ", ";
				EXPECT_EQ(data_h[i*dim+j], data_out_h[a*dim+j])  << "i: " << i << ", j: " << j << ",a " << a;
			}
			a++;
		}else{
			for(int j = 0; j < dim; j++){
				EXPECT_NE(data_h[i*dim+j], data_out_h[a*dim+j]);
				EXPECT_NE(data_h[a*dim+j], data_out_h[a*dim+j]);
				//std::cout << data_h[i*dim+j] << ", ";
			}
			//std::cout << " was deleted";
			
		}
		//std::cout << std::endl;
	}

	

	/*	for(int i = 0; i < output_points; i++){
		for(int j = 0; j < dim; j++){
			std::cout << data_out_h[i*dim+j] << ", " ;
		}
		std::cout << std::endl;
		}*/
	
	/*for(int i = 0; i < output_points; i++){
		for(int j = 0; j < dim; j++){
			std::cout << data_out_h[i*dim+j] << ", ";
		}
		std::cout << std::endl;
		}*/


}


TEST(testPrefixSum, testDeleteInputOutput3){
	unsigned int dim = 31;
	unsigned int n = 513*dim;
	float* data_h = (float*) malloc(n*sizeof(float));
	float* data_out_h = (float*) malloc(n*sizeof(float));
	bool* mask_h = (bool*) malloc(((n/dim)+1)*sizeof(bool));
	for(int i = 0; i < n; i++){
		data_h[i] = i;
	}

	unsigned int output_points = 0;
	for(int i = 0; i < (n/dim)+1; i++){
		mask_h[i] = (i%2)==0;
		if((not mask_h[i]) && i < (n/dim)){
			output_points++;
		}
	}

	unsigned int output_numbers = output_points*dim;

	float* out_d;
	bool* mask_d;
	float* data_d;
	float* data_out_d;

	
	cudaMalloc((void **) &data_d, n*sizeof(float));
	cudaMalloc((void **) &data_out_d, n*sizeof(float));
	cudaMalloc((void **) &mask_d, ((n/dim)+1)*sizeof(bool));

  
	cudaMemcpy(mask_d, mask_h, ((n/dim)+1)*sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(data_d, data_h, n*sizeof(float), cudaMemcpyHostToDevice);

    cudaStream_t stream1;
    checkCudaErrors(cudaStreamCreate(&stream1));
	deleteFromArray(stream1, data_d, mask_d, data_d, n/dim, dim);

	cudaMemcpyAsync(data_out_h, data_d, output_numbers*sizeof(float), cudaMemcpyDeviceToHost, stream1);

	int a = 0;
	for(int i = 0; i < n/dim; i++){
		if(not mask_h[i]){
			for(int j = 0; j < dim; j++){
				//std::cout << data_out_h[a*dim+j] << ", ";
				EXPECT_EQ(data_h[i*dim+j], data_out_h[a*dim+j])  << "i: " << i << ", j: " << j << ",a " << a;
			}
			a++;
		}else{
			for(int j = 0; j < dim; j++){
				EXPECT_NE(data_h[i*dim+j], data_out_h[a*dim+j]);
				EXPECT_NE(data_h[a*dim+j], data_out_h[a*dim+j]);
				//std::cout << data_h[i*dim+j] << ", ";
			}
			//std::cout << " was deleted";
			
		}
		//std::cout << std::endl;
	}

}
