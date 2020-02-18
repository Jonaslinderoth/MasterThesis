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

		// Do GPU scan
		start = std::clock();
		sum_scan_blelloch(d_out_blelloch, d_in, h_in_len);
		duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
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
			std::cout << "Difference in index: " << index_diff << std::endl;
			std::cout << "CPU: " << h_out_naive[index_diff] << std::endl;
			std::cout << "Blelloch: " << h_out_blelloch[index_diff] << std::endl;
			int window_sz = 10;

			std::cout << "Contents: " << std::endl;
			std::cout << "CPU: ";
			for (int i = -(window_sz / 2); i < (window_sz / 2); ++i)
			{
				std::cout << h_out_naive[index_diff + i] << ", ";
			}
			std::cout << std::endl;
			std::cout << "Blelloch: ";
			for (int i = -(window_sz / 2); i < (window_sz / 2); ++i)
			{
				std::cout << h_out_blelloch[index_diff + i] << ", ";
			}
			std::cout << std::endl;
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
	for (int k = 1; k < 20; ++k)
	{
		int h_dataLenght = (1 << k) + 3;
		//unsigned int h_dataLenght = 10;
		std::cout << "h_in size: " << h_dataLenght << std::endl;

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
		h_deleteArray[h_dataLenght] = true;

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
		//to test
		//checkCudaErrors(cudaMemcpy(d_outData, h_data, sizeof(float) * h_dataLenght, cudaMemcpyHostToDevice));

		/*
		//to test the cpu thing actualy works (test of test :( )
		std::cout << std::endl << "bool array: ";
		for(int i = 0; i < (h_dataLenght+1); ++i){
			std::cout << std::to_string(h_deleteArray[i])<< "         ";
		}*/


		std::cout << std::endl;
		// Do CPU for reference
		start = std::clock();
		cpuDeleteFromArray(h_outData_naive, h_deleteArray, h_data, h_indexes , h_dataLenght);
		duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
		std::cout << "CPU time: " << duration << std::endl;

		/*
		std::cout << "input:     ";
		for(int i = 0; i < h_dataLenght; ++i){
			std::cout << std::to_string(h_data[i])<< " ";
		}

		std::cout << std::endl;
		 */



		// Do GPU run
		start = std::clock();
		deleteFromArray(d_outData, d_deleteArray, d_inputData,d_inputIndexes , h_dataLenght);
		duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
		std::cout << "GPU time: " << duration << std::endl;

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
		/*
		std::cout << "cpu result:";
		for(int i = 0; i < (h_dataLenght-howManyDeleted); ++i){
			std::cout << std::to_string(h_outData_naive[i])<< " ";
		}
		std::cout << std::endl;
		 */
		/*
		std::cout << "gpu result: ";
		for(int i = 0; i < (h_dataLenght-howManyDeleted); ++i){
			std::cout << std::to_string(h_outData_gpu[i]) << " ";
		}
		std::cout << std::endl;
		*/

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
		std::cout << "Match: " << match << std::endl;

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

		std::cout << std::endl;


	}

	EXPECT_TRUE(true);



}

