#include "ExperimentClusteringSpeed.h"
#include <unistd.h>
#include <iostream>
#include "../src/testingTools/DataGeneratorBuilder.h"
#include "../src/dataReader/DataReader.h"
#include "../src/randomCudaScripts/Utils.h"
#include "../src/randomCudaScripts/DeleteFromArray.h"
#include "../src/Evaluation.h"
#include "ExperimentDelete.h"
#include <random>

#include <chrono>

void ExperimentDelete::start(){
	unsigned int dim =  128;
	unsigned int numberOfPoints = 1048576;
	unsigned int c = 0; 
	// Count number of tests
	for(unsigned int j = 32; j <= numberOfPoints; j *=2){
		c+=2;
	}

	Experiment::addTests(c);
	
	Experiment::start();
	int seed = 0;
	for(unsigned int j = 32; j <= numberOfPoints; j *=2){
		float deleteFraction = 0.5;
		size_t sizeOfData = j*dim*sizeof(float);
		size_t sizeOfDeleteBools = (j+1)*sizeof(bool);
		size_t sizeOfPrefixsum = (j+1)*sizeof(unsigned int);
		size_t sizeOfOutput = (j)*dim*sizeof(float);

		float* data;
		bool* deleteBool;
		unsigned int* prefixSum;
		float* output1;
		float* output2;

		checkCudaErrors(cudaMallocManaged((void**) &data, sizeOfData));
		checkCudaErrors(cudaMallocManaged((void**) &deleteBool, sizeOfDeleteBools));
		checkCudaErrors(cudaMallocManaged((void**) &prefixSum, sizeOfPrefixsum));
		checkCudaErrors(cudaMallocManaged((void**) &output1, sizeOfOutput));
		checkCudaErrors(cudaMallocManaged((void**) &output2, sizeOfOutput));


		std::default_random_engine gen;
		std::uniform_real_distribution<float> dist(0.0, 1000000.0);
		for(unsigned int i = 0; i < j*dim; i++){
			data[i] = dist(gen);
		}
		for(unsigned int i = 0; i < j; i++){
			deleteBool[i] = (i%2) == 0;
		}


		float millisNaive = 0;
		float millisSpecial = 0;
		cudaStream_t stream;
		checkCudaErrors(cudaStreamCreate(&stream));
		cudaEvent_t start_e, stop_e;
		cudaEventCreate(&start_e);
		cudaEventCreate(&stop_e);
		
		checkCudaErrors(cudaMemPrefetchAsync(prefixSum, sizeOfPrefixsum, 0, stream));
		checkCudaErrors(cudaMemPrefetchAsync(deleteBool, sizeOfDeleteBools, 0, stream));
		sum_scan_blelloch_managed(stream, stream, prefixSum, deleteBool, (size_t)j+1, false);
		// for(unsigned int i = 0; i < j; i++){
		// 	std::cout << prefixSum[i] << " ";
		// }
		// std::cout << std::endl;
		
		
		// Naive
		{
			checkCudaErrors(cudaMemPrefetchAsync(data, sizeOfData, 0, stream));
			checkCudaErrors(cudaMemPrefetchAsync(prefixSum, sizeOfPrefixsum, 0, stream));
			checkCudaErrors(cudaMemPrefetchAsync(output1, sizeOfOutput, 0, stream));

			checkCudaErrors(cudaEventRecord(start_e, stream));

			deleteFromArrayWrapper(stream,data, prefixSum, j, dim, output1);
			
			checkCudaErrors(cudaEventRecord(stop_e, stream));
			checkCudaErrors(cudaEventSynchronize(stop_e));
			cudaEventElapsedTime(&millisNaive, start_e, stop_e);
			
		}
		Experiment::testDone("Naive Number of points: " + std::to_string(j));
		{
			checkCudaErrors(cudaMemPrefetchAsync(data, sizeOfData, 0, stream));
			checkCudaErrors(cudaMemPrefetchAsync(prefixSum, sizeOfPrefixsum, 0, stream));
			checkCudaErrors(cudaMemPrefetchAsync(output2, sizeOfOutput, 0, stream));

			checkCudaErrors(cudaEventRecord(start_e, stream));

			deleteFromArraySpecialWrapper(stream,data, prefixSum, j, dim, output2);
			
			checkCudaErrors(cudaEventRecord(stop_e, stream));
			checkCudaErrors(cudaEventSynchronize(stop_e));
			cudaEventElapsedTime(&millisSpecial, start_e, stop_e);
			
		}

		checkCudaErrors(cudaMemPrefetchAsync(output1, sizeOfOutput, cudaCpuDeviceId, stream));
		checkCudaErrors(cudaMemPrefetchAsync(output2, sizeOfOutput, cudaCpuDeviceId, stream));
		bool eq = true; 
		for(unsigned int i = 0; i < (j*deleteFraction)*dim; i++){
			eq &= output1[i] == output2[i];
			if(!eq){
				this->repportError("Outputs do not match : " + std::to_string(i), this->getName());	
			}
		}
		this->writeLineToFile(std::to_string(j) + ", "
							  + std::to_string(dim) + ", "
							  +  std::to_string(deleteFraction) + ", "
							  ",Naive, "
							  + std::to_string(millisNaive));
		
		this->writeLineToFile(std::to_string(j) + ", "
							  + std::to_string(dim) + ", "
							  +  std::to_string(deleteFraction) + ", "
							  ",Special, "
							  + std::to_string(millisSpecial));

		Experiment::testDone("Special Number of points: " + std::to_string(j));
		
	}
	
	cudaDeviceReset();
	Experiment::stop();
};




