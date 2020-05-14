#include "fakeExperiment.h"
#include <random>
#include <string>
#include <iostream>
#include "../src/randomCudaScripts/Utils.h"
#include "../src/randomCudaScripts/arrayEqual.h"
#include "../src/DOC_GPU/pointsContainedDevice.h"
#include "../src/Fast_DOCGPU/whatDataInCentroid.h"

void fakeExperiment::start(){
	std::random_device rd;
	std::mt19937 gen(rd());
	gen.seed(0); // important to seed
	std::uniform_real_distribution<> dist(-100, 100);
	std::uniform_real_distribution<double> distribution2(9,26);
	unsigned int c = 0;
	Experiment::addTests(c);

	Experiment::start();



	//*********************************************************************************************************************
	int block_size = 1024;
	unsigned long garbage = 0;
	unsigned long width = 15;
	unsigned long breakIntervall = 1;
	// Calculaating sizes
	std::size_t point_dim = 32;
	std::size_t no_of_points = 10000;
	std::size_t no_of_dims = 10; //idk
	std::size_t no_of_centroids = 20;
	unsigned int m = ceilf((float)no_of_dims/(float)no_of_centroids);

	std::size_t floats_in_data = point_dim * no_of_points;
	std::size_t bools_in_dims = no_of_dims * point_dim;
	std::size_t bools_in_output = no_of_points * no_of_dims;
	std::size_t ints_in_output_count = no_of_dims;

	std::size_t size_of_data = floats_in_data*sizeof(float);
	std::size_t size_of_dims = bools_in_dims*sizeof(bool);
	std::size_t size_of_centroids = no_of_centroids*sizeof(unsigned int);
	std::size_t size_of_output = bools_in_output*sizeof(bool);
	std::size_t size_of_output_count = ints_in_output_count*sizeof(unsigned int);

	// allocating on the host
	float* data_h = (float*) malloc(size_of_data);
	bool* dims_h = (bool*) malloc(size_of_dims);
	unsigned int* centroids_h = (unsigned int*) malloc(size_of_centroids);
	bool* output_h = (bool*) malloc(size_of_output);
	unsigned int* output_count_h = (unsigned int*) malloc(size_of_output_count);


	// filling data array with random numbers
	for(int i= 0; i < no_of_points; i++){
		for(int j = 0; j < point_dim; j++){
			data_h[i*point_dim+j] = distribution2(gen);
		}
	}

	// filling dims array with random 50% yes or no
	for(int i= 0; i < no_of_dims; i++){
		for(int j = 0; j < point_dim; j++){
			dims_h[i*point_dim+j] = (distribution2(gen)<13);
		}
	}

	// filling centroid array with 0,1,2,3,4,5... this can give problems
	for(int i= 0; i < no_of_centroids; i++){
		centroids_h[i] = i;
	}

	// allocating on device
	float* data_d;
	bool* dims_d;
	unsigned int* centroids_d;
	bool* output_d;
	unsigned int* output_count_d;

	bool* garbageCleaner_h = (bool*) malloc(size_of_output);
	unsigned int* garbageCleanerCount_h = (unsigned int*) malloc(size_of_output_count);

	for(unsigned int i = 0 ; i < bools_in_output ; i++){
		garbageCleaner_h[i] = (bool)garbage;

	}
	for(unsigned int i = 0 ; i < ints_in_output_count ; i++){
		garbageCleanerCount_h[i] = (unsigned int)garbage;

	}

	cudaMalloc((void **) &data_d, size_of_data);
	cudaMalloc((void **) &dims_d, size_of_dims);
	cudaMalloc((void **) &centroids_d, size_of_centroids);
	cudaMalloc((void **) &output_d, size_of_output);
	cudaMalloc((void **) &output_count_d, size_of_output_count);


	cudaMemcpy(output_d, garbageCleaner_h, size_of_output, cudaMemcpyHostToDevice);
	cudaMemcpy(output_count_d, garbageCleanerCount_h, size_of_output_count, cudaMemcpyHostToDevice);
	//Copy from host to device

	cudaMemcpy(data_d, data_h, size_of_data, cudaMemcpyHostToDevice);
	cudaMemcpy(dims_d, dims_h, size_of_dims, cudaMemcpyHostToDevice);
	cudaMemcpy(centroids_d, centroids_h, size_of_centroids, cudaMemcpyHostToDevice);

	cudaStream_t stream;
	(cudaStreamCreate(&stream));

	//time taking
	cudaEvent_t start_naive, stop_naive;
	cudaEventCreate(&start_naive);
	cudaEventCreate(&stop_naive);
	checkCudaErrors(cudaEventRecord(start_naive, stream));

	pointsContainedKernelNaive(ceil((no_of_dims)/(float)block_size),
							   block_size,
							   stream,
							   data_d,
							   centroids_d,
							   dims_d,
							   output_d,
							   output_count_d,
							   width,
							   point_dim,
							   no_of_points,
							   no_of_dims,
							   m);

	checkCudaErrors(cudaEventRecord(stop_naive, stream));

	float millisReducedReadsNaive = 0;
	cudaEventSynchronize(stop_naive);

	cudaEventElapsedTime(&millisReducedReadsNaive, start_naive, stop_naive);


    (cudaStreamDestroy(stream));


	cudaFree(data_d);
	cudaFree(dims_d);
	cudaFree(centroids_d);
	cudaFree(output_d);
	cudaFree(output_count_d);
	free(data_h);
	free(dims_h);
	free(centroids_h);
	free(output_h);
	free(output_count_h);
	free(garbageCleaner_h);
	free(garbageCleanerCount_h);

	//****************************************************************************************************************************


	Experiment::stop();
}
