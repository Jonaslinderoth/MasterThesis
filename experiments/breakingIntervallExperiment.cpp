
#include "breakingIntervallExperiment.h"
#include <random>
#include <string>
#include <iostream>
#include "../src/randomCudaScripts/Utils.h"
#include "../src/randomCudaScripts/arrayEqual.h"
#include "../src/DOC_GPU/pointsContainedDevice.h"
#include "../src/Fast_DOCGPU/whatDataInCentroid.h"

void breakingIntervallExperiment::start(){

	std::random_device rd;
	std::mt19937 gen(rd());
	gen.seed(0); // important to seed
	std::uniform_real_distribution<> dist(-100, 100);
	std::uniform_real_distribution<double> distribution2(9,26);
	unsigned int c = 1;
	unsigned long breakIntervallIncrease = 1;
	unsigned long breakIntervallEnd = 50;
	unsigned long innerLoopCount = 0;
	for(unsigned long breakIntervall = 1; breakIntervall <= breakIntervallEnd ; breakIntervall+=breakIntervallIncrease){
		innerLoopCount++;
	}
	c = c*innerLoopCount;

	Experiment::addTests(c);

	Experiment::start();


	int block_size = 1024;
	unsigned long garbage = 0;
	unsigned long width = 15;
	// Calculaating sizes
	std::size_t point_dim = 200;
	std::size_t no_of_points = 10000;
	std::size_t no_of_dims = 50; //idk
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


	// filling data array with only 50
	for(int i= 0; i < no_of_points; i++){
		for(int j = 0; j < point_dim; j++){
			data_h[i*point_dim+j] = 50;
		}
	}

	// filling dims array with true
	for(int i= 0; i < no_of_dims; i++){
		for(int j = 0; j < point_dim; j++){
			dims_h[i*point_dim+j] = true;
		}
	}

	// filling centroid array with 0,1,2,3,4,5...
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
	checkCudaErrors(cudaMalloc((void **) &data_d, size_of_data));
	checkCudaErrors(cudaMalloc((void **) &dims_d, size_of_dims));
	checkCudaErrors(cudaMalloc((void **) &centroids_d, size_of_centroids));
	checkCudaErrors(cudaMalloc((void **) &output_d, size_of_output));
	checkCudaErrors(cudaMalloc((void **) &output_count_d, size_of_output_count));


	checkCudaErrors(cudaMemcpy(output_d, garbageCleaner_h, size_of_output, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(output_count_d, garbageCleanerCount_h, size_of_output_count, cudaMemcpyHostToDevice));
	//Copy from host to device

	checkCudaErrors(cudaMemcpy(data_d, data_h, size_of_data, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dims_d, dims_h, size_of_dims, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(centroids_d, centroids_h, size_of_centroids, cudaMemcpyHostToDevice));

	cudaStream_t stream;
	checkCudaErrors((cudaStreamCreate(&stream)));

	//time taking
	cudaEvent_t start_naive, stop_naive;
	checkCudaErrors(cudaEventCreate(&start_naive));
	checkCudaErrors(cudaEventCreate(&stop_naive));
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
	checkCudaErrors(cudaEventSynchronize(stop_naive));

	checkCudaErrors(cudaEventElapsedTime(&millisReducedReadsNaive, start_naive, stop_naive));

	Experiment::writeLineToFile(std::to_string(no_of_points) + ", " + std::to_string(point_dim) + ", " + "naive, 0," + std::to_string(millisReducedReadsNaive));
	Experiment::testDone("Dim: " + std::to_string(point_dim) + " numPoints: " + std::to_string(no_of_points)+ " naive");



	for(unsigned long breakIntervall = 1; breakIntervall <= breakIntervallEnd ; breakIntervall+=breakIntervallIncrease){
		//time taking
		cudaEvent_t start_breaking, stop_breaking;
		checkCudaErrors(cudaEventCreate(&start_breaking));
		checkCudaErrors(cudaEventCreate(&stop_breaking));
		checkCudaErrors(cudaEventRecord(start_breaking, stream));

		pointsContainedKernelNaiveBreak(ceil((no_of_dims)/(float)block_size),
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
											   m,
											   breakIntervall);

		checkCudaErrors(cudaEventRecord(stop_breaking, stream));

		float millisReducedReadsBreaking = 0;
		checkCudaErrors(cudaEventSynchronize(stop_breaking));

		checkCudaErrors(cudaEventElapsedTime(&millisReducedReadsBreaking , start_breaking, stop_breaking));

		Experiment::writeLineToFile(std::to_string(no_of_points) + ", " + std::to_string(point_dim) + ", naiveBreak, " + std::to_string(breakIntervall) + ", " + std::to_string(millisReducedReadsBreaking));
		Experiment::testDone("Dim: " + std::to_string(point_dim) + " numPoints: " + std::to_string(no_of_points)+ " breaking intervall: " + std::to_string(breakIntervall));
	}


	checkCudaErrors((cudaStreamDestroy(stream)));

	// copy from device
	checkCudaErrors(cudaMemcpy(output_h, output_d, size_of_output, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(output_count_h, output_count_d, size_of_output_count, cudaMemcpyDeviceToHost));

	// construnct output
	auto output =  new std::vector<std::vector<bool>*>;
	auto output_count =  new std::vector<unsigned int>;


	for(int i = 0; i < no_of_dims; i++){
		auto a =  new std::vector<bool>;
		for(int j = 0; j < no_of_points; j++){
			a->push_back(output_h[i*no_of_points+j]);
		}
		output->push_back(a);
	}

	for(int i = 0; i < no_of_dims; i++){
		output_count->push_back(output_count_h[i]);
	}

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

	cudaDeviceReset();
	Experiment::stop();
}
