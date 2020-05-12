// #include "breakingIntervallExperiment.h"
// #include <random>
// #include "../src/randomCudaScripts/Utils.h"

// void breakingIntervallExperiment::start(){
// 	std::random_device rd;
// 	std::mt19937 gen(rd());
// 	gen.seed(0); // important to seed
// 	std::uniform_real_distribution<> dist(-100, 100);
// 	std::uniform_real_distribution<double> distribution2(9,26);
// 	unsigned int c = 0;
// 	Experiment::addTests(c);
// 	Experiment::start();
// 	cudaEvent_t start_e, stop_e;
// 	cudaEventCreate(&start_e);
// 	cudaEventCreate(&stop_e);


// 	//*********************************************************************************************************************
// 	int block_size = 1024;
// 	unsigned long garbage = 0;

// 	// Calculaating sizes
// 	std::size_t point_dim = 32;
// 	std::size_t no_of_points = 10000;
// 	std::size_t no_of_dims = 10; //idk
// 	std::size_t no_of_centroids = 20;


// 	std::size_t floats_in_data = point_dim * no_of_points;
// 	std::size_t bools_in_dims = no_of_dims * point_dim;
// 	std::size_t bools_in_output = no_of_points * no_of_dims;
// 	std::size_t ints_in_output_count = no_of_dims;

// 	std::size_t size_of_data = floats_in_data*sizeof(float);
// 	std::size_t size_of_dims = bools_in_dims*sizeof(bool);
// 	std::size_t size_of_centroids = no_of_centroids*sizeof(unsigned int);
// 	std::size_t size_of_output = bools_in_output*sizeof(bool);
// 	std::size_t size_of_output_count = ints_in_output_count*sizeof(unsigned int);

// 	// allocating on the host
// 	float* data_h = (float*) malloc(size_of_data);
// 	bool* dims_h = (bool*) malloc(size_of_dims);
// 	unsigned int* centroids_h = (unsigned int*) malloc(size_of_centroids);
// 	bool* output_h = (bool*) malloc(size_of_output);
// 	unsigned int* output_count_h = (unsigned int*) malloc(size_of_output_count);


// 	// filling data array with random numbers
// 	for(int i= 0; i < no_of_points; i++){
// 		for(int j = 0; j < point_dim; j++){
// 			data_h[i*point_dim+j] = distribution2(gen);
// 		}
// 	}

// 	// filling dims array with random 50% yes or no
// 	for(int i= 0; i < no_of_dims; i++){
// 		for(int j = 0; j < point_dim; j++){
// 			dims_h[i*point_dim+j] = (distribution2(gen)<13);
// 		}
// 	}

// 	// filling centroid array with 0,1,2,3,4,5... this can give problems
// 	for(int i= 0; i < no_of_centroids; i++){
// 		centroids_h[i] = i;
// 	}

// 	// allocating on device
// 	float* data_d;
// 	bool* dims_d;
// 	unsigned int* centroids_d;
// 	bool* output_d;
// 	unsigned int* output_count_d;

// 	bool* garbageCleaner_h = (bool*) malloc(size_of_output);
// 	unsigned int* garbageCleanerCount_h = (unsigned int*) malloc(size_of_output_count);

// 	for(unsigned int i = 0 ; i < bools_in_output ; i++){
// 		garbageCleaner_h[i] = (bool)garbage;

// 	}
// 	for(unsigned int i = 0 ; i < ints_in_output_count ; i++){
// 		garbageCleanerCount_h[i] = (unsigned int)garbage;

// 	}

// 	cudaMalloc((void **) &data_d, size_of_data);
// 	cudaMalloc((void **) &dims_d, size_of_dims);
// 	cudaMalloc((void **) &centroids_d, size_of_centroids);
// 	cudaMalloc((void **) &output_d, size_of_output);
// 	cudaMalloc((void **) &output_count_d, size_of_output_count);


// 	cudaMemcpy(output_d, garbageCleaner_h, size_of_output, cudaMemcpyHostToDevice);
// 	cudaMemcpy(output_count_d, garbageCleanerCount_h, size_of_output_count, cudaMemcpyHostToDevice);
// 	//Copy from host to device

// 	cudaMemcpy(data_d, data_h, size_of_data, cudaMemcpyHostToDevice);
// 	cudaMemcpy(dims_d, dims_h, size_of_dims, cudaMemcpyHostToDevice);
// 	cudaMemcpy(centroids_d, centroids_h, size_of_centroids, cudaMemcpyHostToDevice);

// 	cudaStream_t stream;
// 	(cudaStreamCreate(&stream));

// 	unsigned int version = 0;
// 	checkCudaErrors(cudaEventRecord(start_e, stream));
// 	if(version == 0){
// 		// Call kernel
// 		pointsContainedKernelNaive(ceil((no_of_dims)/(float)block_size),
// 								   block_size,
// 								   stream,
// 								   data_d,
// 								   centroids_d,
// 								   dims_d,
// 								   output_d,
// 								   output_count_d,
// 								   width,
// 								   point_dim,
// 								   no_of_points,
// 								   no_of_dims,
// 								   m);


// 	}else if(version == 1){
// 		pointsContainedKernelSharedMemory(ceil((no_of_dims)/(float)block_size),
// 		                        block_size,
// 								  stream,
// 								  data_d,
// 								  centroids_d,
// 								  dims_d,
// 								  output_d,
// 								  output_count_d,
// 								  width,
// 								  point_dim,
// 								  no_of_points,
// 								  no_of_dims,
// 								  m,
// 								  no_of_centroids);
// 	}else if(version == 2){
// 		pointsContainedKernelSharedMemoryFewBank(ceil((no_of_dims)/(float)block_size),
//                                             block_size,
// 										  stream,
// 										  data_d,
// 										  centroids_d,
// 										  dims_d,
// 										  output_d,
// 										  output_count_d,
// 										  width,
// 										  point_dim,
// 										  no_of_points,
// 										  no_of_dims,
// 										  m,
// 										  no_of_centroids);
// 	}else if(version == 3){
// 		pointsContainedKernelSharedMemoryFewerBank(ceil((no_of_dims)/(float)block_size),
//                                                 block_size,
// 												 stream,
// 												 data_d,
// 												 centroids_d,
// 												 dims_d,
// 												 output_d,
// 												 output_count_d,
// 												 width,
// 												 point_dim,
// 												 no_of_points,
// 												 no_of_dims,
// 												 m,
// 												 no_of_centroids);
// 	}else if(version == 4){
// 		whatDataIsInCentroidKernelFewPointsKernel(
// 												  ceil((no_of_dims)/(float)block_size),
// 												  block_size,
// 												  stream,
// 												  output_d,
// 												  data_d,
// 												  centroids_d,
// 												  dims_d,
// 												  width,
// 												  point_dim,
// 												  no_of_points);
// 	}else if(version == 5){
// 		// Call kernel
// 		pointsContainedKernelNaiveBreak(ceil((no_of_dims)/(float)block_size),
// 								   	   block_size,
// 								   	   stream,
// 								   	   data_d,
// 								   	   centroids_d,
// 								   	   dims_d,
// 								   	   output_d,
// 								   	   output_count_d,
// 								   	   width,
// 								   	   point_dim,
// 								   	   no_of_points,
// 								   	   no_of_dims,
// 								   	   m,
// 								   	   breakIntervall);
// 	}else if(version == 6){
// 		// Call kernel
// 		pointsContainedKernelSharedMemoryBreak(ceil((no_of_dims)/(float)block_size),
// 				                        	   block_size,
// 				                        	   stream,
// 				                        	   data_d,
// 				                        	   centroids_d,
// 				                        	   dims_d,
// 				                        	   output_d,
// 				                        	   output_count_d,
// 				                        	   width,
// 				                        	   point_dim,
// 				                        	   no_of_points,
// 				                        	   no_of_dims,
// 				                        	   m,
// 				                        	   no_of_centroids,
// 				                        	   breakIntervall);
// 	}

// 	checkCudaErrors(cudaEventRecord(stop_e, stream));

//     (cudaStreamDestroy(stream));

// 	// copy from device
// 	cudaMemcpy(output_h, output_d, size_of_output, cudaMemcpyDeviceToHost);
// 	cudaMemcpy(output_count_h, output_count_d, size_of_output_count, cudaMemcpyDeviceToHost);

// 	// construnct output
// 	auto output =  new std::vector<std::vector<bool>*>;
// 	auto output_count =  new std::vector<unsigned int>;


// 	for(int i = 0; i < no_of_dims; i++){
// 		auto a =  new std::vector<bool>;
// 		for(int j = 0; j < no_of_points; j++){
// 			a->push_back(output_h[i*no_of_points+j]);
// 		}
// 		output->push_back(a);
// 	}

// 	for(int i = 0; i < no_of_dims; i++){
// 		output_count->push_back(output_count_h[i]);
// 	}


// 	cudaFree(data_d);
// 	cudaFree(dims_d);
// 	cudaFree(centroids_d);
// 	cudaFree(output_d);
// 	cudaFree(output_count_d);
// 	free(data_h);
// 	free(dims_h);
// 	free(centroids_h);
// 	free(output_h);
// 	free(output_count_h);
// 	free(garbageCleaner_h);
// 	free(garbageCleanerCount_h);

// 	//****************************************************************************************************************************

// 	float millisNaive = 0;
// 	cudaEventSynchronize(stop_e);
// 	cudaEventElapsedTime(&millisReducedReads, start_e, stop_e);


// 	Experiment::writeLineToFile(std::to_string(no_of_points) + ", " + std::to_string(point_dim) + ", " + "Naive, " + std::to_string("passed") +", " + std::to_string(millisNaive));
// 	//Experiment::writeLineToFile(std::to_string(numPoints) + ", " + std::to_string(dim) + ", " + "ReducedReads, " + std::to_string(passed) +", " + std::to_string(millisReducedReads));
// 	Experiment::testDone("Dim: " + std::to_string(point_dim) + " numPoints: " + std::to_string(no_of_points));

// 	cudaDeviceReset();
// 	Experiment::stop();
// }
