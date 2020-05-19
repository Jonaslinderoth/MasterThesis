#include "PointsContainedDeviceNormalData.h"
#include <random>
#include <string>
#include <iostream>
#include <vector>

#include "../src/randomCudaScripts/Utils.h"
#include "../src/randomCudaScripts/arrayEqual.h"
#include "../src/DOC_GPU/pointsContainedDevice.h"
#include "../src/Fast_DOCGPU/whatDataInCentroid.h"
#include "../src/DOC/DOC.h"
#include "../src/testingTools/DataGeneratorBuilder.h"

std::vector<bool>* PointsContainedDeviceNormalData::findDimensionsEx(std::vector<float>* centroid,
																		   std::vector<std::vector<float>* >* points,
																		   float width) {
	std::vector<bool>* result = new std::vector<bool>(centroid->size(), true);
	for(int i = 0; i < points->size(); i++){
		for(int j = 0; j < centroid->size(); j++){
			result->at(j) = result->at(j) && (abs(centroid->at(j)-points->at(i)->at(j)) < width);
		}
	}
	return result;
}


std::vector<int> PointsContainedDeviceNormalData::randIntVec(unsigned int lower,unsigned int upper,unsigned int n){

	std::vector<int> res = std::vector<int>();
	std::uniform_int_distribution<> dis(lower, upper); //inclusive

	for(int i = 0; i < n; i++){
		auto a = dis(gen);

		res.push_back(a);
	}
	return res;
};

std::vector<std::vector<float>*>* PointsContainedDeviceNormalData::pickRandomPointFromData(std::vector<std::vector<float>*>* data, unsigned int size){
	std::vector<int> is = randIntVec(0,data->size()-1,size);
	std::vector<std::vector<float>*>* res = new std::vector<std::vector<float>*>;
	for(int i = 0; i < size; i++){
		res->push_back(data->at(is.at(i)));
	}
	return res;
}


void PointsContainedDeviceNormalData::start(){
	/*
	 * here we generate some data
	 * the data is going to have 15 relevant dimensions.
	 * there are going to be no outliers
	 */
	gen.seed(0); // important to seed
	bool prints = false;
	std::uniform_real_distribution<> dist(0, 100);
	std::uniform_real_distribution<double> distribution2(0,15);
	unsigned int c = 1;
	unsigned long numberOfClusters = 5;
	if(prints){
		numberOfClusters = 2;
	}
	unsigned long numberOfDimensionsUsed = 15;
	if(prints){
		numberOfDimensionsUsed = 1;
	}
	unsigned long size_of_sample = 1024*4;
	if(prints){
		size_of_sample = 3;
	}
	unsigned long breakIntervall = 5;

	struct versionInfo{
		std::string print;
		unsigned int version;
	};
	versionInfo pointsContainedKernelNaiveVI = {"pointsContainedKernelNaive",0};
	versionInfo pointsContainedKernelSharedMemoryVI = {"pointsContainedKernelSharedMemory",1};
	versionInfo pointsContainedKernelSharedMemoryFewBankVI = {"pointsContainedKernelSharedMemoryFewBank",2};
	versionInfo pointsContainedKernelSharedMemoryFewerBankVI = {"pointsContainedKernelSharedMemoryFewerBank",3};
	versionInfo whatDataIsInCentroidKernelFewPointsKernelVI = {"whatDataIsInCentroidKernelFewPointsKernel",4};
	versionInfo pointsContainedKernelNaiveBreakVI = {"pointsContainedKernelNaiveBreak",5};
	versionInfo pointsContainedKernelSharedMemoryBreakVI = {"pointsContainedKernelSharedMemoryBreak",6};

	std::vector<versionInfo> versionVec;
	versionVec.push_back(pointsContainedKernelNaiveVI);
	versionVec.push_back(pointsContainedKernelSharedMemoryVI);
	versionVec.push_back(pointsContainedKernelSharedMemoryFewBankVI);
	versionVec.push_back(pointsContainedKernelSharedMemoryFewerBankVI);
	versionVec.push_back(whatDataIsInCentroidKernelFewPointsKernelVI);
	versionVec.push_back(pointsContainedKernelNaiveBreakVI);
	versionVec.push_back(pointsContainedKernelSharedMemoryBreakVI);


	c = versionVec.size();

	Experiment::addTests(c);

	Experiment::start();


	int block_size = 1024;
	unsigned long garbage = 0;
	unsigned long width = 15;
	// Calculaating sizes
	std::size_t point_dim = 200;
	if(prints){
		 point_dim = 2;
	}
	std::size_t no_of_points = 20000; //per cluster and NEED to be dividable by the clusteres
	if(prints){
		no_of_points = 20;
	}
	std::size_t no_of_dims = 1024; //idk
	std::size_t no_of_centroids = 2;
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

	//making the data array
	DataGeneratorBuilder dgb;

	std::string fineName = "dataExperimentBreakingIntervalNormalData";
	bool res = dgb.buildUClusters(fineName,no_of_points/numberOfClusters,numberOfClusters,width,point_dim,numberOfDimensionsUsed,0);

	DataReader* dr = new DataReader(fineName);
	/*
	std::cout << "data from bgd" << std::endl;
	while(dr->isThereANextPoint()){
		std::vector<float>* point = dr->nextPoint();
		for(std::vector<float>::iterator iter = point->begin() ; iter != point->end() ; ++iter){
			std::cout << std::to_string(*iter) << " ";
		}
		std::cout << std::endl;
	}*/
	if(prints){
		std::cout << "filling data" << std::endl;
	}

	// filling data
	std::vector<std::vector<float>*>* data = new std::vector<std::vector<float>*>;
	for(int i= 0; i < no_of_points; i++){

		if(not dr->isThereANextPoint()){
			if(prints){
				std::cout << "Error data generated is not right size" << std::endl;
			}else{
				this->repportError("Error data generated is not right size", this->getName());
			}
		}

		std::vector<float>* point = dr->nextPoint();
		if(point->size() != point_dim){
			if(prints){
				std::cout << "Error data generated is not right dimension" << std::endl;
			}else{
				this->repportError("Error data generated is not right dimension", this->getName());
			}

		}
		for(int j = 0; j < point_dim; j++){
			data_h[i*point_dim+j] = point->at(j);

		}
		data->push_back(point);
	}


	//now i need to make the dims array
	//to do that i need a sample and use stuff from the CPU version of doc
	if(prints){
		std::cout << "dims being made" << std::endl;
	}
	unsigned int medoidIndex = 0;
	centroids_h[medoidIndex] = randIntVec(0,no_of_points,1).at(0);
	std::vector<float>* mediod = data->at(centroids_h[0]);
	//first centroid
	if(mediod->size() != point_dim){
		this->repportError("Error mediod->size() != point_dim", this->getName());
	}


	for(unsigned int sampleIndex = 0; sampleIndex < no_of_dims; ++sampleIndex){

		if(sampleIndex != 0 and sampleIndex/m != (sampleIndex-1)/m){

			medoidIndex++;
			centroids_h[medoidIndex] = randIntVec(0,no_of_points,1).at(0);
			mediod = data->at(centroids_h[medoidIndex]);

		}
		std::vector<std::vector<float>*>* sample = pickRandomPointFromData(data,size_of_sample);
		std::vector<bool>* D = findDimensionsEx(mediod,sample,width);
		if(D->size() != point_dim){
			this->repportError("Error D->size() != point_dim", this->getName());
		}
		for(unsigned long dimIndex = 0; dimIndex < D->size(); ++dimIndex){
			dims_h[sampleIndex*point_dim+dimIndex] = D->at(dimIndex);
		}
	}
	if(prints){
		std::cout << "data" << std::endl;
		for(unsigned int pointIndex = 0 ;pointIndex < no_of_points ; ++pointIndex){
			for(unsigned int dimIndex = 0 ; dimIndex < point_dim; ++dimIndex){
				std::cout << data_h[pointIndex*point_dim+dimIndex] << " ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
		if(prints){
			std::cout << "medoid index" << std::endl;
		}
		for(unsigned int medoidIndex = 0; medoidIndex < no_of_centroids; ++medoidIndex){
			std::cout << centroids_h[medoidIndex] << " ";
		}
		std::cout << std::endl;;

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



	for(std::vector<versionInfo>::iterator iter = versionVec.begin() ; iter != versionVec.end(); ++iter){
		//time taking
		cudaEvent_t start_naive, stop_naive;
		checkCudaErrors(cudaEventCreate(&start_naive));
		checkCudaErrors(cudaEventCreate(&stop_naive));
		checkCudaErrors(cudaEventRecord(start_naive, stream));

		if(iter->version == 0){
			// Call kernel

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


		}else if(iter->version == 1){
			pointsContainedKernelSharedMemory(ceil((no_of_dims)/(float)block_size),
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
									  no_of_centroids);
		}else if(iter->version == 2){
			pointsContainedKernelSharedMemoryFewBank(ceil((no_of_dims)/(float)block_size),
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
											  no_of_centroids);
		}else if(iter->version == 3){
			pointsContainedKernelSharedMemoryFewerBank(ceil((no_of_dims)/(float)block_size),
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
													 no_of_centroids);
		}else if(iter->version == 4){
			whatDataIsInCentroidKernelFewPointsKernel(
													  ceil((no_of_dims)/(float)block_size),
													  block_size,
													  stream,
													  output_d,
													  data_d,
													  centroids_d,
													  dims_d,
													  width,
													  point_dim,
													  no_of_points);
		}else if(iter->version == 5){
			// Call kernel
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
		}else if(iter->version == 6){
			// Call kernel
			pointsContainedKernelSharedMemoryBreak(ceil((no_of_dims)/(float)block_size),
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
					                        	   no_of_centroids,
					                        	   breakIntervall);
		}
		checkCudaErrors(cudaEventRecord(stop_naive, stream));

		float millisReducedReadsNaive = 0;
		checkCudaErrors(cudaEventSynchronize(stop_naive));

		checkCudaErrors(cudaEventElapsedTime(&millisReducedReadsNaive, start_naive, stop_naive));
		if(not prints)
		{
			std::string bi = std::to_string(breakIntervall);
			std::string text = std::to_string(no_of_points) + ", " + std::to_string(point_dim) + ", " + iter->print + ", " + bi +" , " + std::to_string(millisReducedReadsNaive);
			Experiment::writeLineToFile(text);
			Experiment::testDone(iter->print);

		}

	}



	checkCudaErrors((cudaStreamDestroy(stream)));

	// copy from device
	checkCudaErrors(cudaMemcpy(output_h, output_d, size_of_output, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(output_count_h, output_count_d, size_of_output_count, cudaMemcpyDeviceToHost));

	// construnct output
	auto output =  new std::vector<std::vector<bool>*>;
	auto output_count =  new std::vector<unsigned int>;

	if(prints){
		std::cout << "bool output" << std::endl;
	}
	for(int i = 0; i < no_of_dims; i++){
		auto a =  new std::vector<bool>;
		for(int j = 0; j < no_of_points; j++){
			a->push_back(output_h[i*no_of_points+j]);
			if(prints){
				std::cout << output_h[i*no_of_points+j] << " ";
			}
		}
		output->push_back(a);
		if(prints){
			std::cout << std::endl;
		}
	}
	if(prints){
		std::cout << std::endl;
		std::cout << std::endl;
		std::cout << "output count" << std::endl;
	}
	for(int i = 0; i < no_of_dims; i++){
		output_count->push_back(output_count_h[i]);
		if(prints){
			std::cout << output_count_h[i] << " ";
		}
	}
	if(prints){
		std::cout << std::endl;
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
