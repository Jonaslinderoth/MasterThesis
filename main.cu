#include <iostream>
#include <chrono>
#include <random>
#include <iostream>
#include "src/DOC_GPU/DOCGPU.h"
#include "src/Fast_DOCGPU/Fast_DOCGPU.h"
#include "src/Clustering.h"
#include "test/testData.h"
#include <vector>
#include <assert.h>
#include "src/dataReader/Cluster.h"
#include "src/MineClusGPU/MineClusGPU.h"
#include "src/Fast_DOC/Fast_DOC.h"
#include "src/Fast_DOCGPU/Fast_DOCGPU.h"




int main ()
{
	unsigned int dim = 200;
	unsigned int numPoints = 1000000;
	auto data =  new std::vector<std::vector<float>*>();

	std::default_random_engine generator;
	generator.seed(1);
	std::normal_distribution<float> cluster1(5.0, 2.0);
	std::normal_distribution<float> cluster2(500.0,2.0);
	std::uniform_int_distribution<> outlier(-10000,10000);

	
	for(int i = 0; i < numPoints/2; i++){
		auto point = new std::vector<float>;
		for(int j = 0; j < dim; j++){
			if(j%5 == 0){
				point->push_back(cluster1(generator));				
			}else{
				point->push_back(outlier(generator));
			}
		}
		data->push_back(point);
	}
	std::cout << "first cluster generated" << std::endl;

	for(int i = data->size(); i < numPoints; i++){
		auto point = new std::vector<float>;
		for(int j = 0; j < dim; j++){
			if(j%10 == 0){
				point->push_back(cluster1(generator));				
			}else{
				point->push_back(outlier(generator));
			}
		}
		data->push_back(point);
	}
	std::cout << "data generated" << std::endl;	

	auto gpu = new Fast_DOCGPU(data);
	gpu->setSeed(1);

	auto res_gpu = gpu->findKClusters(2);

	assert(res_gpu.size() == 2);

	assert(res_gpu.at(0).first->size() == numPoints/2);
	assert(res_gpu.at(0).second->size() == dim);
		
	assert(res_gpu.at(1).first->size() == 50);
	assert(res_gpu.at(1).second->size() == numPoints/2);
}