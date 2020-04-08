#include <iostream>
#include <chrono>
#include <random>
#include <iostream>
#include "src/DOC_GPU/DOCGPU.h"
#include "src/Fast_DOCGPU/Fast_DOCGPU.h"
#include "src/Clustering.h"
#include "test/testData.h"
#include <vector>
#include <random>
#include "src/dataReader/Cluster.h"
#include "src/MineClusGPU/MineClusGPU.h"




int main ()
{
	std::default_random_engine gen;
	gen.seed(0);
	std::normal_distribution<float> cluster1(100.0,2.0);
	std::normal_distribution<float> cluster2(1000.0,2.0);
	std::normal_distribution<float> noise(100.0,200000.0);
	unsigned int dim = 100;
	auto data = new std::vector<std::vector<float>*>;
	{
		for(int i = 0; i < 300; i++){
			auto point = new std::vector<float>;
			for(int j = 0; j < dim; j++){
				if(j % 8 == 0){
					point->push_back(cluster1(gen));					
				}else{
					point->push_back(noise(gen));					
				}

			}
			data->push_back(point);
		}
	}


	{
		for(int i = 0; i < 100; i++){
			auto point = new std::vector<float>;
			for(int j = 0; j < dim; j++){
				if(j % 11 == 0){
					point->push_back(cluster2(gen));					
				}else{
					point->push_back(noise(gen));					
				}

			}
			data->push_back(point);
		}
	}
	
	auto c = MineClusGPU(data);
	c.setSeed(2);


	auto res = c.findKClusters(1);


	for(int i = 0; i < res.size(); i++){
		for(int j = 0; j < res.at(i).second->size(); j++){
			std::cout << res.at(i).second->at(j);
		}
		std::cout << std::endl;
	}
}