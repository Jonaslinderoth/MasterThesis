
#include <iostream>
#include <chrono>
#include <random>
#include <iostream>
#include "src/DOC_GPU/DOCGPU.h"
#include "src/Clustering.h"
#include "test/testData.h"
#include <vector>
int main ()
{
	
	std::vector<std::vector<float>*>* data = new std::vector<std::vector<float>*>;
	std::default_random_engine generator;
	generator.seed(100);
	std::uniform_real_distribution<double> distribution(50000.0,500000.0);
	std::uniform_real_distribution<double> distribution2(5.0,15.0);

	for(float i = 0; i < 100; i++){
		std::vector<float>* point1 = new std::vector<float>;
		point1->push_back(distribution2(generator));
		point1->push_back(distribution2(generator));
		point1->push_back(distribution2(generator));
		for(int j = 0; j < 3; j++){
			point1->push_back(distribution(generator));
		}
		data->push_back(point1);
	}

	for(float i = 0; i < 1000; i++){
		std::vector<float>* point1 = new std::vector<float>;
		for(int j = 0; j < 6; j++){
			point1->push_back(distribution(generator));
		}
		data->push_back(point1);
	}
	
	DOCGPU d = DOCGPU(data);
	d.setSeed(1);
	d.setAlpha(0.1);
	d.setBeta(0.25);
	d.setWidth(15);
	auto res = d.findCluster();
	
	std::cout <<"is 6?: " << res.second->size() << std::endl;
	std::cout << " is 1100?: " << res.first->size() << std::endl;
	/*EXPECT_EQ(res.first->size(), 100);
	for(int i = 0; i < res.first->size(); i++){
		for(int j = 0; j < 6; j++){
			std::cout << res.first->at(i)->at(j) << ", ";
		}
		std::cout <<std::endl;
	}*/
}
