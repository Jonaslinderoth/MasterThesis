
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
	std::uniform_real_distribution<double> distribution2(5.0,20.0);

	for(float i = 0; i < 20; i++){
		std::vector<float>* point1 = new std::vector<float>{i};
		for(int i = 0; i < 5; i++){
			point1->push_back(distribution(generator));
		}
		point1->push_back(distribution2(generator));
		data->push_back(point1);

	}

	DOCGPU d = DOCGPU(data);
	d.setSeed(1);
	d.setAlpha(0.1);
	d.setBeta(0.25);
	d.setWidth(15);
	auto res = d.findCluster();

	std::cout << "is 7: " << res.second->size() << std::endl;
	std::cout << "is 20: " << res.first->size() << std::endl;

	/*
	std::vector<std::vector<float>*>* data = new std::vector<std::vector<float>*>;
		{
			auto point = new std::vector<float>{1,1000};
			data->push_back(point);
		}
		{
			auto point = new std::vector<float>{0,100};
			data->push_back(point);
		}
		{
			auto point = new std::vector<float>{1,-100};
			data->push_back(point);
		}
		{
			auto point = new std::vector<float>{0,-1000};
			data->push_back(point);
		}
		DOCGPU d = DOCGPU(data);
		d.setSeed(2);
		std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> res = d.findCluster();
		std::cout << "is 4: " << res.first->size() << std::endl;

		std::cout << "is 2: " << res.second->size() << std::endl;;
	*/

}
