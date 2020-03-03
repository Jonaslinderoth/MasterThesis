
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
	unsigned int rep = 1;
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	for(int irep = 0 ; irep < rep ; irep++){
		std::vector<std::vector<float>*>* data = new std::vector<std::vector<float>*>;
		std::default_random_engine generator;
		generator.seed(100);
		std::uniform_real_distribution<double> distribution(50000.0,500000.0);

		for(float i = 0; i < 20; i++){
			std::vector<float>* point1 = new std::vector<float>{i};
			for(int i = 0; i < 5; i++){
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
		/*
		SUCCEED();
		EXPECT_EQ(res.second->size(),6);
		EXPECT_TRUE(res.second->at(0));
		for(int i = 1; i < 6;i++){
			EXPECT_FALSE(res.second->at(i));
		}

		EXPECT_EQ(res.first->size(), 20);
		*/
		if(res.second->size() != 6 or res.first->size() != 20){
			std::cout << "is 6: " << res.second->size() << std::endl;
			std::cout << "is 20: " << res.first->size() << std::endl;
		}
	}
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/rep << "[µs]" << std::endl;
	std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count()/rep << "[ns]" << std::endl;
	/*
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
	*/
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
