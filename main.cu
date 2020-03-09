
#include <iostream>
#include <chrono>
#include <random>
#include <iostream>
#include "src/DOC_GPU/DOCGPU.h"
#include "src/Clustering.h"
#include "test/testData.h"
#include <vector>
#include "src/DOC_GPU/DOCGPU_Kernels.h"
#include "src/randomCudaScripts/arrayEqual.h"
#include "src/DOC/HyperCube.h"
int main ()
{
	/*
	const unsigned long maxDim = 100;
	for(unsigned long indexMaxDim = 1 ; indexMaxDim < maxDim ; indexMaxDim++){
		unsigned int rep = 3;
		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
		for(int irep = 0 ; irep < rep ; irep++){
			bool fail = true;
			unsigned int point_dim = indexMaxDim;
			unsigned int no_data = 10000;
			unsigned int no_centroids = 20;
			unsigned int no_dims = 10000;
			unsigned int m = no_dims/no_centroids;
			std::default_random_engine generator;
			generator.seed(100);

			std::uniform_real_distribution<double> distribution(15,20);
			std::uniform_real_distribution<double> distribution2(9,26);


			auto data = new std::vector<std::vector<float>*>;
			auto centroids = new std::vector<unsigned int>;
			auto dims = new std::vector<std::vector<bool>*>;

			for(int i = 0; i < no_data; i++){
				auto point = new std::vector<float>;
				for(int j = 0; j < point_dim; j++){
					point->push_back(distribution2(generator));
				}
				data->push_back(point);
				centroids->push_back(i);
			}
			for(int i = data->size()-1; i < no_data; i++){
				auto point = new std::vector<float>;
				for(int j = 0; j < point_dim; j++){
					point->push_back(distribution(generator));
				}
				data->push_back(point);
			}
			for(int i = 0; i < no_dims; i++){
				auto dim = new std::vector<bool>;
				for(int j = 0; j < point_dim; j++){
					dim->push_back(distribution2(generator)< 13);
				}
				dims->push_back(dim);
			}
			const unsigned long version = 3;
			auto c1 = pointsContained(dims, data, centroids,m,10,version);

			auto c = c1.first;

			int t = 0, f = 0;
			for(int i = 0; i < dims->size(); i++){

				auto centroid = data->at(centroids->at(i/m));
				auto cpu = HyperCube(centroid, 10, dims->at(i));
				for(int j = 0; j < data->size(); j++){
					auto d = cpu.pointContained(data->at(j));
					if(d){
						t++;
					}else{
						f++;
					}
					if(d == c->at(i)->at(j)){
						fail = false;
					}
					//EXPECT_EQ(d,c->at(i)->at(j)) << "i: " << i << " j: " <<j << " : " << data->at(j);
				}
			}
			if(fail){
				//std::cout << "fail " << std::endl;
			}
		}

		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
		//std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::seconds> (end - begin).count()/rep << "[s]" << std::endl;
		//std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count()/rep  << "[ms]" << std::endl;
		//std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/rep << "[µs]" << std::endl;
		//std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count()/rep << "[ns]" << std::endl;
		std::cout << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count()/rep  << std::endl;

	}
	*/

	/*
	unsigned int rep = 1;
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	for(int irep = 0 ; irep < rep ; irep++){

		std::vector<std::vector<float>*>* data = new std::vector<std::vector<float>*>;
		std::default_random_engine generator;
		generator.seed(99);
		std::uniform_real_distribution<double> distribution(50000.0,500000.0);
		std::uniform_real_distribution<double> distribution2(5.0,15.0);
		const unsigned long numberOfDimensions =6;
		for(float i = 0; i < 100; i++){
			std::vector<float>* point1 = new std::vector<float>;
			point1->push_back(distribution2(generator));
			point1->push_back(distribution2(generator));
			point1->push_back(distribution2(generator));
			for(int j = 0; j < numberOfDimensions-3; j++){
				point1->push_back(distribution(generator));
			}
			data->push_back(point1);
		}

		for(float i = 0; i < 1000; i++){
			std::vector<float>* point1 = new std::vector<float>;
			for(int j = 0; j < numberOfDimensions; j++){
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

		if(res.first->size() != 1100 or res.second->size() != 6){
			std::cout << "fail" << std::endl;
		}
	}
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::seconds> (end - begin).count()/rep << "[s]" << std::endl;
	std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count()/rep << "[ms]" << std::endl;
	std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/rep << "[µs]" << std::endl;
	std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count()/rep << "[ns]" << std::endl;
	*/

	/*
	std::vector<std::vector<float>*>* data = new std::vector<std::vector<float>*>;
		{
			auto point = new std::vector<float>{10,10};
			data->push_back(point);
		}
		{
			auto point = new std::vector<float>{0,10};
			data->push_back(point);
		}
		{
			auto point = new std::vector<float>{10,0};
			data->push_back(point);
		}
		{
			auto point = new std::vector<float>{0,0};
			data->push_back(point);
		}
		DOCGPU d = DOCGPU(data);
		std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> res = d.findCluster();

		if(!res.second->at(1)){
			std::cout << "fail" << std::endl;
		}
	*/


	unsigned int point_dim = 4;
	unsigned int no_data = 3000;
	unsigned int no_centroids = 20;
	unsigned int no_dims = 1000;
	unsigned int m = no_dims/no_centroids;
	std::default_random_engine generator;
	generator.seed(100);

	std::uniform_real_distribution<double> distribution(15,20);
	std::uniform_real_distribution<double> distribution2(9,26);


	auto data = new std::vector<std::vector<float>*>;
	auto centroids = new std::vector<unsigned int>;
	auto dims = new std::vector<std::vector<bool>*>;

	for(int i = 0; i < no_data; i++){
		auto point = new std::vector<float>;
		for(int j = 0; j < point_dim; j++){
			point->push_back(distribution2(generator));
		}
		data->push_back(point);
		centroids->push_back(i);
	}
	for(int i = data->size()-1; i < no_data; i++){
		auto point = new std::vector<float>;
		for(int j = 0; j < point_dim; j++){
			point->push_back(distribution(generator));
		}
		data->push_back(point);
	}
	for(int i = 0; i < no_dims; i++){
		auto dim = new std::vector<bool>;
		for(int j = 0; j < point_dim; j++){
			dim->push_back(distribution2(generator)< 13);
		}
		dims->push_back(dim);
	}
	auto c0 = pointsContained(dims, data, centroids,m,10,0);
	std::cout << "hello " << std::endl;
	auto c1 = pointsContained(dims, data, centroids,m,10,1);
	auto c2 = pointsContained(dims, data, centroids,m,10,2);
	auto c3 = pointsContained(dims, data, centroids,m,10,3);


	if(not areTheyEqual_h(c0,c1,false)){
		std::cout << "fail" << std::endl;
	}

	if(not areTheyEqual_h(c0,c2,false)){
		std::cout << "fail" << std::endl;
	}

	if(not areTheyEqual_h(c0,c3,false)){
		std::cout << "fail" << std::endl;
	}

	//EXPECT_TRUE(areTheyEqual_h(c1,c0));
	//EXPECT_TRUE(areTheyEqual_h(c2,c0));
	//EXPECT_TRUE(areTheyEqual_h(c3,c0));

	//printPair_h(c0,100);
	//std::cout << "other" << std::endl;
	//printPair_h(c1,100);




	auto c = c1.first;

	int t = 0, f = 0;
	for(int i = 0; i < dims->size(); i++){

		auto centroid = data->at(centroids->at(i/m));
		auto cpu = HyperCube(centroid, 10, dims->at(i));
		for(int j = 0; j < data->size(); j++){
			auto d = cpu.pointContained(data->at(j));
			if(d){
				t++;
			}else{
				f++;
			}
			///EXPECT_EQ(d,c->at(i)->at(j)) << "i: " << i << " j: " <<j << " : " << data->at(j);
		}
	}

}
