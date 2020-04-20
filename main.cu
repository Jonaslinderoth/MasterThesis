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
#include "src/randomCudaScripts/arrayEqual.h"



int main ()
{
	std::mt19937 gen{0};
	gen.seed(2);
	static std::random_device rand;
	std::uniform_int_distribution<int> distSmall(6, 20);
	std::uniform_int_distribution<int> distBig(400, 4000);
	unsigned long small = distSmall(rand);
	unsigned long big = distBig(rand);
	unsigned long point_dim = 2;
	unsigned long no_data = 80;
	unsigned long no_dims = 1;
	unsigned int no_centroids = 20;
	unsigned int m = ceilf((float)no_dims/(float)no_centroids);
	std::default_random_engine generator;
	generator.seed(100);

	std::uniform_real_distribution<double> distribution(15,20);
	std::uniform_real_distribution<double> distribution2(9,26);


	auto data = new std::vector<std::vector<float>*>;
	auto centroids = new std::vector<unsigned int>;
	auto dims = new std::vector<std::vector<bool>*>;

	for(int i = 0; i < no_data/2; i++){
		auto point = new std::vector<float>;
		for(int j = 0; j < point_dim; j++){
			point->push_back(distribution2(generator));
		}
		data->push_back(point);
	}

	for(int i = data->size()-1; i < no_data; i++){
	auto point = new std::vector<float>;
	for(int j = 0; j < point_dim; j++){
	point->push_back(distribution(generator));
	}
	data->push_back(point);
	}

	for(unsigned int i = 0; i < no_centroids; i++){
		centroids->push_back(i);
	}



	for(int i = 0; i < no_dims; i++){
		auto dim = new std::vector<bool>;
		for(int j = 0; j < point_dim; j++){
			dim->push_back(distribution2(generator)< 13);
		}
		dims->push_back(dim);
	}
	auto c0 = pointsContained(dims, data, centroids,m,10,0);
	auto c1 = pointsContained(dims, data, centroids,m,10,1);
	auto c2 = pointsContained(dims, data, centroids,m,10,2);
	auto c3 = pointsContained(dims, data, centroids,m,10,3);
	auto c4 = pointsContained(dims, data, centroids,m,10,4);


	if(!areTheyEqual_h(c1,c0)){
		std::cout << "c1 point_dim: " << point_dim << " no_dims " << no_dims << " no_data " << no_data << std::endl;
	}
	if(!areTheyEqual_h(c2,c0)){
		std::cout << "c2 point_dim: " << point_dim << " no_dims " << no_dims << " no_data " << no_data << std::endl;
	}
	if(!areTheyEqual_h(c3,c0)){
		std::cout  << "c3 point_dim: " << point_dim << " no_dims " << no_dims << " no_data " << no_data << std::endl;
	}
	if(!areTheyEqual_h(c4,c0)){
		std::cout << "c4 point_dim: " << point_dim << " no_dims " << no_dims << " no_data " << no_data << std::endl;
	}
	/*
	std::cout << "c4" << std::endl;
	printPair_h(c4,1000);
	std::cout << "c0" << std::endl;
	printPair_h(c0,1000);
	std::cout << "c1" << std::endl;
	printPair_h(c1,1000);
	*/
	for(int i = 0; i < c1.first->size(); i++){
		delete c1.first->at(i);
	}
	delete c1.first;
	delete c1.second;
	for(int i = 0; i < c2.first->size(); i++){
		delete c2.first->at(i);
	}
	delete c2.first;
	delete c2.second;
	delete c3.first;
	delete c3.second;
	delete c4.first;
	delete c4.second;

	for(int i = 0; i < data->size(); i++){
		delete data->at(i);
	}
	delete data;

	delete centroids;
	for(int i = 0; i < dims->size(); i++){
		delete dims->at(i);
	}
	delete dims;

	//EXPECT_TRUE(areTheyEqual_h(c3,c0)) << " point_dim: " << point_dim << " no_dims " << no_dims << " no_data " << no_data << std::endl;



}
