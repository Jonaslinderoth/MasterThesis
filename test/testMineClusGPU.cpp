#include <gtest/gtest.h>
#include "testingTools.h"
#include <random>
#include "../src/MineClusGPU/MineClusGPU.h"
#include <vector>



TEST(testMineClusGPU, testSetup){
	auto data = new std::vector<std::vector<float>*>;
	auto point1 = new std::vector<float>({1,2});
	data->push_back(point1);

	for(int i = 0; i < 20; i++){
		auto point = new std::vector<float>({1000,20000});
		data->push_back(point);
	}

	
	auto c = MineClusGPU(data);


	auto res = c.findKClusters(1);

	EXPECT_EQ(res.size(), 1);
	EXPECT_EQ(res.at(0).second->at(0), 1);
	EXPECT_EQ(res.at(0).second->at(1), 1);
	
}


TEST(testMineClusGPU, testSetup2){
	auto data = new std::vector<std::vector<float>*>;
	auto point1 = new std::vector<float>({1,2});
	data->push_back(point1);

	for(int i = 0; i < 20; i++){
		auto point = new std::vector<float>({1000,2});
		data->push_back(point);
	}

	
	auto c = MineClusGPU(data);


	auto res = c.findKClusters(1);

	EXPECT_EQ(res.size(), 1);
	EXPECT_EQ(res.at(0).second->at(0), 1);
	EXPECT_EQ(res.at(0).second->at(1), 1);
	
}


TEST(testMineClusGPU, testSetup3){
	auto data = new std::vector<std::vector<float>*>;
	auto point1 = new std::vector<float>({1,2});
	data->push_back(point1);

	for(int i = 0; i < 20; i++){
		auto point = new std::vector<float>({1,2});
		data->push_back(point);
	}

	
	auto c = MineClusGPU(data);


	auto res = c.findKClusters(1);

	EXPECT_EQ(res.size(), 1);
	EXPECT_EQ(res.at(0).second->at(0), 1);
	EXPECT_EQ(res.at(0).second->at(1), 1);
	
}
