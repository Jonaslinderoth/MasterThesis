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

	for(int i = 0; i < 5; i++){
		auto point = new std::vector<float>({1,2});
		data->push_back(point);
	}

	
	auto c = MineClusGPU(data);


	auto res = c.findKClusters(1);

	EXPECT_EQ(res.size(), 1);
	EXPECT_EQ(res.at(0).second->at(0), 1);
	EXPECT_EQ(res.at(0).second->at(1), 1);

	EXPECT_EQ(res.at(0).first->size(), 6);
	for(int i = 0; i < 6; i++){
		EXPECT_EQ(res.at(0).first->at(i)->at(0), 1);
		EXPECT_EQ(res.at(0).first->at(i)->at(1), 2);		
	}


	
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
	EXPECT_EQ(res.at(0).second->size(), 2);
	EXPECT_EQ(res.at(0).second->at(0), 0);
	EXPECT_EQ(res.at(0).second->at(1), 1);

	EXPECT_EQ(res.at(0).first->at(0)->at(0), 1);
	EXPECT_EQ(res.at(0).first->at(0)->at(1), 2);
	
	EXPECT_EQ(res.at(0).first->size(), 21);
	for(int i = 1; i < 21; i++){
		EXPECT_EQ(res.at(0).first->at(i)->at(0), 1000);
		EXPECT_EQ(res.at(0).first->at(i)->at(1), 2);		
	}
	
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
