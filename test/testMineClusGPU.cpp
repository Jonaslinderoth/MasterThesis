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



TEST(testMineClusGPU, test3dims_1){
	auto data = new std::vector<std::vector<float>*>;
	auto point1 = new std::vector<float>({1,2,3});
	data->push_back(point1);

	for(int i = 0; i < 20; i++){
		auto point = new std::vector<float>({1,2,3});
		data->push_back(point);
	}

	
	auto c = MineClusGPU(data);


	auto res = c.findKClusters(1);

	EXPECT_EQ(res.size(), 1);
	EXPECT_EQ(res.at(0).second->size(), 3);
	EXPECT_EQ(res.at(0).second->at(0), 1);
	EXPECT_EQ(res.at(0).second->at(1), 1);
	EXPECT_EQ(res.at(0).second->at(2), 1);
}

TEST(testMineClusGPU, test3dims_2){
	auto data = new std::vector<std::vector<float>*>;
	auto point1 = new std::vector<float>({1,2,3});
	data->push_back(point1);

	for(int i = 0; i < 20; i++){
		auto point = new std::vector<float>({1,2,99999});
		data->push_back(point);
	}

	
	auto c = MineClusGPU(data);


	auto res = c.findKClusters(1);

	EXPECT_EQ(res.size(), 1);
	EXPECT_EQ(res.at(0).second->size(), 3);
	EXPECT_EQ(res.at(0).second->at(0), 1);
	EXPECT_EQ(res.at(0).second->at(1), 1);
	EXPECT_EQ(res.at(0).second->at(2), 0);

	EXPECT_EQ(res.at(0).first->size(), 21);	

	
}

TEST(testMineClusGPU, test10Dims){
	auto data = new std::vector<std::vector<float>*>;
	auto point1 = new std::vector<float>({1,2,3,4,5,6,7,8,9,10});
	data->push_back(point1);

	for(int i = 0; i < 20; i++){
		auto point = new std::vector<float>({1,2,3,999,5,6,7,8,999,10});
		data->push_back(point);
	}

	for(int i = 0; i < 20; i++){
		auto point = new std::vector<float>({-999,2,333,999,5,666,7,8,999,10});
		data->push_back(point);
	}
	
	auto c = MineClusGPU(data);


	auto res = c.findKClusters(1);

	EXPECT_EQ(res.size(), 1);
	EXPECT_EQ(res.at(0).second->size(), 10);
	EXPECT_EQ(res.at(0).second->at(0), 1);
	EXPECT_EQ(res.at(0).second->at(1), 1);
	EXPECT_EQ(res.at(0).second->at(2), 1);
	EXPECT_EQ(res.at(0).second->at(3), 0);
	EXPECT_EQ(res.at(0).second->at(4), 1);
	EXPECT_EQ(res.at(0).second->at(5), 1);
	EXPECT_EQ(res.at(0).second->at(6), 1);
	EXPECT_EQ(res.at(0).second->at(7), 1);
	EXPECT_EQ(res.at(0).second->at(8), 0);
	EXPECT_EQ(res.at(0).second->at(9), 1);


	EXPECT_EQ(res.at(0).first->size(), 21);
	EXPECT_EQ(res.at(0).first->at(0)->at(0), 1);
	EXPECT_EQ(res.at(0).first->at(0)->at(1), 2);
	EXPECT_EQ(res.at(0).first->at(0)->at(2), 3);
	EXPECT_EQ(res.at(0).first->at(0)->at(3), 4);
	EXPECT_EQ(res.at(0).first->at(0)->at(4), 5);
	EXPECT_EQ(res.at(0).first->at(0)->at(5), 6);
	EXPECT_EQ(res.at(0).first->at(0)->at(6), 7);
	EXPECT_EQ(res.at(0).first->at(0)->at(7), 8);
	EXPECT_EQ(res.at(0).first->at(0)->at(8), 9);
	EXPECT_EQ(res.at(0).first->at(0)->at(9), 10);

	for(int i = 0; i < 20; i ++){
		EXPECT_EQ(res.at(0).first->at(i+1)->at(0), 1);
		EXPECT_EQ(res.at(0).first->at(i+1)->at(1), 2);
		EXPECT_EQ(res.at(0).first->at(i+1)->at(2), 3);
		EXPECT_EQ(res.at(0).first->at(i+1)->at(3), 999);
		EXPECT_EQ(res.at(0).first->at(i+1)->at(4), 5);
		EXPECT_EQ(res.at(0).first->at(i+1)->at(5), 6);
		EXPECT_EQ(res.at(0).first->at(i+1)->at(6), 7);
		EXPECT_EQ(res.at(0).first->at(i+1)->at(7), 8);
		EXPECT_EQ(res.at(0).first->at(i+1)->at(8), 999);
		EXPECT_EQ(res.at(0).first->at(i+1)->at(9), 10);		
	}



	
							

	
	
}
