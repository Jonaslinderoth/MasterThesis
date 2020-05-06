#include <gtest/gtest.h>
#include "../src/MineClusGPU/MineClusKernels.h"
#include "testingTools.h"




TEST(testPointsContainedMineClus, testSimple){
	auto candidate = std::vector<bool>({1,1});
	auto data = new std::vector<std::vector<float>*>;
	auto centroid = 0;
	auto width = 10;

	{
		auto point = new std::vector<float>;
		point->push_back(1);
		point->push_back(2);
		data->push_back(point);
	}

	{
		auto point = new std::vector<float>;
		point->push_back(1000);
		point->push_back(2000);
		data->push_back(point);
	}

	auto result = findPointsInClusterTester(candidate, data, centroid, width);

	EXPECT_EQ(result.at(0), 1);
	EXPECT_EQ(result.at(1), 0);
}


TEST(testPointsContainedMineClus, testSimple2){
	auto candidate = std::vector<bool>({1,0});
	auto data = new std::vector<std::vector<float>*>;
	auto centroid = 0;
	auto width = 10;

	{
		auto point = new std::vector<float>;
		point->push_back(1);
		point->push_back(2);
		data->push_back(point);
	}

	{
		auto point = new std::vector<float>;
		point->push_back(9);
		point->push_back(2000);
		data->push_back(point);
	}

	auto result = findPointsInClusterTester(candidate, data, centroid, width);

	EXPECT_EQ(result.at(0), 1);
	EXPECT_EQ(result.at(1), 1);
}


TEST(testPointsContainedMineClus, testSimple3){
	auto candidate = std::vector<bool>({0,1});
	auto data = new std::vector<std::vector<float>*>;
	auto centroid = 0;
	auto width = 10;

	{
		auto point = new std::vector<float>;
		point->push_back(1);
		point->push_back(2);
		data->push_back(point);
	}

	{
		auto point = new std::vector<float>;
		point->push_back(9);
		point->push_back(2000);
		data->push_back(point);
	}

	auto result = findPointsInClusterTester(candidate, data, centroid, width);

	EXPECT_EQ(result.at(0), 1);
	EXPECT_EQ(result.at(1), 0);
}

TEST(testPointsContainedMineClus, test41Points){
	auto candidate = std::vector<bool>({1,0});
	auto data = new std::vector<std::vector<float>*>;
	auto point1 = new std::vector<float>({1,2});
	data->push_back(point1);

	for(int i = 0; i < 20; i++){
		auto point = new std::vector<float>({1000,20000});
		data->push_back(point);
	}
	
	for(int i = 0; i < 20; i++){
		auto point = new std::vector<float>({1,20000});
		data->push_back(point);
	}

	auto res = findPointsInClusterTester(candidate, data, 0, 10);

	for(int i = 0; i < res.size(); i++){
		std::cout << res.at(i);
	}
	std::cout << std::endl;
	EXPECT_EQ(res.size(), 41);
	EXPECT_EQ(res.at(0),1);
	for(int i = 0; i < 20; i++){
		EXPECT_EQ(res.at(i+1),0);		
	}
	for(int i = 0; i < 20; i++){
		EXPECT_EQ(res.at(i+21),1);		
	}	
}
