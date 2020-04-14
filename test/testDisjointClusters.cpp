#include <gtest/gtest.h>
#include "testingTools.h"
#include <random>
#include "../src/MineClus/MineClusKernels.h"
#include <vector>


TEST(testDisjointClusters, testSetup){
	auto data = new std::vector<std::vector<float>*>;
	auto centroids = std::vector<unsigned int>{0,1};
	auto subspaces = std::vector<unsigned int>{5, 2}; // 101, 010
	auto scores = std::vector<float>{100,200};

	{
		auto point = new std::vector<float>{1,2,3};
		data->push_back(point);
	}

	{
		auto point = new std::vector<float>{100,200,300};
		data->push_back(point);
	}

	auto res = disjointClustersTester(data, centroids, subspaces, scores);

	EXPECT_EQ(res.size(),2);
	EXPECT_EQ(res.at(0),0);
	EXPECT_EQ(res.at(1),1);	
}

TEST(testDisjointClusters, testSimple){
	auto data = new std::vector<std::vector<float>*>;
	auto centroids = std::vector<unsigned int>{0,0};
	auto subspaces = std::vector<unsigned int>{5, 2}; // 101, 010
	auto scores = std::vector<float>{200,100};

	{
		auto point = new std::vector<float>{1,2,3};
		data->push_back(point);
	}

	{
		auto point = new std::vector<float>{100,200,300};
		data->push_back(point);
	}

	auto res = disjointClustersTester(data, centroids, subspaces, scores);

	EXPECT_EQ(res.size(),2);
	EXPECT_EQ(res.at(0),1);
	EXPECT_EQ(res.at(1),0);	
}



TEST(testDisjointClusters, testSimple2){
	auto data = new std::vector<std::vector<float>*>;
	auto centroids = std::vector<unsigned int>{0,1};
	auto subspaces = std::vector<unsigned int>{5, 5}; // 101, 101
	auto scores = std::vector<float>{100,200};

	{
		auto point = new std::vector<float>{1,2,3};
		data->push_back(point);
	}

	{
		auto point = new std::vector<float>{100,200,300};
		data->push_back(point);
	}

	auto res = disjointClustersTester(data, centroids, subspaces, scores);

	EXPECT_EQ(res.size(),2);
	EXPECT_EQ(res.at(0),1);
	EXPECT_EQ(res.at(1),1);	
}


TEST(testDisjointClusters, testSimple3){
	auto data = new std::vector<std::vector<float>*>;
	auto centroids = std::vector<unsigned int>{0,0};
	auto subspaces = std::vector<unsigned int>{5, 5}; // 101, 101
	auto scores = std::vector<float>{100,200};

	{
		auto point = new std::vector<float>{1,2,3};
		data->push_back(point);
	}

	{
		auto point = new std::vector<float>{100,200,300};
		data->push_back(point);
	}

	auto res = disjointClustersTester(data, centroids, subspaces, scores);

	EXPECT_EQ(res.size(),2);
	EXPECT_EQ(res.at(0),0);
	EXPECT_EQ(res.at(1),1);	
}

TEST(testDisjointClusters, testLarge1){
	auto data = new std::vector<std::vector<float>*>;
	auto centroids = std::vector<unsigned int>{0,0,0,0,0,0,0,0,0,0};
	auto subspaces = std::vector<unsigned int>(10,2836467999); // 1010 1001 0001 0001 0001 0001 1111
	auto scores = std::vector<float>{200,200,300,400,500,6000,7000,99,919,999}; 

	{
		auto point = new std::vector<float>;
		for(int i = 0; i < 32; i++){
			point->push_back(i);
		}
		data->push_back(point);
	}

	auto res = disjointClustersTester(data, centroids, subspaces, scores);

	EXPECT_EQ(res.size(),10);
	EXPECT_EQ(res.at(0),0);
	EXPECT_EQ(res.at(1),0);
	EXPECT_EQ(res.at(2),0);
	EXPECT_EQ(res.at(3),0);
	EXPECT_EQ(res.at(4),0);
	EXPECT_EQ(res.at(5),0);
	EXPECT_EQ(res.at(6),1);
	EXPECT_EQ(res.at(7),0);
	EXPECT_EQ(res.at(8),0);
	EXPECT_EQ(res.at(9),0);
								
}


TEST(testDisjointClusters, testLarge2){
	auto data = new std::vector<std::vector<float>*>;
	auto centroids = std::vector<unsigned int>{0,0,0,0,0,0,0,0,0,0,0};
	auto subspaces = std::vector<unsigned int>(); // 1010 1001 0001 0001 0001 0001 1111

	for(int i = 0; i < 11; i++){
		subspaces.push_back(2836467999);
		subspaces.push_back(1);
	}
	
	auto scores = std::vector<float>{200,200,300,400,500,6000,7000,99,919,999,0}; 

	{
		auto point = new std::vector<float>;
		for(int i = 0; i < 33; i++){
			point->push_back(i);
		}
		data->push_back(point);
	}

	auto res = disjointClustersTester(data, centroids, subspaces, scores);

	EXPECT_EQ(res.size(),11);
	EXPECT_EQ(res.at(0),0);
	EXPECT_EQ(res.at(1),0);
	EXPECT_EQ(res.at(2),0);
	EXPECT_EQ(res.at(3),0);
	EXPECT_EQ(res.at(4),0);
	EXPECT_EQ(res.at(5),0);
	EXPECT_EQ(res.at(6),1);
	EXPECT_EQ(res.at(7),0);
	EXPECT_EQ(res.at(8),0);
	EXPECT_EQ(res.at(9),0);
	EXPECT_EQ(res.at(10),0);
}								


TEST(testDisjointClusters, testLarge3){
	auto data = new std::vector<std::vector<float>*>;
	auto centroids = std::vector<unsigned int>{0,0,0,0,0,0,0,0,0,0,0};
	auto subspaces = std::vector<unsigned int>(); // 1010 1001 0001 0001 0001 0001 1111

	for(int i = 0; i < 11; i++){
		subspaces.push_back(2836467999);
		subspaces.push_back(1);
	}
	
	auto scores = std::vector<float>{200,200,300,400,500,6000,7000,99,919,999,7000}; 

	{
		auto point = new std::vector<float>;
		for(int i = 0; i < 33; i++){
			point->push_back(i);
		}
		data->push_back(point);
	}

	auto res = disjointClustersTester(data, centroids, subspaces, scores);

	EXPECT_EQ(res.size(),11);
	EXPECT_EQ(res.at(0),0);
	EXPECT_EQ(res.at(1),0);
	EXPECT_EQ(res.at(2),0);
	EXPECT_EQ(res.at(3),0);
	EXPECT_EQ(res.at(4),0);
	EXPECT_EQ(res.at(5),0);
	EXPECT_EQ(res.at(6),1);
	EXPECT_EQ(res.at(7),0);
	EXPECT_EQ(res.at(8),0);
	EXPECT_EQ(res.at(9),0);
	EXPECT_EQ(res.at(10),0);
}								


TEST(testDisjointClusters, testLarge4){
	auto data = new std::vector<std::vector<float>*>;
	auto centroids = std::vector<unsigned int>{1,0,0,0,0,0,0,0,0,0,0};
	auto subspaces = std::vector<unsigned int>(); // 1010 1001 0001 0001 0001 0001 1111

	for(int i = 0; i < 11; i++){
		subspaces.push_back(0);
		subspaces.push_back(1);
	}
	
	auto scores = std::vector<float>{200,200,300,400,500,6000,7000,99,919,999,7001}; 

	{
		auto point = new std::vector<float>;
		for(int i = 0; i < 33; i++){
			point->push_back(i);
		}
		data->push_back(point);
	}

	{
		auto point = new std::vector<float>;
		for(int i = 0; i < 33; i++){
			point->push_back(i+5);
		}
		point->at(32) = 99999;
		data->push_back(point);
	}
	
	auto res = disjointClustersTester(data, centroids, subspaces, scores);

	
	EXPECT_EQ(res.size(),11);
	EXPECT_EQ(res.at(0),1);
	EXPECT_EQ(res.at(1),0);
	EXPECT_EQ(res.at(2),0);
	EXPECT_EQ(res.at(3),0);
	EXPECT_EQ(res.at(4),0);
	EXPECT_EQ(res.at(5),0);
	EXPECT_EQ(res.at(6),0);
	EXPECT_EQ(res.at(7),0);
	EXPECT_EQ(res.at(8),0);
	EXPECT_EQ(res.at(9),0);
	EXPECT_EQ(res.at(10),1);
}								

