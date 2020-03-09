#include <gtest/gtest.h>
#include <iostream>
#include "../src/DOC/DOC.h"
#include "../src/Clustering.h"
#include "../src/DOC_GPU/DOCGPU.h"
#include "../src/Fast_DOC/Fast_DOC.h"
#include <algorithm>
#include "testData.h"


TEST(testClusteringSetup, testSetup){
std::vector<std::vector<float>*>* data = new std::vector<std::vector<float>*>;
	int a = 0;
	for(float i = 9; i <= 12; i++){
		for(float j = 9; j <= 12; j++){
			std::vector<float>* point1 = new std::vector<float>{i,j};
			data->push_back(point1);
			a++;
		}
	}

	int b = 0;
	for(float i = 60; i <= 65; i++){
		for(float j = 0; j <= 50; j++){
			std::vector<float>* point1 = new std::vector<float>{i,j};
			data->push_back(point1);
			b++;
		}
	}

	{
		DOCGPU* d = new DOCGPU(data, 0.1, 0.25, 5);
		d->setSeed(1);	
		auto res = d->findCluster();
		SUCCEED();
		EXPECT_TRUE(res.second->at(0));
		EXPECT_FALSE(res.second->at(1));
		EXPECT_EQ(res.first->size(), 306);
	};

	

	DOC* d = new DOC(data, 0.1, 0.25, 5);
	d->setSeed(1);
	
	{
		auto res = d->findCluster();
		SUCCEED();
		EXPECT_TRUE(res.second->at(0));
		EXPECT_FALSE(res.second->at(1));
		EXPECT_EQ(res.first->size(), 306);
	};


	
	{
		Clustering* c = d;
		auto res = c->findCluster();
		SUCCEED();
	

		EXPECT_TRUE(res.second->at(0));
		EXPECT_FALSE(res.second->at(1));
		EXPECT_EQ(res.first->size(), 306);
		};


	{
		Clustering* c =  new DOC(data, 0.1, 0.25, 5);
		c->setSeed(10);
		auto res = c->findCluster();
		SUCCEED();
	

		EXPECT_TRUE(res.second->at(0));
		EXPECT_FALSE(res.second->at(1));
		EXPECT_EQ(res.first->size(), 306);
		};
	
	
	for(int i = 0; i< data->size(); i++){
		delete data->at(i);
	}
	delete data;
}



class testClusteringPattern : public testing::TestWithParam<std::string>
	{
	public:		
		void SetUp()
		{
		}
		void TearDown()
		{
		}
};

Clustering* constructClustering(std::string type, std::vector<std::vector<float>*>* data){
	if (type == "DOC"){
		return new DOC(data);
	}else if (type == "DOCGPU"){
		return new DOCGPU(data);
	}else if (type == "Fast_DOC"){
		return new Fast_DOC(data);
	}
};

INSTANTIATE_TEST_CASE_P(ValidInput,
                        testClusteringPattern,
                        ::testing::Values(
										  "DOC",
										  "DOCGPU",
										  "Fast_DOC"),
                        );
 
TEST_P(testClusteringPattern, testSetup){
	std::vector<std::vector<float>*>* data = new std::vector<std::vector<float>*>;
	int a = 0;
	for(float i = 9; i <= 12; i++){
		for(float j = 9; j <= 12; j++){
			std::vector<float>* point1 = new std::vector<float>{i,j};
			data->push_back(point1);
			a++;
		}
	}

	int b = 0;
	for(float i = 60; i <= 65; i++){
		for(float j = 0; j <= 50; j++){
			std::vector<float>* point1 = new std::vector<float>{i,j};
			data->push_back(point1);
			b++;
		}
	}
	
	Clustering* c =  constructClustering(GetParam(), data);
	c->setSeed(1);
	c->setWidth(5);
	auto res = c->findCluster();
	
	SUCCEED();
		

	//EXPECT_TRUE(res.second->at(0));
	//EXPECT_FALSE(res.second->at(1));
	//EXPECT_EQ(res.first->size(), 306);
}



TEST_P(testClusteringPattern, testCluster1){
	std::vector<std::vector<float>*>* data = data_4dim2cluster();

	Clustering* c =  constructClustering(GetParam(), data);
	c->setWidth(6);
	c->setSeed(100);
	std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> res = c->findCluster();
	
	SUCCEED();
	EXPECT_EQ(res.second->size(), 4);
	EXPECT_TRUE(res.second->at(0));
	EXPECT_FALSE(res.second->at(1));
	EXPECT_FALSE(res.second->at(2));
	EXPECT_FALSE(res.second->at(3));

	EXPECT_LT(abs((int)res.first->size()-(int)numPoints_4dim2cluster()), 20);


}


TEST_P(testClusteringPattern, testCluster2){
	std::vector<std::vector<float>*>* data = new std::vector<std::vector<float>*>;
	{
		auto a = new std::vector<float>{1,2,3};
		data->push_back(a);
	}

	{
		auto a = new std::vector<float>{10,2,3};
		data->push_back(a);
	}

	{
		auto a = new std::vector<float>{-10,2,3};
		data->push_back(a);
	}
	{
		auto a = new std::vector<float>{100,2,3};
		data->push_back(a);
	}
	{
		auto a = new std::vector<float>{-100,2,3};
		data->push_back(a);
	}

	Clustering* c =  constructClustering(GetParam(), data);
	c->setWidth(5);
	c->setSeed(1);
	std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> res = c->findCluster();
	
	SUCCEED();
	EXPECT_EQ(res.first->size(), 5);
	EXPECT_EQ(res.second->size(), 3);
	EXPECT_FALSE(res.second->at(0));
	EXPECT_TRUE(res.second->at(1));
	EXPECT_TRUE(res.second->at(2));
}


TEST_P(testClusteringPattern, testCluster3){
	std::vector<std::vector<float>*>* data = new std::vector<std::vector<float>*>;
	{
		auto a = new std::vector<float>{1.1,2.1,3.1};
		data->push_back(a);
	}

	{
		auto a = new std::vector<float>{10.2,2.2,3.2};
		data->push_back(a);
	}

	{
		auto a = new std::vector<float>{-10.3,2.3,3.3};
		data->push_back(a);
	}
	{
		auto a = new std::vector<float>{100.4,2.4,3.4};
		data->push_back(a);
	}
	{
		auto a = new std::vector<float>{-100,2,3};
		data->push_back(a);
	}

	Clustering* c =  constructClustering(GetParam(), data);
	c->setWidth(5);
	c->setSeed(1);
	std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> res = c->findCluster();
	
	SUCCEED();
	EXPECT_EQ(res.first->size(), 5);
	EXPECT_EQ(res.second->size(), 3);
	EXPECT_FALSE(res.second->at(0));
	EXPECT_TRUE(res.second->at(1));
	EXPECT_TRUE(res.second->at(2));
}
