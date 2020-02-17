#include <gtest/gtest.h>
#include <iostream>
#include "../src/DOC/DOC.h"
#include "../src/Clustering.h"


TEST(testCLustering, testSetup){
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
	//std::cout << a << ", " << b << std::endl;

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
