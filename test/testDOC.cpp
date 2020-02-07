#include <gtest/gtest.h>
#include <iostream>
#include "../src/DOC/DOC.h"
#include <vector>
#include "testData.h"
#include <cmath>



TEST(testDOC, testConstructor){
	DOC d = DOC();
	SUCCEED();
	EXPECT_EQ(d.size(), 0);
}


TEST(testDOC, testConstructor2){
	DOC d = DOC(new std::vector<std::vector<float>*>);
	SUCCEED();
	EXPECT_EQ(d.size(), 0);
}


TEST(testDOC, testConstructor3){
	auto v = new std::vector<std::vector<float>*>;
	auto v1 = new std::vector<float>;
	v1->push_back(1.1);
	v1->push_back(1.2);
	v1->push_back(1.3);
	v->push_back(v1);
	DOC d = DOC(v);
	SUCCEED();
	EXPECT_EQ(d.size(), 1);
}

TEST(testDOC, testAddPoint){
	DOC d = DOC(new std::vector<std::vector<float>*>);
	SUCCEED();
	EXPECT_EQ(d.size(), 0);
	auto v1 = new std::vector<float>;
		v1->push_back(1.1);
		v1->push_back(1.2);
		v1->push_back(1.3);
	auto v2 = new std::vector<float>;
		v2->push_back(2.1);
		v2->push_back(2.2);
		v2->push_back(2.3);
	EXPECT_TRUE(d.addPoint(v1));
	EXPECT_EQ(d.size(), 1);
	EXPECT_TRUE(d.addPoint(v2));
	EXPECT_EQ(d.size(), 2);
}

TEST(testDOC, testFindDimensions){
	DOC d = DOC(new std::vector<std::vector<float>*>);

	std::vector<float> centroid{20, 20, 20};
	std::vector<float> point1{20, 20, 20};
	std::vector<float> point2{20, 10, 20};
	std::vector<float> point3{20, 10, 20};
	std::vector<float> point4{20, 10, 10};

	auto v = new std::vector<std::vector<float>*>;
	v->push_back(&point1);
	v->push_back(&point2);
	v->push_back(&point3);
	v->push_back(&point4);

	auto res = d.findDimensions(&centroid, v,5);
	EXPECT_TRUE(res->at(0));
	EXPECT_FALSE(res->at(1));
	EXPECT_FALSE(res->at(2));
}


TEST(testDOC, testFindDimensions2){
	DOC d = DOC();

	std::vector<float> centroid{20, 20, 20};
	std::vector<float> point1{20, 20, 20};
	std::vector<float> point2{20, 10, 20};
	std::vector<float> point3{20, 10, 20};
	std::vector<float> point4{20, 10, 10};
	std::vector<float> point5{10, 10, 10};

	auto v = new std::vector<std::vector<float>*>;
	v->push_back(&point1);
	v->push_back(&point2);
	v->push_back(&point3);
	v->push_back(&point4);
	v->push_back(&point5);

	auto res = d.findDimensions(&centroid, v,5);
	EXPECT_FALSE(res->at(0));
	EXPECT_FALSE(res->at(1));
	EXPECT_FALSE(res->at(2));

}


TEST(testDOC, testFindCluster){
	std::vector<std::vector<float>*>* data = new std::vector<std::vector<float>*>;
	for(float i = 0; i < 200; i++){
		for(float j = 9; j < 15; j++){
			std::vector<float>* point1 = new std::vector<float>{j,i};
			data->push_back(point1);

		}
	}
	DOC d = DOC(data, 0.1, 0.25, 5);
	auto res = d.findCluster();
	SUCCEED();




	/* int i = 0;
	   for(int i = 0; i < res.first.size(); i++){
		std::cout << res.first.at(i).at(0) << ", ";
	}
	std::cout << std::endl;

	i = 0;
	for(int i = 0; i < res.first.size(); i++){
		std::cout << res.first.at(i).at(1) << ", ";
	}
	std::cout << std::endl;
	*/

	EXPECT_TRUE(res.second->at(0));
	EXPECT_FALSE(res.second->at(1));

	int count = 0;
	EXPECT_EQ(res.first->size(), 1200);
}

TEST(testDOC, testHardcodedRandom){
	std::vector<std::vector<float>*>* data = new std::vector<std::vector<float>*>;
		for(float i = 0; i < 100; i++){
			for(float j = 9; j < 12; j++){
				std::vector<float>* point1 = new std::vector<float>{j,i};
				data->push_back(point1);
			}
		}
		DOC d = DOC(data);
		d.setRandom(false);
		d.setRandomStub({0,1,2,3,4});
		auto s = d.pickRandom(10);
		EXPECT_EQ(s->at(0), data->at(0));
		EXPECT_EQ(s->at(1), data->at(1));
		EXPECT_EQ(s->at(2), data->at(2));
		EXPECT_EQ(s->at(3), data->at(3));
		EXPECT_EQ(s->at(4), data->at(4));
		EXPECT_EQ(s->at(5), data->at(0));
		EXPECT_NE(s->at(5), data->at(1));
}


TEST(testDOC, testFindCluster2){
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

	DOC d = DOC(data, 0.1, 0.25, 5);
	auto res = d.findCluster();
	SUCCEED();


	EXPECT_TRUE(res.second->at(0));
	EXPECT_FALSE(res.second->at(1));

	EXPECT_EQ(res.first->size(), 306);

	for(int i = 0; i< data->size(); i++){
		delete data->at(i);
	}
	delete data;
	delete res.first;
	delete res.second;
}

TEST(testDOC, testMu){
	auto d = DOC();
	EXPECT_EQ(d.mu(1,1), 4);
	EXPECT_EQ(d.mu(2,2), 32);
	EXPECT_EQ(d.mu(3,2), 48);
	EXPECT_EQ(d.mu(3,3), 192);
	EXPECT_EQ(d.mu(3,4), 768);
}


TEST(testDOC, testFindCluster3){
	std::vector<std::vector<float>*>* data = data_4dim2cluster();

	//std::cout << a << ", " << b << std::endl;

	DOC d = DOC(data, 0.1, 0.25, 5);
	auto res = d.findCluster();
	SUCCEED();


	EXPECT_TRUE(res.second->at(0));
	EXPECT_FALSE(res.second->at(1));
	EXPECT_FALSE(res.second->at(2));
	EXPECT_FALSE(res.second->at(3));

	EXPECT_LT(abs((int)res.first->size()-(int)numPoints_4dim2cluster()), 10);

	for(int i = 0; i< data->size(); i++){
		delete data->at(i);
	}
	delete data;
	delete res.first;
	delete res.second;
}


TEST(testDOC, testFindCluster4Bad){
	std::vector<std::vector<float>*>* data = new std::vector<std::vector<float>*>;

	for(int i = 0; i < 600; i+=10){
	  std::vector<float>* point1 = new std::vector<float>;
		for(float j = 0; j < 20; j++){
		  point1->push_back(i);
		}
		data->push_back(point1);
	}


	for(int i = 0; i < 5; i+=1){
	  std::vector<float>* point1 = new std::vector<float>;
		for(float j = 0; j < 20; j++){
		  point1->push_back(i);
		}
		data->push_back(point1);
	}

	
	//std::cout << a << ", " << b << std::endl;

	DOC d = DOC(data, 0.1, 0.25, 5);
	auto res = d.findCluster();
	SUCCEED();

	
	EXPECT_EQ(res.second->size(), 20);
	for(int i = 0; i < 20; i++){
	  EXPECT_FALSE(res.second->at(i));
	}

	EXPECT_EQ(res.first->size(), 65);

	for(int i = 0; i< data->size(); i++){
		delete data->at(i);
	}
	delete data;
	delete res.first;
	delete res.second;
}
