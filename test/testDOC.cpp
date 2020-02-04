#include <gtest/gtest.h>
#include <iostream>
#include "../src/DOC/DOC.h"
#include <vector>



TEST(testDOC, testConstructor){
	DOC d = DOC();
	SUCCEED();
	EXPECT_EQ(d.size(), 0);
}


TEST(testDOC, testConstructor2){
	DOC d = DOC(std::vector<std::vector<float>>());
	SUCCEED();
	EXPECT_EQ(d.size(), 0);
}


TEST(testDOC, testConstructor3){
	auto v = std::vector<std::vector<float>>();
	auto v1 = std::vector<float>();
	v1.push_back(1.1);
	v1.push_back(1.2);
	v1.push_back(1.3);
	v.push_back(v1);
	DOC d = DOC(v);
	SUCCEED();
	EXPECT_EQ(d.size(), 1);
}

TEST(testDOC, testAddPoint){
	DOC d = DOC(std::vector<std::vector<float>>());
	SUCCEED();
	EXPECT_EQ(d.size(), 0);
	auto v1 = std::vector<float>();
		v1.push_back(1.1);
		v1.push_back(1.2);
		v1.push_back(1.3);
	auto v2 = std::vector<float>();
		v2.push_back(2.1);
		v2.push_back(2.2);
		v2.push_back(2.3);
	EXPECT_TRUE(d.addPoint(v1));
	EXPECT_EQ(d.size(), 1);
	EXPECT_TRUE(d.addPoint(v2));
	EXPECT_EQ(d.size(), 2);
}

TEST(testDOC, testFindDimensions){
	DOC d = DOC(std::vector<std::vector<float>>());

	std::vector<float> centroid{20, 20, 20};
	std::vector<float> point1{20, 20, 20};
	std::vector<float> point2{20, 10, 20};
	std::vector<float> point3{20, 10, 20};
	std::vector<float> point4{20, 10, 10};

	auto v = std::vector<std::vector<float>>();
	v.push_back(point1);
	v.push_back(point2);
	v.push_back(point3);
	v.push_back(point4);

	auto res = d.findDimensions(centroid, v,5);
	EXPECT_TRUE(res.at(0));
	EXPECT_FALSE(res.at(1));
	EXPECT_FALSE(res.at(2));
}


TEST(testDOC, testFindDimensions2){
	DOC d = DOC(std::vector<std::vector<float>>());

	std::vector<float> centroid{20, 20, 20};
	std::vector<float> point1{20, 20, 20};
	std::vector<float> point2{20, 10, 20};
	std::vector<float> point3{20, 10, 20};
	std::vector<float> point4{20, 10, 10};
	std::vector<float> point5{10, 10, 10};

	auto v = std::vector<std::vector<float>>();
	v.push_back(point1);
	v.push_back(point2);
	v.push_back(point3);
	v.push_back(point4);
	v.push_back(point5);

	auto res = d.findDimensions(centroid, v,5);
	EXPECT_FALSE(res.at(0));
	EXPECT_FALSE(res.at(1));
	EXPECT_FALSE(res.at(2));

}


TEST(testDOC, testFindCluster){
	std::vector<std::vector<float>> data = std::vector<std::vector<float>>();
	for(int i = 0; i < 100; i++){
		for(int j = 9; j < 12; j++){
			std::vector<float> point1{j,i};
			data.push_back(point1);
		}
	}
	DOC d = DOC(data);
	auto res = d.findCluster();
	SUCCEED();

	EXPECT_TRUE(res.second.at(0));
	EXPECT_FALSE(res.second.at(1));

	int count = 0;
	EXPECT_EQ(res.first.size(), 300);





}
