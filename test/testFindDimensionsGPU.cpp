#include <gtest/gtest.h>
#include <iostream>
#include "../src/DOC_GPU/HyperCube.h"

TEST(testHyperCubeGPU, testFindDimmensionsInit){
	auto a = new std::vector<std::vector<float>*>;
	auto b = new std::vector<float>{1,2,3,4};
	a->push_back(b);

	auto c = std::vector<std::vector<std::vector<float>*>*>();
	c.push_back(a);
	
	auto res = findDimmensions(a, c);
	SUCCEED();
	EXPECT_EQ(res->size(), 1);
}


TEST(testHyperCubeGPU, testFindDimmensions){
	auto ps = new std::vector<std::vector<float>*>;
	auto p = new std::vector<float>{1,1,1};
	ps->push_back(p);
	auto xss = std::vector<std::vector<std::vector<float>*>*>();
	auto xs = new std::vector<std::vector<float>*>;
	auto x1 = new std::vector<float>{1,2,1000};
	auto x2 = new std::vector<float>{2,1, 1000};
	xs->push_back(x1);
	xs->push_back(x2);
	xss.push_back(xs);

	
	auto res = findDimmensions(ps, xss);
	SUCCEED();
	EXPECT_EQ(res->size(), 1);
	EXPECT_EQ(res->at(0)->size(), 3);
	EXPECT_TRUE(res->at(0)->at(0));
	EXPECT_TRUE(res->at(0)->at(1));
	EXPECT_FALSE(res->at(0)->at(2));
	
}
