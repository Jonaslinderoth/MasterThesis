#include <gtest/gtest.h>
#include "../src/Fast_DOC/Fast_DOC.h"


TEST(testFastDOC, testFindClusterSimple1){
	auto data = new std::vector<std::vector<float>*>;
	{
		auto point = new std::vector<float>{1,10000};
		data->push_back(point);
	}
	{
		auto point = new std::vector<float>{2,10000};
		data->push_back(point);
	}
	{
		auto point = new std::vector<float>{3,10000};
		data->push_back(point);
	}
	{
		auto point = new std::vector<float>{4,10000};
		data->push_back(point);
	}
	{
		auto point = new std::vector<float>{10000,1};
		data->push_back(point);
	}

	auto c = new Fast_DOC(data);
	c->setSeed(1);
	auto res = c->findCluster();

	EXPECT_EQ(res.first->size(), 5);
	EXPECT_EQ(res.second->size(), 2);
	EXPECT_FALSE(res.second->at(0));
	EXPECT_FALSE(res.second->at(1));
	
}
