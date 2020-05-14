#include <gtest/gtest.h>
#include "testingTools.h"
#include <random>
#include "../src/MineClusGPU/MineClusGPUnified.h"
#include "../experiments/genTestData.cpp"

TEST(testMnist, testGen){
	auto data = getMnist();
	EXPECT_EQ(data->size(), 10000);
	for(int i = 0; i < data->size(); i++){
		EXPECT_EQ(data->at(i)->size(), 784);
	}
}


TEST(testMnist, testMineClus){
	auto data = getMnist();
	auto c = MineClusGPUnified(data);
	c.setSeed(1);

	auto res = c.findKClusters(1);
	SUCCEED();
}
