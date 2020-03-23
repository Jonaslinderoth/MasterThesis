#include <gtest/gtest.h>
#include "../src/MineClus/MineClusKernels.h"

bool readBit(unsigned int value, unsigned int index){
	return ((value & (1 << index)) >> index);
}

TEST(testCreateItemSet, testSimple){
	auto data = new std::vector<std::vector<float>*>;
	auto point = new std::vector<float>;
	point->push_back(0);
	point->push_back(-10);
	point->push_back(100);
	data->push_back(point);

	auto res = createItemSetTester(data, 0, 1);

	EXPECT_EQ(res.size(), 1);

	EXPECT_EQ(((res.at(0) & (1 << 0)) >> 0), 1);
	EXPECT_EQ(((res.at(0) & (1 << 1)) >> 1), 1);
	EXPECT_EQ(((res.at(0) & (1 << 2)) >> 2), 1);

	EXPECT_TRUE(readBit(res.at(0), 0));
	EXPECT_TRUE(readBit(res.at(0), 1));
	EXPECT_TRUE(readBit(res.at(0), 2));
}


TEST(testCreateItemSet, testWithTwoPoints){
	auto data = new std::vector<std::vector<float>*>;
	{
		auto point = new std::vector<float>;
		point->push_back(0);
		point->push_back(-10);
		point->push_back(100);
		data->push_back(point);
	}

	{
		auto point = new std::vector<float>;
		point->push_back(5);
		point->push_back(-15);
		point->push_back(105);
		data->push_back(point);
	}

	auto res = createItemSetTester(data, 0, 10);

	EXPECT_EQ(res.size(), 2);

	EXPECT_TRUE(readBit(res.at(0), 0));
	EXPECT_TRUE(readBit(res.at(0), 1));
	EXPECT_TRUE(readBit(res.at(0), 2));

	EXPECT_TRUE(readBit(res.at(1), 0));
	EXPECT_TRUE(readBit(res.at(1), 1));
	EXPECT_TRUE(readBit(res.at(1), 2));

	
	res = createItemSetTester(data, 1, 10);
	EXPECT_EQ(res.size(), 2);
	EXPECT_TRUE(readBit(res.at(0), 0));
	EXPECT_TRUE(readBit(res.at(0), 1));
	EXPECT_TRUE(readBit(res.at(0), 2));

	EXPECT_TRUE(readBit(res.at(1), 0));
	EXPECT_TRUE(readBit(res.at(1), 1));
	EXPECT_TRUE(readBit(res.at(1), 2));
}


TEST(testCreateItemSet, testWithTwoPoints65Dim){
	auto data = new std::vector<std::vector<float>*>;
	{
		auto point = new std::vector<float>;
		for(int i = 0; i < 65; i++){
			point->push_back(i);
		}
		data->push_back(point);
	}

	{
		auto point = new std::vector<float>;
		for(int i = 0; i < 65; i++){
			if(i%2 == 0){
				point->push_back(i);				
			}else{
				point->push_back(10000);
			}

		}
		data->push_back(point);
	}

	auto res = createItemSetTester(data, 0, 10);

	EXPECT_EQ(res.size(), 6);

	for(int i = 0; i < 32; i++){
		EXPECT_TRUE(readBit(res.at(0), i));		
	}
	for(int i = 0; i < 32; i++){
		EXPECT_EQ(readBit(res.at(1), i), (i%2 == 0));		
	}
	for(int i = 0; i < 32; i++){
		EXPECT_TRUE(readBit(res.at(2), i));		
	}
	for(int i = 0; i < 32; i++){
		EXPECT_EQ(readBit(res.at(3), i), (i%2 == 0));		
	}
	EXPECT_EQ(readBit(res.at(4), 0), 1);
	EXPECT_EQ(readBit(res.at(4), 1), 0);
	EXPECT_EQ(readBit(res.at(4), 30), 0);

	EXPECT_EQ(readBit(res.at(5), 0), 1);
	EXPECT_EQ(readBit(res.at(5), 1), 0);
	EXPECT_EQ(readBit(res.at(5), 30), 0);
}
