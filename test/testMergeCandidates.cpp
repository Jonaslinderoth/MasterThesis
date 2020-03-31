#include <gtest/gtest.h>
#include "../src/MineClus/MineClusKernels.h"
#include "testingTools.h"


TEST(testMergeCandidates, testWith2Dim){
	auto candidates = std::vector<std::vector<bool>>();
	{
		auto point = std::vector<bool>({0,1});
		auto point2 = std::vector<bool>({1,0});
		candidates.push_back(point);
		candidates.push_back(point2);
	}

	auto result = mergeCandidatesTester(candidates, 2);
	
	EXPECT_EQ(result.first.size(), 1);
	EXPECT_EQ(result.first.at(0), 3); // 3 = 000000011

	EXPECT_EQ(result.second.size(),1);
	EXPECT_EQ(result.second.at(0),0);
}

TEST(testMergeCandidates, testWith2Dim_2){
	auto candidates = std::vector<std::vector<bool>>();
	{
		auto point = std::vector<bool>({1,1});
		candidates.push_back(point);
	}

	auto result = mergeCandidatesTester(candidates);
	EXPECT_EQ(result.first.size(), 0);
}


TEST(testMergeCandidates, testWith3Dim){
	auto candidates = std::vector<std::vector<bool>>();
	{
		auto point = std::vector<bool>({0,0,1});
		auto point2 = std::vector<bool>({0,1,0});
		auto point3 = std::vector<bool>({1,0,0});
		candidates.push_back(point);
		candidates.push_back(point2);
		candidates.push_back(point3);
	}

	auto result = mergeCandidatesTester(candidates);
	EXPECT_EQ(result.first.size(), 3);
	EXPECT_EQ(result.first.at(2), 3); // 3 = 000000011
	EXPECT_EQ(result.first.at(1), 5); // 5 = 000000101
	EXPECT_EQ(result.first.at(0), 6); // 6 = 000000110
	
	EXPECT_EQ(result.second.size(),3);
	EXPECT_EQ(result.second.at(0),0);
	EXPECT_EQ(result.second.at(1),0);
	EXPECT_EQ(result.second.at(2),0);
	
}



TEST(testMergeCandidates, testWith4Dim){
	auto candidates = std::vector<std::vector<bool>>();
	{
		auto point = std::vector<bool>({0,0,0,1});
		auto point2 = std::vector<bool>({0,0,1,0});
		auto point3 = std::vector<bool>({0,1,0,0});
		auto point4 = std::vector<bool>({1,0,0,0});
		candidates.push_back(point);
		candidates.push_back(point2);
		candidates.push_back(point3);
		candidates.push_back(point4);
	}

	auto result = mergeCandidatesTester(candidates);
	EXPECT_EQ(result.first.size(), 6);
	EXPECT_EQ(result.first.at(5), 3); // 3 = 000000011
	EXPECT_EQ(result.first.at(4), 5); // 5 = 000000101
	EXPECT_EQ(result.first.at(2), 9); // 9 = 000001001
	EXPECT_EQ(result.first.at(3), 6); // 6 = 000000110
	EXPECT_EQ(result.first.at(1), 10); // 10 = 000001010
	EXPECT_EQ(result.first.at(0), 12); // 12 = 000001100

	EXPECT_EQ(result.second.size(),6);
	EXPECT_EQ(result.second.at(0),0);
	EXPECT_EQ(result.second.at(1),0);
	EXPECT_EQ(result.second.at(2),0);
	EXPECT_EQ(result.second.at(3),0);
	EXPECT_EQ(result.second.at(4),0);
	EXPECT_EQ(result.second.at(5),0);
}


TEST(testMergeCandidates, testWith44Dim){
	auto candidates = std::vector<std::vector<bool>>();
	{
		auto point = std::vector<bool>();
		for(int i = 0; i < 44; i++){
			point.push_back(0);
		}
		point.at(0) = 1;
		candidates.push_back(point);
	}
	{
		auto point = std::vector<bool>();
		for(int i = 0; i < 44; i++){
			point.push_back(0);
		}
		point.at(43) = 1;
		candidates.push_back(point);
	}

	auto result = mergeCandidatesTester(candidates);
	EXPECT_EQ(result.first.size(), 2);
	EXPECT_EQ(readBit(result.first.at(0),0), 1);
	EXPECT_EQ(readBit(result.first.at(1),11), 1);

	
	
}

TEST(testMergeCandidates, testWith44Dim_2){
	auto candidates = std::vector<std::vector<bool>>();
	{
		auto point = std::vector<bool>();
		for(int i = 0; i < 44; i++){
			point.push_back(0);
		}
		point.at(0) = 1;
		candidates.push_back(point);
	}
	{
		auto point = std::vector<bool>();
		for(int i = 0; i < 44; i++){
			point.push_back(0);
		}
		point.at(30) = 1;
		candidates.push_back(point);
	}
	{
		auto point = std::vector<bool>();
		for(int i = 0; i < 44; i++){
			point.push_back(0);
		}
		point.at(43) = 1;
		candidates.push_back(point);
	}

	auto result = mergeCandidatesTester(candidates);
	EXPECT_EQ(result.first.size(), 6);
	EXPECT_EQ(readBit(result.first.at(0),0), 1);
	EXPECT_EQ(readBit(result.first.at(0),30), 1);
	EXPECT_EQ(readBit(result.first.at(3),11), 0);

	
	EXPECT_EQ(readBit(result.first.at(1),0), 1);
	EXPECT_EQ(readBit(result.first.at(1),30), 0);
	EXPECT_EQ(readBit(result.first.at(4),11), 1);

	
	EXPECT_EQ(readBit(result.first.at(2),0), 0);
	EXPECT_EQ(readBit(result.first.at(2),30), 1);
	EXPECT_EQ(readBit(result.first.at(5),11), 1);	
}

