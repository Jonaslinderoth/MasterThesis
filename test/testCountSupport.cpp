#include <gtest/gtest.h>
#include "../src/MineClus/MineClusKernels.h"
#include "testingTools.h"


TEST(testCountSupport, test2){
	auto itemSet = std::vector<std::vector<bool>>();
	auto candidates = std::vector<std::vector<bool>>();

	candidates.push_back({1,0});
	candidates.push_back({0,1});


	itemSet.push_back({1,0});
	itemSet.push_back({0,1});
	itemSet.push_back({1,1});

	
	auto result = countSupportTester(candidates, itemSet, 1, 0.25);
	auto support = std::get<0>(result);
	auto score = std::get<1>(result);
	auto toBeDeleted = std::get<2>(result);

	EXPECT_EQ(support.size(), 2);
	EXPECT_EQ(support.at(0), 2);
	EXPECT_EQ(support.at(1), 2);

	EXPECT_EQ(score.size(), 2);
	EXPECT_EQ(score.at(0), mu(1,2, 0.25));
	EXPECT_EQ(score.at(1), mu(1,2, 0.25));


	EXPECT_EQ(toBeDeleted.size(), 2);
	EXPECT_EQ(toBeDeleted.at(0), false);
	EXPECT_EQ(toBeDeleted.at(1), false);
}

TEST(testCountSupport, test3){
	auto itemSet = std::vector<std::vector<bool>>();
	auto candidates = std::vector<std::vector<bool>>();

	candidates.push_back({1,0,0});
	candidates.push_back({0,1,1});
	candidates.push_back({1,1,1});


	itemSet.push_back({1,0,0});
	itemSet.push_back({0,1,0});
	itemSet.push_back({1,1,0});

	
	auto result = countSupportTester(candidates, itemSet, 1, 0.25);
	auto support = std::get<0>(result);
	auto score = std::get<1>(result);
	auto toBeDeleted = std::get<2>(result);

	EXPECT_EQ(support.size(), 3);
	EXPECT_EQ(support.at(0), 2);
	EXPECT_EQ(support.at(1), 0);
	EXPECT_EQ(support.at(2), 0);

	EXPECT_EQ(score.size(), 3);
	EXPECT_EQ(score.at(0), mu(1,2, 0.25));
	EXPECT_EQ(score.at(1), mu(0,0, 0.25));
	EXPECT_EQ(score.at(2), mu(0,0, 0.25));


	EXPECT_EQ(toBeDeleted.size(), 3);
	EXPECT_EQ(toBeDeleted.at(0), false);
	EXPECT_EQ(toBeDeleted.at(1), true);
	EXPECT_EQ(toBeDeleted.at(2), true);
}



TEST(testCountSupport, test33){
	auto itemSet = std::vector<std::vector<bool>>();
	auto candidates = std::vector<std::vector<bool>>();

	{
		auto point = std::vector<bool>();
		for(int i = 0; i < 33; i++){
			point.push_back(0);
		}
		point.at(0) = 1;
		candidates.push_back(point);
	}
	{
		auto point = std::vector<bool>();
		for(int i = 0; i < 33; i++){
			point.push_back(0);
		}
		point.at(32) = 1;
		candidates.push_back(point);
	}

	
	{
		auto point = std::vector<bool>();
		for(int i = 0; i < 33; i++){
			point.push_back(0);
		}
		point.at(0) = 1;
		itemSet.push_back(point);
	}	
	{
		auto point = std::vector<bool>();
		for(int i = 0; i < 33; i++){
			point.push_back(0);
		}
		point.at(32) = 1;
		itemSet.push_back(point);
	}
	{
		auto point = std::vector<bool>();
		for(int i = 0; i < 33; i++){
			point.push_back(0);
		}
		point.at(32) = 1;
		itemSet.push_back(point);
	}
	{
		auto point = std::vector<bool>();
		for(int i = 0; i < 33; i++){
			point.push_back(0);
		}
		point.at(32) = 1;
		point.at(30) = 1;
		itemSet.push_back(point);
	}
	{
		auto point = std::vector<bool>();
		for(int i = 0; i < 33; i++){
			point.push_back(0);
		}
		point.at(32) = 1;
		point.at(0) = 1;
		itemSet.push_back(point);
	}
	
	auto result = countSupportTester(candidates, itemSet, 1, 0.25);
	auto support = std::get<0>(result);
	auto score = std::get<1>(result);
	auto toBeDeleted = std::get<2>(result);

	EXPECT_EQ(support.size(), 2);
	EXPECT_EQ(support.at(0), 2);
	EXPECT_EQ(support.at(1), 4);

	EXPECT_EQ(score.size(), 2);
	EXPECT_EQ(score.at(0), mu(1,2, 0.25));
	EXPECT_EQ(score.at(1), mu(1,4, 0.25));


	EXPECT_EQ(toBeDeleted.size(), 2);
	EXPECT_EQ(toBeDeleted.at(0), false);
	EXPECT_EQ(toBeDeleted.at(1), false);
}




