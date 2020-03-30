#include <gtest/gtest.h>
#include "../src/MineClus/MineClusKernels.h"
#include "testingTools.h"


TEST(testExtractMax, testSimple1){
	std::vector<bool> oldCandidate = std::vector<bool>({1,0,0});
	std::vector<std::vector<bool>> newCandidate = std::vector<std::vector<bool>>();
	auto point = std::vector<bool>({1,1,0});
	newCandidate.push_back(point);
	auto oldScore = 10;
	auto newScore = std::vector<float>({100});
	auto oldCentroid = 0;
	auto newCentroid = 1000;
	
	auto result = extractMaxTester(oldCandidate, oldScore, oldCentroid,
								   newCandidate, newScore, newCentroid,0);

	EXPECT_EQ(result.first.size(), 2);
	EXPECT_EQ(result.first.at(0), 3);
	EXPECT_EQ(result.first.at(1), 1000);
	EXPECT_EQ(result.second, 100);
}

TEST(testExtractMax, testSimple2){
	std::vector<bool> oldCandidate = std::vector<bool>({1,0,0});
	std::vector<std::vector<bool>> newCandidate = std::vector<std::vector<bool>>();
	{	auto point = std::vector<bool>({1,1,0});
		newCandidate.push_back(point);}
	auto point = std::vector<bool>({1,1,1});
	newCandidate.push_back(point);
	auto oldScore = 10;
	auto newScore = std::vector<float>({99,100});
	auto oldCentroid = 0;
	auto newCentroid = 1000;
	
	auto result = extractMaxTester(oldCandidate, oldScore, oldCentroid,
								   newCandidate, newScore, newCentroid,1);

	EXPECT_EQ(result.first.size(), 2);
	EXPECT_EQ(result.first.at(0), 7);
	EXPECT_EQ(result.first.at(1), 1000);
	EXPECT_EQ(result.second, 100);
}



TEST(testExtractMax, testDim55){
	std::vector<bool> oldCandidate = std::vector<bool>();
	std::vector<std::vector<bool>> newCandidate;
	std::vector<bool> point = std::vector<bool>();


	for(int i = 0; i < 55; i++){
		oldCandidate.push_back(i % 4);
	}

	for(int i = 0; i < 55; i++){
		point.push_back(i % 2);		
	}

	newCandidate.push_back(point);

	auto oldScore = 10;
	auto newScore = std::vector<float>({100});
	auto oldCentroid = 0;
	auto newCentroid = 1000;
	
	auto result = extractMaxTester(oldCandidate, oldScore, oldCentroid,
								   newCandidate, newScore, newCentroid,0);

	EXPECT_EQ(result.first.size(), 3);
	EXPECT_EQ(result.first.at(0), 2863311530);
	EXPECT_EQ(result.first.at(1), 2796202);
	EXPECT_EQ(result.first.at(2), 1000);
	EXPECT_EQ(result.second, 100);
}


TEST(testExtractMax, testDim55SmallerScore){
	std::vector<bool> oldCandidate = std::vector<bool>();
	std::vector<std::vector<bool>> newCandidate;
	std::vector<bool> point = std::vector<bool>();


	for(int i = 0; i < 55; i++){
		oldCandidate.push_back(i % 4);
	}

	for(int i = 0; i < 55; i++){
		point.push_back(i % 2);		
	}

	newCandidate.push_back(point);
	

	auto oldScore = 10;
	auto newScore = std::vector<float>({1});
	auto oldCentroid = 0;
	auto newCentroid = 1000;
	
	auto result = extractMaxTester(oldCandidate, oldScore, oldCentroid,
								   newCandidate, newScore, newCentroid,0);

	EXPECT_EQ(result.first.size(), 3);
	EXPECT_EQ(result.first.at(0), 4008636142);
	EXPECT_EQ(result.first.at(1), 7270126);
	EXPECT_EQ(result.first.at(2), 0);
	EXPECT_EQ(result.second, 10);
}

TEST(testExtractMax, testDim55_2){
	std::vector<bool> oldCandidate = std::vector<bool>();
	std::vector<std::vector<bool>> newCandidate;



	for(int i = 0; i < 55; i++){
		oldCandidate.push_back(i % 4);
	}

	
	{
		std::vector<bool> point = std::vector<bool>();
		for(int i = 0; i < 55; i++){
		point.push_back(i % 3);		
	}

		newCandidate.push_back(point);}

	std::vector<bool> point = std::vector<bool>();
	for(int i = 0; i < 55; i++){
		point.push_back(i % 2);		
	}
	
	newCandidate.push_back(point);	

	auto oldScore = 10;
	auto newScore = std::vector<float>({100, 111});
	auto oldCentroid = 0;
	auto newCentroid = 1000;
	
	auto result = extractMaxTester(oldCandidate, oldScore, oldCentroid,
								   newCandidate, newScore, newCentroid,1);

	EXPECT_EQ(result.first.size(), 3);
	EXPECT_EQ(result.first.at(0), 2863311530);
	EXPECT_EQ(result.first.at(1), 2796202);
	EXPECT_EQ(result.first.at(2), 1000);
	EXPECT_EQ(result.second, 111);
}

TEST(testExtractMax, testDim55_3){
	std::vector<bool> oldCandidate = std::vector<bool>();
	std::vector<std::vector<bool>> newCandidate;


	for(int i = 0; i < 55; i++){
		oldCandidate.push_back(i % 4);
	}

	
	{
		std::vector<bool> point = std::vector<bool>();
		for(int i = 0; i < 55; i++){
		point.push_back(i % 2);		
	}

		newCandidate.push_back(point);}	
	
	{
		std::vector<bool> point = std::vector<bool>();
		for(int i = 0; i < 55; i++){
		point.push_back(i % 3);		
	}

		newCandidate.push_back(point);}


	auto oldScore = 10;
	auto newScore = std::vector<float>({100, 111});
	auto oldCentroid = 0;
	auto newCentroid = 1000;
	
	auto result = extractMaxTester(oldCandidate, oldScore, oldCentroid,
								   newCandidate, newScore, newCentroid,0);

	EXPECT_EQ(result.first.size(), 3);
	EXPECT_EQ(result.first.at(0), 2863311530);
	EXPECT_EQ(result.first.at(1), 2796202);
	EXPECT_EQ(result.first.at(2), 1000);
	EXPECT_EQ(result.second, 100);
}
