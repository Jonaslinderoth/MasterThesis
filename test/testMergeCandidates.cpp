#include <gtest/gtest.h>
#include "../src/MineClusGPU/MergeCandidates.h"
#include "testingTools.h"
#include <map>
#include <random>

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

void compareResults(std::pair<std::vector<unsigned int>,std::vector<bool>> res1, std::pair<std::vector<unsigned int>,std::vector<bool>> res2, int version = 0){
	EXPECT_EQ(res1.first.size(), res2.first.size());
	EXPECT_EQ(res1.second.size(), res2.second.size());

	std::map<unsigned int, unsigned int> map1;
	std::map<unsigned int, unsigned int> map2;
	if(version == 0 || version == 1){
		for(int i = 0; i < res1.first.size(); i++){
			if(map1.find(res1.first.at(i)) == map1.end()){
				map1[res1.first.at(i)] = 1;			
			}else{
				map1[res1.first.at(i)]++;
			} 
		}

		for(int i = 0; i < res2.first.size(); i++){
			if(map2.find(res2.first.at(i)) == map2.end()){
				map2[res2.first.at(i)] = 1;			
			}else{
				map2[res2.first.at(i)]++;
			} 
		}

		EXPECT_TRUE(std::equal(map1.begin(), map1.end(),
							   map2.begin()));

		for (auto const& pair: map1) {
			ASSERT_EQ(map1[pair.first] , map2[pair.first]) << pair.first;
		}
	}
	if(version == 0 || version == 2){
		std::map<unsigned int, unsigned int> map1_1;
		std::map<unsigned int, unsigned int> map2_1;
		for(int i = 0; i < res1.second.size(); i++){
			if(map1_1.find(res1.second.at(i)) == map1_1.end()){
				map1_1[res1.second.at(i)] = 1;			
			}else{
				map1_1[res1.second.at(i)]++;
			} 
		}
		for(int i = 0; i < res2.second.size(); i++){
			if(map2_1.find(res2.second.at(i)) == map2_1.end()){
				map2_1[res2.second.at(i)] = 1;			
			}else{
				map2_1[res2.second.at(i)]++;
			} 
		}
		EXPECT_TRUE(std::equal(map1_1.begin(), map1_1.end(),
							   map2_1.begin()));
	}
}

TEST(testMergeCandidates, testMergeSmemSimple2points1chunk2){
	auto candidates = std::vector<std::vector<bool>>();
	{
		auto point = std::vector<bool>({0,1});
		auto point2 = std::vector<bool>({1,0});
		candidates.push_back(point);
		candidates.push_back(point2);
	}

	auto resSmem = mergeCandidatesTester(candidates, 2, SharedMemoryMerge, 2);


	EXPECT_EQ(resSmem.first.size(), 1);
	EXPECT_EQ(resSmem.second.size(), 1);
	EXPECT_EQ(resSmem.first.at(0), 3);
	EXPECT_EQ(resSmem.second.at(0), 0);

}

TEST(testMergeCandidates, testMergeSmemSimple2points1chunk){
	auto candidates = std::vector<std::vector<bool>>();
	{
		auto point = std::vector<bool>({0,1});
		auto point2 = std::vector<bool>({1,0});
		candidates.push_back(point);
		candidates.push_back(point2);
	}

	auto resultNaive = mergeCandidatesTester(candidates, 2, NaiveMerge);
	auto resSmem = mergeCandidatesTester(candidates, 2, SharedMemoryMerge, 2);



	EXPECT_EQ(resSmem.first.size(), 1);
	EXPECT_EQ(resSmem.second.size(), 1);
	EXPECT_EQ(resSmem.first.at(0), 3);
	EXPECT_EQ(resSmem.second.at(0), 0);

	EXPECT_EQ(resultNaive.first.size(), 1);
	EXPECT_EQ(resultNaive.second.size(), 1);
	EXPECT_EQ(resultNaive.first.at(0), 3);
	EXPECT_EQ(resultNaive.second.at(0), 0);
	
	compareResults(resSmem, resultNaive);
}

TEST(testMergeCandidates, testMergeSmemSimple2points2chunks){
	auto candidates = std::vector<std::vector<bool>>();
	{
		auto point = std::vector<bool>({0,0,0,1});
		auto point2 = std::vector<bool>({0,0,1,0});
		candidates.push_back(point);
		candidates.push_back(point2);
	}

	auto resultSmem = mergeCandidatesTester(candidates, 2, SharedMemoryMerge, 1);
	auto resultNaive = mergeCandidatesTester(candidates, 2, NaiveMerge);
	compareResults(resultSmem, resultNaive);
}

TEST(testMergeCandidates, testMergeSmemSimple4points2chunks){
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

	auto resultSmem = mergeCandidatesTester(candidates, 2, SharedMemoryMerge, 2);
	auto resultNaive = mergeCandidatesTester(candidates, 2, NaiveMerge);

	
	compareResults(resultSmem, resultNaive);
}

TEST(testMergeCandidates, testMergeSmemSimple4points1chunk){
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

	auto resultSmem = mergeCandidatesTester(candidates, 2, SharedMemoryMerge, 2);
	auto resultNaive = mergeCandidatesTester(candidates, 2, NaiveMerge);
	compareResults(resultSmem, resultNaive);
}



TEST(testMergeCandidates, testMergeSmemSimple6points3chunk){
	auto candidates = std::vector<std::vector<bool>>();
	{
		auto point = std::vector<bool>({0,0,0,0,1});
		auto point2 = std::vector<bool>({0,0,0,1,0});
		auto point3 = std::vector<bool>({0,0,1,0,0});
		auto point4 = std::vector<bool>({0,1,0,0,0});
		auto point5 = std::vector<bool>({1,0,0,0,0});
		auto point6 = std::vector<bool>({1,0,0,0,0});
		candidates.push_back(point);
		candidates.push_back(point2);
		candidates.push_back(point3);
		candidates.push_back(point4);
		candidates.push_back(point5);
		candidates.push_back(point6);
	}

	auto resultSmem = mergeCandidatesTester(candidates, 2, SharedMemoryMerge, 2);
	auto resultNaive = mergeCandidatesTester(candidates, 2, NaiveMerge);
	compareResults(resultSmem, resultNaive);
}

TEST(testMergeCandidates, testMergeSmemSimple6points3chunk6Dim){
	auto candidates = std::vector<std::vector<bool>>();
	{
		auto point = std::vector<bool>({0,0,0,0,0,1});
		auto point2 = std::vector<bool>({0,0,0,0,1,0});
		auto point3 = std::vector<bool>({0,0,0,1,0,0});
		auto point4 = std::vector<bool>({0,0,1,0,0,0});
		auto point5 = std::vector<bool>({0,1,0,0,0,0});
		auto point6 = std::vector<bool>({1,0,0,0,0,0});
		candidates.push_back(point);
		candidates.push_back(point2);
		candidates.push_back(point3);
		candidates.push_back(point4);
		candidates.push_back(point5);
		candidates.push_back(point6);
	}

	auto resultSmem = mergeCandidatesTester(candidates, 2, SharedMemoryMerge, 2);
	auto resultNaive = mergeCandidatesTester(candidates, 2, NaiveMerge);
	compareResults(resultSmem, resultNaive);
}
  
 
TEST(testMergeCandidates, testMergeSmemSimple5points3chunkMissAligned){
	auto candidates = std::vector<std::vector<bool>>();
	{
		auto point = std::vector<bool>({0,0,0,0,1});
		auto point2 = std::vector<bool>({0,0,0,1,0});
		auto point3 = std::vector<bool>({0,0,1,0,0});
		auto point4 = std::vector<bool>({0,1,0,0,0});
		auto point5 = std::vector<bool>({1,0,0,0,0});
		candidates.push_back(point);
		candidates.push_back(point2);
		candidates.push_back(point3);
		candidates.push_back(point4);
		candidates.push_back(point5);
	}

	auto resultSmem = mergeCandidatesTester(candidates, 2, SharedMemoryMerge, 2);
	auto resultNaive = mergeCandidatesTester(candidates, 2, NaiveMerge);
	compareResults(resultSmem, resultNaive);
}



 
TEST(testMergeCandidates, testMergeSmemRandom){
	unsigned int dim = 4;
	unsigned int numberOfPoints = 8;
	std::mt19937 rng(std::random_device{}());
	rng.seed(0);
	auto candidates = std::vector<std::vector<bool>>();
	for(int i = 0; i < numberOfPoints; i++){
		auto point = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			point.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		candidates.push_back(point);
	}

	auto resultSmem = mergeCandidatesTester(candidates, 2, SharedMemoryMerge, 4);
	auto resultNaive = mergeCandidatesTester(candidates, 2, NaiveMerge);
	compareResults(resultSmem, resultNaive);
}

TEST(testMergeCandidates, testMergeSmemRandom2){
	unsigned int dim = 4;
	unsigned int numberOfPoints = 8;
	std::mt19937 rng(std::random_device{}());
	rng.seed(0);
	auto candidates = std::vector<std::vector<bool>>();
	for(int i = 0; i < numberOfPoints; i++){
		auto point = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			point.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		candidates.push_back(point);
	}

	auto resultSmem = mergeCandidatesTester(candidates, 2, SharedMemoryMerge, 8);
	auto resultNaive = mergeCandidatesTester(candidates, 2, NaiveMerge);
	compareResults(resultSmem, resultNaive);
}


TEST(testMergeCandidates, testMergeEarlyStoppingRandom){
	unsigned int dim = 4;
	unsigned int numberOfPoints = 8;
	std::mt19937 rng(std::random_device{}());
	rng.seed(0);
	auto candidates = std::vector<std::vector<bool>>();
	for(int i = 0; i < numberOfPoints; i++){
		auto point = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			point.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		candidates.push_back(point);
	}

	auto resultSmem = mergeCandidatesTester(candidates, 2, SharedMemoryMerge, 8);
	auto resultNaive = mergeCandidatesTester(candidates, 2, EarlyStoppingMerge);
	compareResults(resultSmem, resultNaive, 2);
}

TEST(testMergeCandidates, testMergeEarlyStoppingRandom2){
	unsigned int dim = 666;
	unsigned int numberOfPoints = 32;
	std::mt19937 rng(std::random_device{}());
	rng.seed(0);
	auto candidates = std::vector<std::vector<bool>>();
	for(int i = 0; i < numberOfPoints; i++){
		auto point = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			point.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		candidates.push_back(point);
	}

	auto resultSmem = mergeCandidatesTester(candidates, 2, SharedMemoryMerge, 8);
	auto resultNaive = mergeCandidatesTester(candidates, 2, EarlyStoppingMerge);
	compareResults(resultSmem, resultNaive, 2);
}




TEST(testMergeCandidates, testMergeSmemRandom3){
	unsigned int dim = 4;
	unsigned int numberOfPoints = 8;
	std::mt19937 rng(std::random_device{}());
	rng.seed(0);
	auto candidates = std::vector<std::vector<bool>>();
	for(int i = 0; i < numberOfPoints; i++){
		auto point = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			point.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		candidates.push_back(point);
	}

	auto resultSmem = mergeCandidatesTester(candidates, 2, SharedMemoryMerge, 1024);
	auto resultNaive = mergeCandidatesTester(candidates, 2, NaiveMerge);
	compareResults(resultSmem, resultNaive);
}


TEST(testMergeCandidates, testMergeSmemRandom4){
	unsigned int dim = 66;
	unsigned int numberOfPoints = 8;
	std::mt19937 rng(std::random_device{}());
	rng.seed(0);
	auto candidates = std::vector<std::vector<bool>>();
	for(int i = 0; i < numberOfPoints; i++){
		auto point = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			point.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		candidates.push_back(point);
	}

	auto resultSmem = mergeCandidatesTester(candidates, 2, SharedMemoryMerge, 1024);
	auto resultNaive = mergeCandidatesTester(candidates, 2, NaiveMerge);

	compareResults(resultSmem, resultNaive);
}



TEST(testMergeCandidates, testMergeSmemRandom5){
	unsigned int dim = 66;
	unsigned int numberOfPoints = 888;
	std::mt19937 rng(std::random_device{}());
	rng.seed(0);
	auto candidates = std::vector<std::vector<bool>>();
	for(int i = 0; i < numberOfPoints; i++){
		auto point = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			point.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		candidates.push_back(point);
	}

	auto resultSmem = mergeCandidatesTester(candidates, 2, SharedMemoryMerge, 1024);
	auto resultNaive = mergeCandidatesTester(candidates, 2, NaiveMerge);

	compareResults(resultSmem, resultNaive);
}


TEST(testMergeCandidates, testMergeSmemRandom6){
	unsigned int dim = 6;
	unsigned int numberOfPoints = 9;
	std::mt19937 rng(std::random_device{}());
	rng.seed(0);
	auto candidates = std::vector<std::vector<bool>>();
	for(int i = 0; i < numberOfPoints; i++){
		auto point = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			point.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		candidates.push_back(point);
	}

	auto resultSmem = mergeCandidatesTester(candidates, 2, SharedMemoryMerge, 8);
	auto resultNaive = mergeCandidatesTester(candidates, 2, NaiveMerge);

	compareResults(resultSmem, resultNaive);
}

TEST(testMergeCandidates, testMergeSmem6){
	unsigned int dim = 9;
	unsigned int numberOfPoints = 9;
	auto candidates = std::vector<std::vector<bool>>();
	for(int i = 0; i < numberOfPoints; i++){
		auto point = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			point.push_back((i == j));
		}
		candidates.push_back(point);
	}

	auto resultSmem = mergeCandidatesTester(candidates, 2, SharedMemoryMerge, 8);
	auto resultNaive = mergeCandidatesTester(candidates, 2, NaiveMerge);

	compareResults(resultSmem, resultNaive);
}
TEST(testMergeCandidates, testMergeSmem7){
	unsigned int dim = 15;
	unsigned int numberOfPoints = 15;
	auto candidates = std::vector<std::vector<bool>>();
	for(int i = 0; i < numberOfPoints; i++){
		auto point = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			point.push_back((i == j));
		}
		candidates.push_back(point);
	}

	auto resultSmem = mergeCandidatesTester(candidates, 2, SharedMemoryMerge, 8);
	auto resultNaive = mergeCandidatesTester(candidates, 2, NaiveMerge);

	compareResults(resultSmem, resultNaive);
}
TEST(testMergeCandidates, testMergeSmem8){
	unsigned int dim = 16;
	unsigned int numberOfPoints = 16;
	auto candidates = std::vector<std::vector<bool>>();
	for(int i = 0; i < numberOfPoints; i++){
		auto point = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			point.push_back((i == j));
		}
		candidates.push_back(point);
	}

	auto resultSmem = mergeCandidatesTester(candidates, 2, SharedMemoryMerge, 8);
	auto resultNaive = mergeCandidatesTester(candidates, 2, NaiveMerge);

	compareResults(resultSmem, resultNaive);
}

TEST(testMergeCandidates, testMergeSmemRandom7_3){
	unsigned int dim = 333;
	unsigned int numberOfPoints = 333;
	std::mt19937 rng(std::random_device{}());
	rng.seed(0);
	auto candidates = std::vector<std::vector<bool>>();
	for(int i = 0; i < numberOfPoints; i++){
		auto point = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			point.push_back(i == j);
		}
		candidates.push_back(point);
	}

	auto resultSmem = mergeCandidatesTester(candidates, 2, SharedMemoryMerge, 9999);
	auto resultNaive = mergeCandidatesTester(candidates, 2, NaiveMerge);

	compareResults(resultSmem, resultNaive);
}

TEST(testMergeCandidates, testMergeSmemRandom7_2){
	unsigned int dim = 33;
	unsigned int numberOfPoints = 1111;
	std::mt19937 rng(std::random_device{}());
	rng.seed(0);
	auto candidates = std::vector<std::vector<bool>>();
	for(int i = 0; i < numberOfPoints; i++){
		auto point = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			point.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		candidates.push_back(point);
	}

	auto resultSmem = mergeCandidatesTester(candidates, 2, SharedMemoryMerge, 9999);
	auto resultNaive = mergeCandidatesTester(candidates, 2, NaiveMerge);

	compareResults(resultSmem, resultNaive);
}

TEST(testMergeCandidates, _SUPER_SLOW_testMergeSmemRandom7_4){
	unsigned int dim = 1111;
	unsigned int numberOfPoints = 1111;
	std::mt19937 rng(std::random_device{}());
	rng.seed(0);
	auto candidates = std::vector<std::vector<bool>>();
	for(int i = 0; i < numberOfPoints; i++){
		auto point = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			point.push_back(i == j);
		}
		candidates.push_back(point);
	}

	auto resultSmem = mergeCandidatesTester(candidates, 2, SharedMemoryMerge, 9999);
	auto resultNaive = mergeCandidatesTester(candidates, 2, NaiveMerge);

	// auto res_s = std::vector<std::vector<bool>>();
	// auto res_n = std::vector<std::vector<bool>>();
	
	// for(int i = 0; i < numberOfPoints*(numberOfPoints+1)/2 -numberOfPoints; i++){
	// 	auto point_s = std::vector<bool>();
	// 	auto point_n = std::vector<bool>();
	// 	for(int j = 0; j < ceilf((float)dim/32); j++){
	// 		point_s.push_back(resultSmem.first.at(j*(numberOfPoints*(numberOfPoints+1)/2 -numberOfPoints) + i));
	// 		point_n.push_back(resultNaive.first.at(j*(numberOfPoints*(numberOfPoints+1)/2 -numberOfPoints) + i));
	// 	}
	// 	res_s.push_back(point_s);
	// 	res_n.push_back(point_n);
	// }
	// int count = 0;
	// for(int i = 0; i < res_n.size(); i++){
	// 	count = 0;
	// 	for(int j = 0; j < res_s.size(); j++){
	// 		bool found = true;
	// 		for(int k = 0; k < res_s.at(0).size(); k++){
	// 			found &= res_s.at(i).at(k) == res_n.at(j).at(k);
	// 		}
	// 		count += found;
	// 	}
	// 	EXPECT_EQ(count, 1);
	// }

	compareResults(resultSmem, resultNaive);
}


TEST(testMergeCandidates, testMergeSmemRandom7_5){
	unsigned int dim = 1111;
	unsigned int numberOfPoints = 11;
	std::mt19937 rng(std::random_device{}());
	rng.seed(0);
	auto candidates = std::vector<std::vector<bool>>();
	for(int i = 0; i < numberOfPoints; i++){
		auto point = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			point.push_back(i == j);
		}
		candidates.push_back(point);
	}

	auto resultSmem = mergeCandidatesTester(candidates, 2, SharedMemoryMerge, 9999);
	auto resultNaive = mergeCandidatesTester(candidates, 2, NaiveMerge);
	compareResults(resultSmem, resultNaive);
}


TEST(testMergeCandidates, testMergeSmemRandom7_6){
	unsigned int dim = 11111;
	unsigned int numberOfPoints = 11;
	std::mt19937 rng(std::random_device{}());
	rng.seed(0);
	auto candidates = std::vector<std::vector<bool>>();
	for(int i = 0; i < numberOfPoints; i++){
		auto point = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			point.push_back(i == j);
		}
		candidates.push_back(point);
	}

	auto resultSmem = mergeCandidatesTester(candidates, 2, SharedMemoryMerge, 9999);
	auto resultNaive = mergeCandidatesTester(candidates, 2, NaiveMerge);
	compareResults(resultSmem, resultNaive);
}

TEST(testMergeCandidates, testMergeSmemRandom7_7){
	unsigned int dim = 64;
	unsigned int numberOfPoints = 4;
	std::mt19937 rng(std::random_device{}());
	rng.seed(0);
	auto candidates = std::vector<std::vector<bool>>();
	for(int i = 0; i < numberOfPoints; i++){
		auto point = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			point.push_back(i == j);
		}
		candidates.push_back(point);
	}

	auto resultSmem = mergeCandidatesTester(candidates, 2, SharedMemoryMerge, 2);
	auto resultSmem2 = mergeCandidatesTester(candidates, 2, SharedMemoryMerge, 16);
	auto resultNaive = mergeCandidatesTester(candidates, 2, NaiveMerge);

	// unsigned int numberOfNewCandidates = numberOfPoints*(numberOfPoints+1)/2 - numberOfPoints;
	// for(int i = 0; i < numberOfNewCandidates; i++){
	// 	for(int j = 0; j < 2; j++){
	// 		for(int k = 0; k < 32; k++){
	// 			std::cout << readBit(resultSmem.first.at(j*numberOfNewCandidates +i),k);
	// 		}
	// 	}
	// 	std::cout << std::endl;
	// }
	// std::cout << std::endl;
	// for(int i = 0; i < numberOfNewCandidates; i++){
	// 	for(int j = 0; j < 2; j++){
	// 		for(int k = 0; k < 32; k++){
	// 			std::cout << readBit(resultSmem2.first.at(j*numberOfNewCandidates +i),k);
	// 		}
	// 	}
	// 	std::cout << std::endl;
	// }
	
	compareResults(resultSmem, resultNaive);
	compareResults(resultSmem, resultSmem2);
}


TEST(testMergeCandidates, _SUPER_SLOW_testMergeSmemRandom7){
	unsigned int dim = 333;
	unsigned int numberOfPoints = 1111;
	std::mt19937 rng(std::random_device{}());
	rng.seed(0);
	auto candidates = std::vector<std::vector<bool>>();
	for(int i = 0; i < numberOfPoints; i++){
		auto point = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			point.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		candidates.push_back(point);
	}

	auto resultSmem = mergeCandidatesTester(candidates, 2, SharedMemoryMerge, 512);
	auto resultNaive = mergeCandidatesTester(candidates, 2, NaiveMerge);

	compareResults(resultSmem, resultNaive);
}




TEST(testMergeCandidates, _SUPER_SLOW_testMergeSmemRandom8){
	unsigned int dim = 666;
	unsigned int numberOfPoints = 1111;
	std::mt19937 rng(std::random_device{}());
	rng.seed(0);
	auto candidates = std::vector<std::vector<bool>>();
	for(int i = 0; i < numberOfPoints; i++){
		auto point = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			point.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		candidates.push_back(point);
	}

	auto resultSmem = mergeCandidatesTester(candidates, 2, SharedMemoryMerge, 1024);
	auto resultNaive = mergeCandidatesTester(candidates, 2, NaiveMerge);

	compareResults(resultSmem, resultNaive);
}


TEST(testMergeCandidates, _SUPER_SLOW_testMergeSmemRandom9){
	unsigned int dim = 66;
	unsigned int numberOfPoints = 8888;
	std::mt19937 rng(std::random_device{}());
	rng.seed(0);
	auto candidates = std::vector<std::vector<bool>>();
	for(int i = 0; i < numberOfPoints; i++){
		auto point = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			point.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		candidates.push_back(point);
	}

	auto resultSmem = mergeCandidatesTester(candidates, 2, SharedMemoryMerge, 1024);
	auto resultNaive = mergeCandidatesTester(candidates, 2, NaiveMerge);

	compareResults(resultSmem, resultNaive);
}

TEST(testMergeCandidates, _SUPER_SLOW_testMergeSmemRandom10_Naive){
	unsigned int dim = 32*4;
	unsigned int numberOfPoints = 1000;
	std::mt19937 rng(std::random_device{}());
	rng.seed(0);
	auto candidates = std::vector<std::vector<bool>>();
	for(int i = 0; i < numberOfPoints; i++){
		auto point = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			point.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		candidates.push_back(point);
	}

	auto resultNaive = mergeCandidatesTester(candidates, 2, NaiveMerge);

}

TEST(testMergeCandidates, _SUPER_SLOW_testMergeSmemRandom10_Naive2){
	unsigned int dim = 32*4;
	unsigned int numberOfPoints = 3000;
	std::mt19937 rng(std::random_device{}());
	rng.seed(0);
	auto candidates = std::vector<std::vector<bool>>();
	for(int i = 0; i < numberOfPoints; i++){
		auto point = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			point.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		candidates.push_back(point);
	}

	auto resultNaive = mergeCandidatesTester(candidates, 2, NaiveMerge);

}

TEST(testMergeCandidates, _SUPER_SLOW_testMergeSmemRandom10_Smem1024){
	unsigned int dim = 32*4;
	unsigned int numberOfPoints = 3000;
	std::mt19937 rng(std::random_device{}());
	rng.seed(0);
	auto candidates = std::vector<std::vector<bool>>();
	for(int i = 0; i < numberOfPoints; i++){
		auto point = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			point.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		candidates.push_back(point);
	}

	auto resultSmem = mergeCandidatesTester(candidates, 2, SharedMemoryMerge, 1024);

}


TEST(testMergeCandidates, _SUPER_SLOW_testMergeSmemRandom10_Smem64){
	unsigned int dim = 32*4;
	unsigned int numberOfPoints = 3000;
	std::mt19937 rng(std::random_device{}());
	rng.seed(0);
	auto candidates = std::vector<std::vector<bool>>();
	for(int i = 0; i < numberOfPoints; i++){
		auto point = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			point.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		candidates.push_back(point);
	}

	auto resultSmem = mergeCandidatesTester(candidates, 2, SharedMemoryMerge, 64);

}


TEST(testMergeCandidates, _SUPER_SLOW_testMergeSmemRandom10_Smem32){
	unsigned int dim = 32*4;
	unsigned int numberOfPoints = 3000;
	std::mt19937 rng(std::random_device{}());
	rng.seed(0);
	auto candidates = std::vector<std::vector<bool>>();
	for(int i = 0; i < numberOfPoints; i++){
		auto point = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			point.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		candidates.push_back(point);
	}

	auto resultSmem = mergeCandidatesTester(candidates, 2, SharedMemoryMerge, 32);

}
