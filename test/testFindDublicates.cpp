#include <gtest/gtest.h>
#include "../src/MineClus/MineClusKernels.h"
#include "testingTools.h"
#include <random>


TEST(testFindDublicates, test2Candidates){
	auto candidates = std::vector<std::vector<bool>>();

	{
		auto point = std::vector<bool>({0,1});
		candidates.push_back(point);
	}
	{
		auto point = std::vector<bool>({0,1});
		candidates.push_back(point);
	}

	auto result = findDublicatesTester(candidates, Naive);
	auto result2 = findDublicatesTester(candidates, Breaking);
	auto result3 = findDublicatesTester(candidates, MoreBreaking);


	EXPECT_EQ(result.size(), 2);
	EXPECT_EQ(result2.size(), 2);
	EXPECT_EQ(result3.size(), 2);

	EXPECT_EQ(result.at(0), 0);
	EXPECT_EQ(result.at(1), 1);

	EXPECT_EQ(result2.at(0), 0);
	EXPECT_EQ(result2.at(1), 1);

	EXPECT_EQ(result3.at(0), 0);
	EXPECT_EQ(result3.at(1), 1);
}

TEST(testFindDublicates, test1Candidate){
	auto candidates = std::vector<std::vector<bool>>();

	{
		auto point = std::vector<bool>({1,1});
		candidates.push_back(point);
	}


	auto result = findDublicatesTester(candidates, Naive);
	auto result2 = findDublicatesTester(candidates, Breaking);
	auto result3 = findDublicatesTester(candidates, MoreBreaking);


	EXPECT_EQ(result.size(), 1);
	EXPECT_EQ(result2.size(), 1);
	EXPECT_EQ(result3.size(), 1);

	EXPECT_EQ(result.at(0), 0);

	EXPECT_EQ(result2.at(0), 0);

	EXPECT_EQ(result3.at(0), 0);
}




TEST(testFindDublicates, test3Candidates){
	auto candidates = std::vector<std::vector<bool>>();

	{
		auto point = std::vector<bool>({0,1});
		candidates.push_back(point);
	}
	{
		auto point = std::vector<bool>({1,1});
		candidates.push_back(point);
	}
	{
		auto point = std::vector<bool>({0,1});
		candidates.push_back(point);
	}

	auto result = findDublicatesTester(candidates, Naive);
	auto result2 = findDublicatesTester(candidates, Breaking);
	auto result3 = findDublicatesTester(candidates, MoreBreaking);


	EXPECT_EQ(result.size(), 3);
	EXPECT_EQ(result2.size(), 3);
	EXPECT_EQ(result3.size(), 3);

	EXPECT_EQ(result.at(0), 0);
	EXPECT_EQ(result.at(1), 0);
	EXPECT_EQ(result.at(2), 1);

	EXPECT_EQ(result2.at(0), 0);
	EXPECT_EQ(result2.at(1), 0);
	EXPECT_EQ(result2.at(2), 1);

	EXPECT_EQ(result3.at(0), 0);
	EXPECT_EQ(result3.at(1), 0);
	EXPECT_EQ(result3.at(2), 1);
}



TEST(testFindDublicates, test1000Candidates){
	auto candidates = std::vector<std::vector<bool>>();

	for(int i = 0; i < 1000; i++){
		auto point = std::vector<bool>({0,1});
		candidates.push_back(point);
	}

	auto result = findDublicatesTester(candidates, Naive);
	auto result2 = findDublicatesTester(candidates, Breaking);
	auto result3 = findDublicatesTester(candidates, MoreBreaking);

	
	EXPECT_EQ(result.size(), 1000);
	EXPECT_EQ(result2.size(), 1000);
	EXPECT_EQ(result3.size(), 1000);
	
	EXPECT_EQ(result.at(0), 0);
	EXPECT_EQ(result2.at(0), 0);
	EXPECT_EQ(result3.at(0), 0);

	for(unsigned int i = 1; i < 1000; i++){
		EXPECT_EQ(result.at(i), 1);
		EXPECT_EQ(result2.at(i), 1);
		EXPECT_EQ(result3.at(i), 1);	
	}	
}


TEST(testFindDublicates, test100CandidatesIn200Dim){
	auto candidates = std::vector<std::vector<bool>>();

	for(int i = 0; i < 1000; i++){
		auto point = std::vector<bool>();
		for(int j = 0; j < 200; j++){
			point.push_back(j%20 == 0);
		}
		candidates.push_back(point);			

	}

	auto result = findDublicatesTester(candidates, Naive);
	auto result2 = findDublicatesTester(candidates, Breaking);
	auto result3 = findDublicatesTester(candidates, MoreBreaking);

	
	EXPECT_EQ(result.size(), 1000);
	EXPECT_EQ(result2.size(), 1000);
	EXPECT_EQ(result3.size(), 1000);
	
	EXPECT_EQ(result.at(0), 0);
	EXPECT_EQ(result2.at(0), 0);
	EXPECT_EQ(result3.at(0), 0);

	for(unsigned int i = 1; i < 1000; i++){
		EXPECT_EQ(result.at(i), 1);
		EXPECT_EQ(result2.at(i), 1);
		EXPECT_EQ(result3.at(i), 1);	
	}	
}


TEST(testFindDublicates, test100CandidatesIn200Dim2){
	auto candidates = std::vector<std::vector<bool>>();

	for(int i = 0; i < 1000; i++){
		auto point = std::vector<bool>();
		for(int j = 0; j < 200; j++){
			point.push_back(j%20 == 0);
		}
		candidates.push_back(point);			
	}
	for(int i = 0; i < 10; i++){
		auto point = std::vector<bool>();
		for(int j = 0; j < 200; j++){
			point.push_back(j%10 == 0);
		}
		candidates.push_back(point);			
	}
	auto result = findDublicatesTester(candidates, Naive);
	auto result2 = findDublicatesTester(candidates, Breaking);
	auto result3 = findDublicatesTester(candidates, MoreBreaking);

	
	EXPECT_EQ(result.size(), 1010);
	EXPECT_EQ(result2.size(), 1010);
	EXPECT_EQ(result3.size(), 1010);
	
	EXPECT_EQ(result.at(0), 0);
	EXPECT_EQ(result2.at(0), 0);
	EXPECT_EQ(result3.at(0), 0);

	for(unsigned int i = 1; i < 1000; i++){
		EXPECT_EQ(result.at(i), 1);
		EXPECT_EQ(result2.at(i), 1);
		EXPECT_EQ(result3.at(i), 1);	
	}

	EXPECT_EQ(result.at(1000), 0);
	EXPECT_EQ(result2.at(1000), 0);
	EXPECT_EQ(result3.at(1000), 0);
	
	for(unsigned int i = 1001; i < 1010; i++){
		EXPECT_EQ(result.at(i), 1);
		EXPECT_EQ(result2.at(i), 1);
		EXPECT_EQ(result3.at(i), 1);	
	}
}



TEST(testFindDublicates, _SUPER_SLOW_test10000CandidatesIn200DimNaive){
	auto candidates = std::vector<std::vector<bool>>();

	for(int i = 0; i < 10000; i++){
		auto point = std::vector<bool>();
		for(int j = 0; j < 200; j++){
			point.push_back(j%20 == 0);
		}
		candidates.push_back(point);			
	}
	for(int i = 0; i < 10; i++){
		auto point = std::vector<bool>();
		for(int j = 0; j < 200; j++){
			point.push_back(j%10 == 0);
		}
		candidates.push_back(point);			
	}
	auto result = findDublicatesTester(candidates, Naive);

	EXPECT_EQ(result.size(), 10010);
	
	EXPECT_EQ(result.at(0), 0);
	
	for(unsigned int i = 1; i < 10000; i++){
		EXPECT_EQ(result.at(i), 1);
	}

	EXPECT_EQ(result.at(10000), 0);
	
	for(unsigned int i = 10001; i < 10010; i++){
		EXPECT_EQ(result.at(i), 1);
	}
}


TEST(testFindDublicates, test10000CandidatesIn200DimBreaking){
	auto candidates = std::vector<std::vector<bool>>();

	for(int i = 0; i < 10000; i++){
		auto point = std::vector<bool>();
		for(int j = 0; j < 200; j++){
			point.push_back(j%20 == 0);
		}
		candidates.push_back(point);			
	}
	for(int i = 0; i < 10; i++){
		auto point = std::vector<bool>();
		for(int j = 0; j < 200; j++){
			point.push_back(j%10 == 0);
		}
		candidates.push_back(point);			
	}
	auto result = findDublicatesTester(candidates, Breaking);

	EXPECT_EQ(result.size(), 10010);
	
	EXPECT_EQ(result.at(0), 0);
	
	for(unsigned int i = 1; i < 10000; i++){
		EXPECT_EQ(result.at(i), 1);
	}

	EXPECT_EQ(result.at(10000), 0);
	
	for(unsigned int i = 10001; i < 10010; i++){
		EXPECT_EQ(result.at(i), 1);
	}
}



TEST(testFindDublicates, test10000CandidatesIn200DimMoreBreaking){
	auto candidates = std::vector<std::vector<bool>>();

	for(int i = 0; i < 10000; i++){
		auto point = std::vector<bool>();
		for(int j = 0; j < 200; j++){
			point.push_back(j%20 == 0);
		}
		candidates.push_back(point);			
	}
	for(int i = 0; i < 10; i++){
		auto point = std::vector<bool>();
		for(int j = 0; j < 200; j++){
			point.push_back(j%10 == 0);
		}
		candidates.push_back(point);			
	}
	auto result = findDublicatesTester(candidates, MoreBreaking);

	EXPECT_EQ(result.size(), 10010);
	
	EXPECT_EQ(result.at(0), 0);
	
	for(unsigned int i = 1; i < 10000; i++){
		EXPECT_EQ(result.at(i), 1);
	}

	EXPECT_EQ(result.at(10000), 0);
	
	for(unsigned int i = 10001; i < 10010; i++){
		EXPECT_EQ(result.at(i), 1);
	}
}



TEST(testFindDublicates, _SUPER_SLOW_test100000CandidatesIn200){
	auto candidates = std::vector<std::vector<bool>>();
	std::default_random_engine gen;
	gen.seed(0);
	std::normal_distribution<double> dist(1.0,1.0);
	
	for(int i = 0; i < 100000; i++){
		auto point = std::vector<bool>();
		for(int j = 0; j < 200; j++){
			point.push_back(dist(gen) < 1);
		}
		candidates.push_back(point);			
	}
	auto result0 = findDublicatesTester(candidates, Naive);
	auto result1 = findDublicatesTester(candidates, Breaking);
	auto result2 = findDublicatesTester(candidates, MoreBreaking);

	EXPECT_EQ(result0.size(), 100000);
	EXPECT_EQ(result1.size(), 100000);
	EXPECT_EQ(result2.size(), 100000);

	for(unsigned int i = 1; i < 100000; i++){
		EXPECT_EQ(result0.at(i), result1.at(i));
		EXPECT_EQ(result0.at(i), result2.at(i));
	}
}
