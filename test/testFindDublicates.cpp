#include <gtest/gtest.h>
#include "../src/MineClusGPU/FindDublicates.h"
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
	auto result4 = findDublicatesTester(candidates, Hash);


	EXPECT_EQ(result.size(), 2);
	EXPECT_EQ(result2.size(), 2);
	EXPECT_EQ(result3.size(), 2);

	EXPECT_EQ(result.at(0), 0);
	EXPECT_EQ(result.at(1), 1);

	EXPECT_EQ(result2.at(0), 0);
	EXPECT_EQ(result2.at(1), 1);

	EXPECT_EQ(result3.at(0), 0);
	EXPECT_EQ(result3.at(1), 1);

	EXPECT_EQ(result4.at(0), 0);
	EXPECT_EQ(result4.at(1), 1);
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
	auto result4 = findDublicatesTester(candidates, Hash);

	EXPECT_EQ(result.size(), 1);
	EXPECT_EQ(result2.size(), 1);
	EXPECT_EQ(result3.size(), 1);

	EXPECT_EQ(result.at(0), 0);

	EXPECT_EQ(result2.at(0), 0);

	EXPECT_EQ(result3.at(0), 0);

	EXPECT_EQ(result4.at(0), 0);
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
	auto result4 = findDublicatesTester(candidates, Hash);

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

	EXPECT_EQ(result4.at(0), 0);
	EXPECT_EQ(result4.at(1), 0);
	EXPECT_EQ(result4.at(2), 1);
}


TEST(testFindDublicates, test3Candidates_2){
	auto candidates = std::vector<std::vector<bool>>();

	{
		auto point = std::vector<bool>({0,1});
		candidates.push_back(point);
	}
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
	auto result4 = findDublicatesTester(candidates, Hash);

	EXPECT_EQ(result.size(), 3);
	EXPECT_EQ(result2.size(), 3);
	EXPECT_EQ(result3.size(), 3);

	EXPECT_EQ(result.at(0), 0);
	EXPECT_EQ(result.at(1), 1);
	EXPECT_EQ(result.at(2), 1);

	EXPECT_EQ(result2.at(0), 0);
	EXPECT_EQ(result2.at(1), 1);
	EXPECT_EQ(result2.at(2), 1);

	EXPECT_EQ(result3.at(0), 0);
	EXPECT_EQ(result3.at(1), 1);
	EXPECT_EQ(result3.at(2), 1);

	EXPECT_EQ(result4.at(0), 0);
	EXPECT_EQ(result4.at(1), 1);
	EXPECT_EQ(result4.at(2), 1);
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
	auto result4 = findDublicatesTester(candidates, Hash);
	
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
	unsigned int count = 0;
	for(unsigned int i = 0; i < 1000; i++){
		count += result4.at(i);
	}
	EXPECT_EQ(count,999);
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


TEST(testFindDublicates, test10000CandidatesIn200DimHash){
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
	auto result = findDublicatesTester(candidates, Hash);

	EXPECT_EQ(result.size(), 10010);

	unsigned int count = 0;

	
	for(unsigned int i = 0; i < 10000; i++){
		count +=result.at(i);
	}
		EXPECT_EQ(count, 10000-1);

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



TEST(testFindDublicates, testSimpleHash){
	// this thest should be like this
	// 001|001|000 = 1*1 + 2*1 + 3*0
	// 001|001|000 = 1*1 + 2*1 + 3*0
	// 000|000|100 = 1*0 + 2*0 + 3*1
	auto candidates = std::vector<std::vector<bool>>();
	{
		auto point = std::vector<bool>(96,0);
		EXPECT_EQ(point.size(), 96);
		point.at(0) = 1;
		point.at(32) = 1;
		candidates.push_back(point);
		candidates.push_back(point);
	}

	{
		auto point = std::vector<bool>(96,0);
		point.at(64) = 1;
		candidates.push_back(point);
	}
	
	auto result = findDublicatesTester(candidates, Hash);
	EXPECT_EQ(result.size(), 3);
	EXPECT_EQ(result.at(0), 0);
	EXPECT_EQ(result.at(1), 1);
	EXPECT_EQ(result.at(2), 0);
}


TEST(testFindDublicates, testSimpleNaive){
	auto candidates = std::vector<std::vector<bool>>();
	{
		auto point = std::vector<bool>(96,0);
		EXPECT_EQ(point.size(), 96);
		point.at(0) = 1;
		point.at(32) = 1;
		candidates.push_back(point);
		candidates.push_back(point);
	}

	{
		auto point = std::vector<bool>(96,0);
		point.at(64) = 1;
		candidates.push_back(point);
	}
	
	auto result = findDublicatesTester(candidates, Naive);
	EXPECT_EQ(result.size(), 3);
	EXPECT_EQ(result.at(0), 0);
	EXPECT_EQ(result.at(1), 1);
	EXPECT_EQ(result.at(2), 0);
}



TEST(testFindDublicates, testLarge){
	unsigned int dim = 1000;
	auto candidates = std::vector<std::vector<bool>>();
	for(int i = 0; i < dim; i++){
		auto point = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			point.push_back(j<i);
		}
		candidates.push_back(point);
	}
	
	auto result = findDublicatesTester(candidates, Hash);
	EXPECT_EQ(result.size(), dim);
	for(int i = 0; i < dim; i++){
		EXPECT_EQ(result.at(i), 0) << i;
	}
}


TEST(testFindDublicates, _SUPER_SLOW_testLarge2){
	unsigned int dim = 10000;
	auto candidates = std::vector<std::vector<bool>>();
	for(int i = 0; i < dim; i++){
		auto point = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			point.push_back(j<i);
		}
		candidates.push_back(point);
	}
	
	auto result = findDublicatesTester(candidates, Hash);
	EXPECT_EQ(result.size(), dim);
	for(int i = 0; i < dim; i++){
		EXPECT_EQ(result.at(i), 0) << i;
	}
}

TEST(testFindDublicates, _SUPER_SLOW_testLarge3){
	unsigned int dim = 10000;
	unsigned int copies = 50;
	auto candidates = std::vector<std::vector<bool>>();
	for(int k = 0; k < copies; k++){
		for(int i = 0; i < dim; i++){
			auto point = std::vector<bool>();
			for(int j = 0; j < dim; j++){
				point.push_back(j<i);
			}
			candidates.push_back(point);
		}
	}
	
	auto result = findDublicatesTester(candidates, Hash);
	EXPECT_EQ(result.size(), (copies)*dim);

	EXPECT_EQ(candidates.size(), dim*copies);
	auto counts = std::vector<unsigned int>(dim,0);
	

	for(int j = 0; j < copies; j++){
		for(int i = 0; i < dim; i++){
			counts.at(i) += result.at(j*dim+i);
		}
	}
	for(int i = 0; i < dim; i++){
		EXPECT_EQ(counts.at(i), copies-1) << i;
	}
}

TEST(testFindDublicates, _SUPER_SLOW_testLarge4MoreBreaking){
	unsigned int dim = 10000;
	unsigned int copies = 50;
	auto candidates = std::vector<std::vector<bool>>();
	for(int k = 0; k < copies; k++){
		for(int i = 0; i < dim; i++){
			auto point = std::vector<bool>();
			for(int j = 0; j < dim; j++){
				point.push_back(j==i);
			}
			candidates.push_back(point);
		}
	}
	EXPECT_EQ(candidates.size(), dim*copies);
	auto result = findDublicatesTester(candidates, MoreBreaking);
	EXPECT_EQ(result.size(), (copies)*dim);

	auto counts = std::vector<unsigned int>(dim,0);
	

	for(int j = 0; j < copies; j++){
		for(int i = 0; i < dim; i++){
			counts.at(i) += result.at(j*dim+i);
		}
	}
	for(int i = 0; i < dim; i++){
		EXPECT_EQ(counts.at(i), copies-1) << i;
	}
}

TEST(testFindDublicates, _SUPER_SLOW_testLarge4){
	unsigned int dim = 10000;
	unsigned int copies = 50;
	auto candidates = std::vector<std::vector<bool>>();
	for(int k = 0; k < copies; k++){
		for(int i = 0; i < dim; i++){
			auto point = std::vector<bool>();
			for(int j = 0; j < dim; j++){
				point.push_back(j==i);
			}
			candidates.push_back(point);
		}
	}
	EXPECT_EQ(candidates.size(), dim*copies);
	auto result = findDublicatesTester(candidates, Hash);
	EXPECT_EQ(result.size(), (copies)*dim);

	auto counts = std::vector<unsigned int>(dim,0);
	

	for(int j = 0; j < copies; j++){
		for(int i = 0; i < dim; i++){
			counts.at(i) += result.at(j*dim+i);
		}
	}
	for(int i = 0; i < dim; i++){
		EXPECT_EQ(counts.at(i), copies-1) << i;
	}
}


TEST(testFindDublicates, testLarge5){
	unsigned int dim = 100;
	unsigned int copies = 50;
	auto candidates = std::vector<std::vector<bool>>();
	for(int k = 0; k < copies; k++){
		for(int i = 0; i < dim; i++){
			auto point = std::vector<bool>();
			for(int j = 0; j < dim; j++){
				point.push_back(j<i);
			}
			candidates.push_back(point);
		}
	}
	
	auto result = findDublicatesTester(candidates, Hash);
	EXPECT_EQ(result.size(), (copies)*dim);

	EXPECT_EQ(candidates.size(), dim*copies);
	auto counts = std::vector<unsigned int>(dim,0);
	

	for(int j = 0; j < copies; j++){
		for(int i = 0; i < dim; i++){
			counts.at(i) += result.at(j*dim+i);
		}
	}
	for(int i = 0; i < dim; i++){
		EXPECT_EQ(counts.at(i), copies-1) << i;
	}
}


TEST(testFindDublicates, testLarge5Naive){
	unsigned int dim = 100;
	unsigned int copies = 50;
	auto candidates = std::vector<std::vector<bool>>();
	for(int k = 0; k < copies; k++){
		for(int i = 0; i < dim; i++){
			auto point = std::vector<bool>();
			for(int j = 0; j < dim; j++){
				point.push_back(j<i);
			}
			candidates.push_back(point);
		}
	}
	
	auto result = findDublicatesTester(candidates, Naive);
	EXPECT_EQ(result.size(), (copies)*dim);

	EXPECT_EQ(candidates.size(), dim*copies);
	auto counts = std::vector<unsigned int>(dim,0);
	

	for(int j = 0; j < copies; j++){
		for(int i = 0; i < dim; i++){
			counts.at(i) += result.at(j*dim+i);
		}
	}
	for(int i = 0; i < dim; i++){
		EXPECT_EQ(counts.at(i), copies-1) << i;
	}
}


TEST(testFindDublicates, _SUPER_SLOW_testLarge6){
	unsigned int dim = 100;
	unsigned int copies = 5000;
	std::default_random_engine gen;
	gen.seed(0);
	std::normal_distribution<double> dist(1.0,1.0);

	auto candidates = std::vector<std::vector<bool>>();
	for(int k = 0; k < copies; k++){
		for(int i = 0; i < dim; i++){
			auto point = std::vector<bool>();
			for(int j = 0; j < dim; j++){
				point.push_back(dist(gen) < -1.6);
			}
			candidates.push_back(point);
		}
	}

	
	EXPECT_EQ(candidates.size(), dim*copies);
	auto result = findDublicatesTester(candidates, Hash);
	EXPECT_EQ(result.size(), copies*dim);

	auto deletedResult = std::vector<std::vector<bool>>();
	for(int i = 0; i < copies*dim; i++){
		if(!result.at(i)){
			auto r = std::vector<bool>();
			for(int j = 0; j < dim; j++){
				r.push_back(candidates.at(i).at(j));
			}
			deletedResult.push_back(r);
		}
	}

	
	for(int i = 0; i < deletedResult.size(); i++){
		for(int j = 0; j < deletedResult.size(); j++){
			if(i != j){
				bool r = false;
				for(int k = 0; k < dim; k++){
					r |= (deletedResult.at(i).at(k) != deletedResult.at(j).at(k)); 
				}
				EXPECT_TRUE(r) << "i " << i << " j: " << j;
			}
		}		
	}
}


TEST(testFindDublicates, _SUPER_SLOW_testLarge6MoreBreaking){
	unsigned int dim = 100;
	unsigned int copies = 5000;
	std::default_random_engine gen;
	gen.seed(0);
	std::normal_distribution<double> dist(1.0,1.0);

	auto candidates = std::vector<std::vector<bool>>();
	for(int k = 0; k < copies; k++){
		for(int i = 0; i < dim; i++){
			auto point = std::vector<bool>();
			for(int j = 0; j < dim; j++){
				point.push_back(dist(gen) < -1.6);
			}
			candidates.push_back(point);
		}
	}

	
	EXPECT_EQ(candidates.size(), dim*copies);
	auto result = findDublicatesTester(candidates, MoreBreaking);
	EXPECT_EQ(result.size(), copies*dim);

	auto deletedResult = std::vector<std::vector<bool>>();
	for(int i = 0; i < copies*dim; i++){
		if(!result.at(i)){
			auto r = std::vector<bool>();
			for(int j = 0; j < dim; j++){
				r.push_back(candidates.at(i).at(j));
			}
			deletedResult.push_back(r);
		}
	}
	
	for(int i = 0; i < deletedResult.size(); i++){
		for(int j = 0; j < deletedResult.size(); j++){
			if(i != j){
				bool r = false;
				for(int k = 0; k < dim; k++){
					r |= (deletedResult.at(i).at(k) != deletedResult.at(j).at(k)); 
				}
				EXPECT_TRUE(r) << "i " << i << " j: " << j;
			}
		}		
	}
}


TEST(testFindDublicates, _SUPER_SLOW_testLarge6Comparison){
	unsigned int dim = 100;
	unsigned int copies = 5000;
	std::default_random_engine gen;
	gen.seed(0);
	std::normal_distribution<double> dist(1.0,1.0);

	auto candidates = std::vector<std::vector<bool>>();
	for(int k = 0; k < copies; k++){
		for(int i = 0; i < dim; i++){
			auto point = std::vector<bool>();
			for(int j = 0; j < dim; j++){
				point.push_back(dist(gen) < -1.6);
			}
			candidates.push_back(point);
		}
	}

	
	EXPECT_EQ(candidates.size(), dim*copies);
	auto result = findDublicatesTester(candidates, MoreBreaking);
	auto result2 = findDublicatesTester(candidates, Hash);
	EXPECT_EQ(result.size(), copies*dim);
	EXPECT_EQ(result2.size(), copies*dim);

	auto deletedResult = std::vector<std::vector<bool>>();
	for(int i = 0; i < copies*dim; i++){
		if(!result.at(i)){
			auto r = std::vector<bool>();
			for(int j = 0; j < dim; j++){
				r.push_back(candidates.at(i).at(j));
			}
			deletedResult.push_back(r);
		}
	}

	auto deletedResult2 = std::vector<std::vector<bool>>();
	for(int i = 0; i < copies*dim; i++){
		if(!result2.at(i)){
			auto r = std::vector<bool>();
			for(int j = 0; j < dim; j++){
				r.push_back(candidates.at(i).at(j));
			}
			deletedResult2.push_back(r);
		}
	}

	EXPECT_EQ(deletedResult.size(),deletedResult2.size());

	unsigned int count = 0;
	for(int i = 0; i < deletedResult.size(); i++){
		for(int j = 0; j < deletedResult.size(); j++){
			bool r3 = true;
			for(int k = 0; k < dim; k++){
				r3 &= deletedResult.at(i).at(k) == deletedResult2.at(j).at(k);
			}
			count += r3;
			if(i != j){
				bool r = false;
				bool r2 = false;
				for(int k = 0; k < dim; k++){
					r |= (deletedResult.at(i).at(k) != deletedResult.at(j).at(k));
					r2 |= (deletedResult2.at(i).at(k) != deletedResult2.at(j).at(k)); 
				}
				EXPECT_TRUE(r) << "i " << i << " j: " << j;
				EXPECT_TRUE(r2) << "i " << i << " j: " << j;
			}
		}		
	}
	EXPECT_EQ(count, deletedResult.size());
}


TEST(testFindDublicates, _SUPER_SLOW_testLarge6Comparison2){
	unsigned int dim = 10000;
	unsigned int copies = 50;
	std::default_random_engine gen;
	gen.seed(0);
	std::normal_distribution<double> dist(1.0,1.0);

	auto candidates = std::vector<std::vector<bool>>();
	for(unsigned long long int k = 0; k < dim; k++){
		auto point = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			point.push_back(dist(gen) < -1.6);
		}
		for(unsigned long long int i = 0; i < copies; i++){
			candidates.push_back(point);
		}
	}

	
	EXPECT_EQ(candidates.size(), dim*copies);
	auto result = findDublicatesTester(candidates, MoreBreaking);
	auto result2 = findDublicatesTester(candidates, Hash);
	EXPECT_EQ(result.size(), copies*dim);
	EXPECT_EQ(result2.size(), copies*dim);

	std::cout << "data created" << std::endl;

	auto deletedResult = std::vector<std::vector<bool>>();
	for(unsigned long long int i = 0; i < copies*dim; i++){
		if(!result.at(i)){
			auto r = std::vector<bool>();
			for(unsigned long long int j = 0; j < dim; j++){
				r.push_back(candidates.at(i).at(j));
			}
			deletedResult.push_back(r);
		}
	}

	std::cout << "delete results 1 are created" << std::endl;
	
	auto deletedResult2 = std::vector<std::vector<bool>>();
	for(unsigned long long int i = 0; i < copies*dim; i++){
		if(!result2.at(i)){
			auto r = std::vector<bool>();
			for(unsigned long long int j = 0; j < dim; j++){
				r.push_back(candidates.at(i).at(j));
			}
			deletedResult2.push_back(r);
		}
	}

	EXPECT_EQ(deletedResult.size(),deletedResult2.size());
	EXPECT_EQ(deletedResult.size(),dim);

	std::cout << "sizes are correct" << std::endl;
	
	std::cout << "delete results 2 are created" << std::endl;

	unsigned int count = 0;
	for(unsigned long long int i = 0; i < deletedResult.size(); i++){
		for(unsigned long long int j = 0; j < deletedResult.size(); j++){
			bool r3 = true;
			for(unsigned long long int k = 0; k < dim; k++){
				r3 &= deletedResult.at(i).at(k) == deletedResult2.at(j).at(k);
			}
			count += r3;
			if(i != j){
				bool r = false;
				bool r2 = false;
				for(unsigned long long int k = 0; k < dim; k++){
					r |= (deletedResult.at(i).at(k) != deletedResult.at(j).at(k));
					r2 |= (deletedResult2.at(i).at(k) != deletedResult2.at(j).at(k)); 
				}
				EXPECT_TRUE(r) << "i " << i << " j: " << j;
				EXPECT_TRUE(r2) << "i " << i << " j: " << j;
			}
		}		
	}
	EXPECT_EQ(count, deletedResult.size());
}
