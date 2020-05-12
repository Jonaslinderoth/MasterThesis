#include <gtest/gtest.h>
#include "../src/MineClusGPU/CountSupport.h"
#include "testingTools.h"
#include <vector>
#include <random>


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

	EXPECT_FLOAT_EQ(support.size(), 2);
	EXPECT_FLOAT_EQ(support.at(0), 2);
	EXPECT_FLOAT_EQ(support.at(1), 2);

	EXPECT_FLOAT_EQ(score.size(), 2);
	EXPECT_FLOAT_EQ(score.at(0), mu(1,2, 0.25));
	EXPECT_FLOAT_EQ(score.at(1), mu(1,2, 0.25));


	EXPECT_FLOAT_EQ(toBeDeleted.size(), 2);
	EXPECT_FLOAT_EQ(toBeDeleted.at(0), false);
	EXPECT_FLOAT_EQ(toBeDeleted.at(1), false);
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

	EXPECT_FLOAT_EQ(support.size(), 3);
	EXPECT_FLOAT_EQ(support.at(0), 2);
	EXPECT_FLOAT_EQ(support.at(1), 0);
	EXPECT_FLOAT_EQ(support.at(2), 0);

	EXPECT_FLOAT_EQ(score.size(), 3);
	EXPECT_FLOAT_EQ(score.at(0), mu(1,2, 0.25));
	EXPECT_FLOAT_EQ(score.at(1), mu(0,0, 0.25));
	EXPECT_FLOAT_EQ(score.at(2), mu(0,0, 0.25));


	EXPECT_FLOAT_EQ(toBeDeleted.size(), 3);
	EXPECT_FLOAT_EQ(toBeDeleted.at(0), false);
	EXPECT_FLOAT_EQ(toBeDeleted.at(1), true);
	EXPECT_FLOAT_EQ(toBeDeleted.at(2), true);
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

	EXPECT_FLOAT_EQ(support.size(), 2);
	EXPECT_FLOAT_EQ(support.at(0), 2);
	EXPECT_FLOAT_EQ(support.at(1), 4);

	EXPECT_FLOAT_EQ(score.size(), 2);
	EXPECT_FLOAT_EQ(score.at(0), mu(1,2, 0.25));
	EXPECT_FLOAT_EQ(score.at(1), mu(1,4, 0.25));


	EXPECT_FLOAT_EQ(toBeDeleted.size(), 2);
	EXPECT_FLOAT_EQ(toBeDeleted.at(0), false);
	EXPECT_FLOAT_EQ(toBeDeleted.at(1), false);
}




TEST(testCountSupport, test2_Smem){
	auto itemSet = std::vector<std::vector<bool>>();
	auto candidates = std::vector<std::vector<bool>>();

	candidates.push_back({1,0});
	candidates.push_back({0,1});


	itemSet.push_back({1,0});
	itemSet.push_back({0,1});

	
	auto result = countSupportTester(candidates, itemSet, 1, 0.25, SmemCount);
	auto support = std::get<0>(result);
	auto score = std::get<1>(result);
	auto toBeDeleted = std::get<2>(result);

	EXPECT_FLOAT_EQ(support.size(), 2);
	EXPECT_FLOAT_EQ(support.at(0), 1);
	EXPECT_FLOAT_EQ(support.at(1), 1);

	EXPECT_FLOAT_EQ(score.size(), 2);
	EXPECT_FLOAT_EQ(score.at(0), mu(1,1, 0.25));
	EXPECT_FLOAT_EQ(score.at(1), mu(1,1, 0.25));


	EXPECT_FLOAT_EQ(toBeDeleted.size(), 2);
	EXPECT_FLOAT_EQ(toBeDeleted.at(0), false);
	EXPECT_FLOAT_EQ(toBeDeleted.at(1), false);
}

TEST(testCountSupport, test2_Smem_2){
	auto itemSet = std::vector<std::vector<bool>>();
	auto candidates = std::vector<std::vector<bool>>();

	candidates.push_back({1,0});
	candidates.push_back({0,1});


	itemSet.push_back({1,0});
	itemSet.push_back({0,1});
	itemSet.push_back({1,1});

	
	auto result = countSupportTester(candidates, itemSet, 1, 0.25, SmemCount);
	auto support = std::get<0>(result);
	auto score = std::get<1>(result);
	auto toBeDeleted = std::get<2>(result);

	EXPECT_FLOAT_EQ(support.size(), 2);
	EXPECT_FLOAT_EQ(support.at(0), 2);
	EXPECT_FLOAT_EQ(support.at(1), 2);

	EXPECT_FLOAT_EQ(score.size(), 2);
	EXPECT_FLOAT_EQ(score.at(0), mu(1,2, 0.25));
	EXPECT_FLOAT_EQ(score.at(1), mu(1,2, 0.25));


	EXPECT_FLOAT_EQ(toBeDeleted.size(), 2);
	EXPECT_FLOAT_EQ(toBeDeleted.at(0), false);
	EXPECT_FLOAT_EQ(toBeDeleted.at(1), false);
}


TEST(testCountSupport, test32_Smem){
	auto transactions = std::vector<std::vector<bool>>();
	auto candidates = std::vector<std::vector<bool>>();


	for(int i = 0; i < 32; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < 32; j++){
			candidate.push_back(i==j);
			transaction.push_back(i==j);
		}
		candidates.push_back(candidate);
		transactions.push_back(transaction);
	}

	auto result = countSupportTester(candidates, transactions, 1, 0.25, SmemCount);
	auto support = std::get<0>(result);
	auto score = std::get<1>(result);
	auto toBeDeleted = std::get<2>(result);

	EXPECT_FLOAT_EQ(support.size(), 32);
	EXPECT_FLOAT_EQ(score.size(), 32);
	EXPECT_FLOAT_EQ(toBeDeleted.size(), 32);
	for(int i = 0; i < 32; i++){
		EXPECT_FLOAT_EQ(support.at(i), 1) << i;
		EXPECT_FLOAT_EQ(score.at(i), mu(1,1, 0.25)) << i;
		EXPECT_FLOAT_EQ(toBeDeleted.at(i), false) << i;
	}

}

TEST(testCountSupport, test32_Smem_2){
	auto transactions = std::vector<std::vector<bool>>();
	auto candidates = std::vector<std::vector<bool>>();


	for(int i = 0; i < 32; i++){
		std::vector<bool> candidate = std::vector<bool>();
		std::vector<bool>  transaction = std::vector<bool>();
		for(int j = 0; j < 32; j++){
			candidate.push_back(i==j);
			transaction.push_back(i==j);
		}
		candidates.push_back(candidate);
		transactions.push_back(transaction);
	}

	auto result = countSupportTester(candidates, transactions, 2, 0.25, SmemCount);
	auto support = std::get<0>(result);
	auto score = std::get<1>(result);
	auto toBeDeleted = std::get<2>(result);

	EXPECT_FLOAT_EQ(support.size(), 32);
	EXPECT_FLOAT_EQ(score.size(), 32);
	EXPECT_FLOAT_EQ(toBeDeleted.size(), 32);
	for(int i = 0; i < 32; i++){
		EXPECT_FLOAT_EQ(support.at(i), 1);
		EXPECT_FLOAT_EQ(score.at(i), mu(1,1, 0.25));
		EXPECT_FLOAT_EQ(toBeDeleted.at(i), true);
	}
}




TEST(testCountSupport, test64_Smem){
	auto transactions = std::vector<std::vector<bool>>();
	auto candidates = std::vector<std::vector<bool>>();


	for(int i = 0; i < 64; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < 64; j++){
			candidate.push_back(i==j);
			transaction.push_back(i==j);
		}
		candidates.push_back(candidate);
		transactions.push_back(transaction);
	}

	auto result = countSupportTester(candidates, transactions, 2, 0.25, SmemCount);
	auto support = std::get<0>(result);
	auto score = std::get<1>(result);
	auto toBeDeleted = std::get<2>(result);

	EXPECT_FLOAT_EQ(support.size(), 64);
	EXPECT_FLOAT_EQ(score.size(), 64);
	EXPECT_FLOAT_EQ(toBeDeleted.size(), 64);
	for(int i = 0; i < 64; i++){
		EXPECT_FLOAT_EQ(support.at(i), 1) << i;
		EXPECT_FLOAT_EQ(score.at(i), mu(1,1, 0.25)) << i;
		EXPECT_FLOAT_EQ(toBeDeleted.at(i), true) << i;
	}
}


TEST(testCountSupport, test192_Smem_2){
	auto transactions = std::vector<std::vector<bool>>();
	auto candidates = std::vector<std::vector<bool>>();
	unsigned int n = 192; 

	for(int i = 0; i < n; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < n; j++){
			candidate.push_back(i==j);
			transaction.push_back(i==j);
		}
		candidates.push_back(candidate);
		transactions.push_back(transaction);
	}

	auto result = countSupportTester(candidates, transactions, 2, 0.25, SmemCount);
	auto support = std::get<0>(result);
	auto score = std::get<1>(result);
	auto toBeDeleted = std::get<2>(result);

	EXPECT_FLOAT_EQ(support.size(), n);
	EXPECT_FLOAT_EQ(score.size(), n);
	EXPECT_FLOAT_EQ(toBeDeleted.size(), n);
	for(int i = 0; i < n; i++){
		EXPECT_FLOAT_EQ(support.at(i), 1);
		EXPECT_FLOAT_EQ(score.at(i), mu(1,1, 0.25));
		EXPECT_FLOAT_EQ(toBeDeleted.at(i), true);
	}
}


TEST(testCountSupport, test384_Smem_2){
	auto transactions = std::vector<std::vector<bool>>();
	auto candidates = std::vector<std::vector<bool>>();
	unsigned int n = 384; 

	for(int i = 0; i < n; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < n; j++){
			candidate.push_back(i==j);
			transaction.push_back(i==j);
		}
		candidates.push_back(candidate);
		transactions.push_back(transaction);
	}

	auto result = countSupportTester(candidates, transactions, 2, 0.25, SmemCount);
	auto support = std::get<0>(result);
	auto score = std::get<1>(result);
	auto toBeDeleted = std::get<2>(result);

	EXPECT_FLOAT_EQ(support.size(), n);
	EXPECT_FLOAT_EQ(score.size(), n);
	EXPECT_FLOAT_EQ(toBeDeleted.size(), n);
	for(int i = 0; i < n; i++){
		ASSERT_EQ(support.at(i), 1);
		EXPECT_FLOAT_EQ(score.at(i), mu(1,1, 0.25));
		EXPECT_FLOAT_EQ(toBeDeleted.at(i), true);
	}
}
TEST(testCountSupport, test1024_Smem_2){
	auto transactions = std::vector<std::vector<bool>>();
	auto candidates = std::vector<std::vector<bool>>();
	unsigned int n = 1024; 

	for(int i = 0; i < n; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < n; j++){
			candidate.push_back(i==j);
			transaction.push_back(i==j);
		}
		candidates.push_back(candidate);
		transactions.push_back(transaction);
	}

	auto result = countSupportTester(candidates, transactions, 2, 0.25, SmemCount);
	auto support = std::get<0>(result);
	auto score = std::get<1>(result);
	auto toBeDeleted = std::get<2>(result);

	EXPECT_FLOAT_EQ(support.size(), n);
	EXPECT_FLOAT_EQ(score.size(), n);
	EXPECT_FLOAT_EQ(toBeDeleted.size(), n);
	for(int i = 0; i < n; i++){
		ASSERT_EQ(support.at(i), 1) << i;
		EXPECT_FLOAT_EQ(score.at(i), mu(1,1, 0.25)) << i;
		EXPECT_FLOAT_EQ(toBeDeleted.at(i), true) << i;
	}
}

TEST(testCountSupport, test33_Smem_2){
	auto transactions = std::vector<std::vector<bool>>();
	auto candidates = std::vector<std::vector<bool>>();
	unsigned int n = 33; 

	for(int i = 0; i < n; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < n; j++){
			candidate.push_back(i==j);
			transaction.push_back(i==j);
		}
		candidates.push_back(candidate);
		transactions.push_back(transaction);
	}

	auto result = countSupportTester(candidates, transactions, 2, 0.25, SmemCount);
	auto support = std::get<0>(result);
	auto score = std::get<1>(result);
	auto toBeDeleted = std::get<2>(result);

	EXPECT_FLOAT_EQ(support.size(), n);
	EXPECT_FLOAT_EQ(score.size(), n);
	EXPECT_FLOAT_EQ(toBeDeleted.size(), n);
	for(int i = 0; i < n; i++){
		ASSERT_EQ(support.at(i), 1) <<i;
		EXPECT_FLOAT_EQ(score.at(i), mu(1,1, 0.25))<<i;
		EXPECT_FLOAT_EQ(toBeDeleted.at(i), true)<<i;
	}
}


TEST(testCountSupport, test66_Smem_2){
	auto transactions = std::vector<std::vector<bool>>();
	auto candidates = std::vector<std::vector<bool>>();
	unsigned int n = 66; 

	for(int i = 0; i < n; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < n; j++){
			candidate.push_back(i==j);
			transaction.push_back(i==j);
		}
		candidates.push_back(candidate);
		transactions.push_back(transaction);
	}

	auto result = countSupportTester(candidates, transactions, 2, 0.25, SmemCount);
	auto support = std::get<0>(result);
	auto score = std::get<1>(result);
	auto toBeDeleted = std::get<2>(result);

	EXPECT_FLOAT_EQ(support.size(), n);
	EXPECT_FLOAT_EQ(score.size(), n);
	EXPECT_FLOAT_EQ(toBeDeleted.size(), n);
	for(int i = 0; i < n; i++){
		ASSERT_EQ(support.at(i), 1) <<i;
		EXPECT_FLOAT_EQ(score.at(i), mu(1,1, 0.25))<<i;
		EXPECT_FLOAT_EQ(toBeDeleted.at(i), true)<<i;
	}
}

TEST(testCountSupport, test666_Smem_2){
	auto transactions = std::vector<std::vector<bool>>();
	auto candidates = std::vector<std::vector<bool>>();
	unsigned int n = 666; 

	for(int i = 0; i < n; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < n; j++){
			candidate.push_back(i==j);
			transaction.push_back(i==j);
		}
		candidates.push_back(candidate);
		transactions.push_back(transaction);
	}

	auto result = countSupportTester(candidates, transactions, 2, 0.25, SmemCount);
	auto support = std::get<0>(result);
	auto score = std::get<1>(result);
	auto toBeDeleted = std::get<2>(result);

	EXPECT_FLOAT_EQ(support.size(), n);
	EXPECT_FLOAT_EQ(score.size(), n);
	EXPECT_FLOAT_EQ(toBeDeleted.size(), n);
	for(int i = 0; i < n; i++){
		ASSERT_EQ(support.at(i), 1) <<i;
		EXPECT_FLOAT_EQ(score.at(i), mu(1,1, 0.25))<<i;
		EXPECT_FLOAT_EQ(toBeDeleted.at(i), true)<<i;
	}
}


TEST(testCountSupport, test_dim384_candidates32_transactions32_Smem){
	auto transactions = std::vector<std::vector<bool>>();
	auto candidates = std::vector<std::vector<bool>>();
	unsigned int dim = 384;
	unsigned int nCandidates = 32;
	unsigned int nTransactions = 32;

	for(int i = 0; i < nCandidates; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			candidate.push_back(i==j);
		}
		candidates.push_back(candidate);
	}

	for(int i = 0; i < nTransactions; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			transaction.push_back(i==j);
		}
		transactions.push_back(transaction);
	}
	
	auto result = countSupportTester(candidates, transactions, 1, 0.25, SmemCount);
	auto support = std::get<0>(result);
	auto score = std::get<1>(result);
	auto toBeDeleted = std::get<2>(result);

	EXPECT_FLOAT_EQ(support.size(), nCandidates);
	EXPECT_FLOAT_EQ(score.size(), nCandidates);
	EXPECT_FLOAT_EQ(toBeDeleted.size(), nCandidates);
	for(int i = 0; i < nCandidates; i++){
		ASSERT_EQ(support.at(i), 1) <<i;
		EXPECT_FLOAT_EQ(score.at(i), mu(1,1, 0.25))<<i;
		EXPECT_FLOAT_EQ(toBeDeleted.at(i), false)<<i;
	}
}




TEST(testCountSupport, test_dim384_candidates32_transactions64_Smem){
	auto transactions = std::vector<std::vector<bool>>();
	auto candidates = std::vector<std::vector<bool>>();
	unsigned int dim = 384;
	unsigned int nCandidates = 32;
	unsigned int nTransactions = 64;

	for(int i = 0; i < nCandidates; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			candidate.push_back(i==j);
		}
		candidates.push_back(candidate);
	}

	for(int i = 0; i < nTransactions; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			transaction.push_back(i==j);
		}
		transactions.push_back(transaction);
	}
	
	auto resultSmem = countSupportTester(candidates, transactions, 1, 0.25, SmemCount);
	auto resultNaive = countSupportTester(candidates, transactions, 1, 0.25);

	EXPECT_FLOAT_EQ(std::get<0>(resultSmem).size(), std::get<0>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<1>(resultSmem).size(), std::get<1>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<2>(resultSmem).size(), std::get<2>(resultNaive).size());

	for(unsigned int i = 0; i < std::get<2>(resultSmem).size(); i++){
		EXPECT_FLOAT_EQ(std::get<0>(resultSmem).at(i), std::get<0>(resultNaive).at(i)) << i;
		EXPECT_FLOAT_EQ(std::get<1>(resultSmem).at(i), std::get<1>(resultNaive).at(i)) << i;
		EXPECT_FLOAT_EQ(std::get<2>(resultSmem).at(i), std::get<2>(resultNaive).at(i)) << i;		
	}
	
}






TEST(testCountSupport, test_dim384_candidates32_transactions32_Random_Smem){
	auto transactions = std::vector<std::vector<bool>>();
	auto candidates = std::vector<std::vector<bool>>();
	std::mt19937 rng(std::random_device{}());
	rng.seed(0);
	unsigned int dim = 384;
	unsigned int nCandidates = 32;
	unsigned int nTransactions = 32;

	for(int i = 0; i < nCandidates; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			candidate.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		candidates.push_back(candidate);
	}

	for(int i = 0; i < nTransactions; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			transaction.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		transactions.push_back(transaction);
	}
	
	auto resultSmem = countSupportTester(candidates, transactions, 1, 0.25, SmemCount);
	auto resultNaive = countSupportTester(candidates, transactions, 1, 0.25);

	EXPECT_FLOAT_EQ(std::get<0>(resultSmem).size(), std::get<0>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<1>(resultSmem).size(), std::get<1>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<2>(resultSmem).size(), std::get<2>(resultNaive).size());

	for(unsigned int i = 0; i < std::get<2>(resultSmem).size(); i++){
		EXPECT_FLOAT_EQ(std::get<0>(resultSmem).at(i), std::get<0>(resultNaive).at(i));
		EXPECT_FLOAT_EQ(std::get<1>(resultSmem).at(i), std::get<1>(resultNaive).at(i));
		EXPECT_FLOAT_EQ(std::get<2>(resultSmem).at(i), std::get<2>(resultNaive).at(i));		
	}
}





TEST(testCountSupport, test_dim384_candidates32_transactions64_Random_Smem){
	auto transactions = std::vector<std::vector<bool>>();
	auto candidates = std::vector<std::vector<bool>>();
	std::mt19937 rng(std::random_device{}());
	rng.seed(0);
	unsigned int dim = 384;
	unsigned int nCandidates = 32;
	unsigned int nTransactions = 64;

	for(int i = 0; i < nCandidates; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			candidate.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		candidates.push_back(candidate);
	}

	for(int i = 0; i < nTransactions; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			transaction.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		transactions.push_back(transaction);
	}
	
	auto resultSmem = countSupportTester(candidates, transactions, 1, 0.25, SmemCount);
	auto resultNaive = countSupportTester(candidates, transactions, 1, 0.25);

	EXPECT_FLOAT_EQ(std::get<0>(resultSmem).size(), std::get<0>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<1>(resultSmem).size(), std::get<1>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<2>(resultSmem).size(), std::get<2>(resultNaive).size());

	for(unsigned int i = 0; i < std::get<2>(resultSmem).size(); i++){
		EXPECT_FLOAT_EQ(std::get<0>(resultSmem).at(i), std::get<0>(resultNaive).at(i));
		EXPECT_FLOAT_EQ(std::get<1>(resultSmem).at(i), std::get<1>(resultNaive).at(i));
		EXPECT_FLOAT_EQ(std::get<2>(resultSmem).at(i), std::get<2>(resultNaive).at(i));		
	}
}



TEST(testCountSupport, test_dim384_candidates32_transactions1024_Random_Smem){
	auto transactions = std::vector<std::vector<bool>>();
	auto candidates = std::vector<std::vector<bool>>();
	std::mt19937 rng(std::random_device{}());
	rng.seed(0);
	unsigned int dim = 384;
	unsigned int nCandidates = 32;
	unsigned int nTransactions = 1024;

	for(int i = 0; i < nCandidates; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			candidate.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		candidates.push_back(candidate);
	}

	for(int i = 0; i < nTransactions; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			transaction.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		transactions.push_back(transaction);
	}
	
	auto resultSmem = countSupportTester(candidates, transactions, 1, 0.25, SmemCount);
	auto resultNaive = countSupportTester(candidates, transactions, 1, 0.25);

	EXPECT_FLOAT_EQ(std::get<0>(resultSmem).size(), std::get<0>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<1>(resultSmem).size(), std::get<1>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<2>(resultSmem).size(), std::get<2>(resultNaive).size());

	for(unsigned int i = 0; i < std::get<2>(resultSmem).size(); i++){
		EXPECT_FLOAT_EQ(std::get<0>(resultSmem).at(i), std::get<0>(resultNaive).at(i))<<i;
		EXPECT_FLOAT_EQ(std::get<1>(resultSmem).at(i), std::get<1>(resultNaive).at(i))<<i;
		EXPECT_FLOAT_EQ(std::get<2>(resultSmem).at(i), std::get<2>(resultNaive).at(i))<<i;		
	}
}


TEST(testCountSupport, test_dim384_candidates2048_transactions1024_Random_Smem){
	auto transactions = std::vector<std::vector<bool>>();
	auto candidates = std::vector<std::vector<bool>>();
	std::mt19937 rng(std::random_device{}());
	rng.seed(0);
	unsigned int dim = 384;
	unsigned int nCandidates = 2048;
	unsigned int nTransactions = 1024;

	for(int i = 0; i < nCandidates; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			candidate.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		candidates.push_back(candidate);
	}

	for(int i = 0; i < nTransactions; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			transaction.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		transactions.push_back(transaction);
	}
	
	auto resultSmem = countSupportTester(candidates, transactions, 1, 0.25, SmemCount);
	auto resultNaive = countSupportTester(candidates, transactions, 1, 0.25);

	EXPECT_FLOAT_EQ(std::get<0>(resultSmem).size(), std::get<0>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<1>(resultSmem).size(), std::get<1>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<2>(resultSmem).size(), std::get<2>(resultNaive).size());

	for(unsigned int i = 0; i < std::get<2>(resultSmem).size(); i++){
		EXPECT_FLOAT_EQ(std::get<0>(resultSmem).at(i), std::get<0>(resultNaive).at(i)) <<i;
		EXPECT_FLOAT_EQ(std::get<1>(resultSmem).at(i), std::get<1>(resultNaive).at(i)) <<i;
		EXPECT_FLOAT_EQ(std::get<2>(resultSmem).at(i), std::get<2>(resultNaive).at(i)) <<i;		
	}

}

TEST(testCountSupport, test_dim384_candidates48_transactions1024_Random_Smem){
	auto transactions = std::vector<std::vector<bool>>();
	auto candidates = std::vector<std::vector<bool>>();
	std::mt19937 rng(std::random_device{}());
	rng.seed(0);
	unsigned int dim = 384;
	unsigned int nCandidates = 48;
	unsigned int nTransactions = 1024;

	for(int i = 0; i < nCandidates; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			candidate.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		candidates.push_back(candidate);
	}

	for(int i = 0; i < nTransactions; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			transaction.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		transactions.push_back(transaction);
	}
	
	auto resultSmem = countSupportTester(candidates, transactions, 1, 0.25, SmemCount);
	auto resultNaive = countSupportTester(candidates, transactions, 1, 0.25);

	EXPECT_FLOAT_EQ(std::get<0>(resultSmem).size(), std::get<0>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<1>(resultSmem).size(), std::get<1>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<2>(resultSmem).size(), std::get<2>(resultNaive).size());

	for(unsigned int i = 0; i < std::get<2>(resultSmem).size(); i++){
		EXPECT_FLOAT_EQ(std::get<0>(resultSmem).at(i), std::get<0>(resultNaive).at(i));
		EXPECT_FLOAT_EQ(std::get<1>(resultSmem).at(i), std::get<1>(resultNaive).at(i));
		EXPECT_FLOAT_EQ(std::get<2>(resultSmem).at(i), std::get<2>(resultNaive).at(i));		
	}
}


TEST(testCountSupport, test_dim384_candidates96_transactions1024_Random_Smem){
	auto transactions = std::vector<std::vector<bool>>();
	auto candidates = std::vector<std::vector<bool>>();
	std::mt19937 rng(std::random_device{}());
	rng.seed(0);
	unsigned int dim = 384;
	unsigned int nCandidates = 96;
	unsigned int nTransactions = 1024;

	for(int i = 0; i < nCandidates; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			candidate.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		candidates.push_back(candidate);
	}

	for(int i = 0; i < nTransactions; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			transaction.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		transactions.push_back(transaction);
	}
	
	auto resultSmem = countSupportTester(candidates, transactions, 1, 0.25, SmemCount);
	auto resultNaive = countSupportTester(candidates, transactions, 1, 0.25);

	EXPECT_FLOAT_EQ(std::get<0>(resultSmem).size(), std::get<0>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<1>(resultSmem).size(), std::get<1>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<2>(resultSmem).size(), std::get<2>(resultNaive).size());

	for(unsigned int i = 0; i < std::get<2>(resultSmem).size(); i++){
		EXPECT_FLOAT_EQ(std::get<0>(resultSmem).at(i), std::get<0>(resultNaive).at(i));
		EXPECT_FLOAT_EQ(std::get<1>(resultSmem).at(i), std::get<1>(resultNaive).at(i));
		EXPECT_FLOAT_EQ(std::get<2>(resultSmem).at(i), std::get<2>(resultNaive).at(i));		
	}
}




TEST(testCountSupport, test_dim384_candidates1025_transactions1024_Random_Smem){
	auto transactions = std::vector<std::vector<bool>>();
	auto candidates = std::vector<std::vector<bool>>();
	std::mt19937 rng(std::random_device{}());
	rng.seed(0);
	unsigned int dim = 384;
	unsigned int nCandidates = 1025;
	unsigned int nTransactions = 1024;

	for(int i = 0; i < nCandidates; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			candidate.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		candidates.push_back(candidate);
	}

	for(int i = 0; i < nTransactions; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			transaction.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		transactions.push_back(transaction);
	}
	
	auto resultSmem = countSupportTester(candidates, transactions, 1, 0.25, SmemCount);
	auto resultNaive = countSupportTester(candidates, transactions, 1, 0.25);

	EXPECT_FLOAT_EQ(std::get<0>(resultSmem).size(), std::get<0>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<1>(resultSmem).size(), std::get<1>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<2>(resultSmem).size(), std::get<2>(resultNaive).size());

	for(unsigned int i = 0; i < std::get<2>(resultSmem).size(); i++){
		EXPECT_FLOAT_EQ(std::get<0>(resultSmem).at(i), std::get<0>(resultNaive).at(i)) << i;
		EXPECT_FLOAT_EQ(std::get<1>(resultSmem).at(i), std::get<1>(resultNaive).at(i)) << i;
		EXPECT_FLOAT_EQ(std::get<2>(resultSmem).at(i), std::get<2>(resultNaive).at(i)) << i;		
	}
}


TEST(testCountSupport, test_dim384_candidates1025_transactions32_Random_Smem){
	auto transactions = std::vector<std::vector<bool>>();
	auto candidates = std::vector<std::vector<bool>>();
	std::mt19937 rng(std::random_device{}());
	rng.seed(0);
	unsigned int dim = 384;
	unsigned int nCandidates = 1025;
	unsigned int nTransactions = 32;

	for(int i = 0; i < nCandidates; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			candidate.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		candidates.push_back(candidate);
	}

	for(int i = 0; i < nTransactions; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			transaction.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		transactions.push_back(transaction);
	}
	
	auto resultSmem = countSupportTester(candidates, transactions, 1, 0.25, SmemCount);
	auto resultNaive = countSupportTester(candidates, transactions, 1, 0.25);

	EXPECT_FLOAT_EQ(std::get<0>(resultSmem).size(), std::get<0>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<1>(resultSmem).size(), std::get<1>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<2>(resultSmem).size(), std::get<2>(resultNaive).size());

	for(unsigned int i = 0; i < std::get<2>(resultSmem).size(); i++){
		EXPECT_FLOAT_EQ(std::get<0>(resultSmem).at(i), std::get<0>(resultNaive).at(i)) << i;
		EXPECT_FLOAT_EQ(std::get<1>(resultSmem).at(i), std::get<1>(resultNaive).at(i)) << i;
		EXPECT_FLOAT_EQ(std::get<2>(resultSmem).at(i), std::get<2>(resultNaive).at(i)) << i;		
	}
}

TEST(testCountSupport, test_dim384_candidates33_transactions32_Random_Smem){
	auto transactions = std::vector<std::vector<bool>>();
	auto candidates = std::vector<std::vector<bool>>();
	std::mt19937 rng(std::random_device{}());
	rng.seed(0);
	unsigned int dim = 384;
	unsigned int nCandidates = 33;
	unsigned int nTransactions = 32;

	for(int i = 0; i < nCandidates; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			candidate.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		candidates.push_back(candidate);
	}

	for(int i = 0; i < nTransactions; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			transaction.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		transactions.push_back(transaction);
	}
	
	auto resultSmem = countSupportTester(candidates, transactions, 1, 0.25, SmemCount);
	auto resultNaive = countSupportTester(candidates, transactions, 1, 0.25);

	EXPECT_FLOAT_EQ(std::get<0>(resultSmem).size(), std::get<0>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<1>(resultSmem).size(), std::get<1>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<2>(resultSmem).size(), std::get<2>(resultNaive).size());

	for(unsigned int i = 0; i < std::get<2>(resultSmem).size(); i++){
		EXPECT_FLOAT_EQ(std::get<0>(resultSmem).at(i), std::get<0>(resultNaive).at(i)) << i;
		EXPECT_FLOAT_EQ(std::get<1>(resultSmem).at(i), std::get<1>(resultNaive).at(i)) << i;
		EXPECT_FLOAT_EQ(std::get<2>(resultSmem).at(i), std::get<2>(resultNaive).at(i)) << i;		
	}
}

TEST(testCountSupport, test_dim384_candidates1024_transactions1024_Random_Smem){
	auto transactions = std::vector<std::vector<bool>>();
	auto candidates = std::vector<std::vector<bool>>();
	std::mt19937 rng(std::random_device{}());
	rng.seed(0);
	unsigned int dim = 384;
	unsigned int nCandidates = 1024;
	unsigned int nTransactions = 1024;

	for(int i = 0; i < nCandidates; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			candidate.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		candidates.push_back(candidate);
	}

	for(int i = 0; i < nTransactions; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			transaction.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		transactions.push_back(transaction);
	}
	
	auto resultSmem = countSupportTester(candidates, transactions, 1, 0.25, SmemCount);
	auto resultNaive = countSupportTester(candidates, transactions, 1, 0.25);

	EXPECT_FLOAT_EQ(std::get<0>(resultSmem).size(), std::get<0>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<1>(resultSmem).size(), std::get<1>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<2>(resultSmem).size(), std::get<2>(resultNaive).size());

	for(unsigned int i = 0; i < std::get<2>(resultSmem).size(); i++){
		EXPECT_FLOAT_EQ(std::get<0>(resultSmem).at(i), std::get<0>(resultNaive).at(i));
		EXPECT_FLOAT_EQ(std::get<1>(resultSmem).at(i), std::get<1>(resultNaive).at(i));
		EXPECT_FLOAT_EQ(std::get<2>(resultSmem).at(i), std::get<2>(resultNaive).at(i));		
	}
}



TEST(testCountSupport, test_dim384_candidates32_transactions69_Random_Smem){
	auto transactions = std::vector<std::vector<bool>>();
	auto candidates = std::vector<std::vector<bool>>();
	std::mt19937 rng(std::random_device{}());
	rng.seed(0);
	unsigned int dim = 384;
	unsigned int nCandidates = 32;
	unsigned int nTransactions = 69;

	for(int i = 0; i < nCandidates; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			candidate.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		candidates.push_back(candidate);
	}

	for(int i = 0; i < nTransactions; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			transaction.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		transactions.push_back(transaction);
	}
	
	auto resultSmem = countSupportTester(candidates, transactions, 1, 0.25, SmemCount);
	auto resultNaive = countSupportTester(candidates, transactions, 1, 0.25);

	EXPECT_FLOAT_EQ(std::get<0>(resultSmem).size(), std::get<0>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<1>(resultSmem).size(), std::get<1>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<2>(resultSmem).size(), std::get<2>(resultNaive).size());

	for(unsigned int i = 0; i < std::get<2>(resultSmem).size(); i++){
		EXPECT_FLOAT_EQ(std::get<0>(resultSmem).at(i), std::get<0>(resultNaive).at(i));
		EXPECT_FLOAT_EQ(std::get<1>(resultSmem).at(i), std::get<1>(resultNaive).at(i));
		EXPECT_FLOAT_EQ(std::get<2>(resultSmem).at(i), std::get<2>(resultNaive).at(i));		
	}
}


TEST(testCountSupport, test_dim384_candidates64_transactions32_Random_Smem){
	auto transactions = std::vector<std::vector<bool>>();
	auto candidates = std::vector<std::vector<bool>>();
	std::mt19937 rng(std::random_device{}());
	rng.seed(0);
	unsigned int dim = 384;
	unsigned int nCandidates = 64;
	unsigned int nTransactions = 32;

	for(int i = 0; i < nCandidates; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			candidate.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		candidates.push_back(candidate);
	}

	for(int i = 0; i < nTransactions; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			transaction.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		transactions.push_back(transaction);
	}
	
	auto resultSmem = countSupportTester(candidates, transactions, 1, 0.25, SmemCount);
	auto resultNaive = countSupportTester(candidates, transactions, 1, 0.25);

	EXPECT_FLOAT_EQ(std::get<0>(resultSmem).size(), std::get<0>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<1>(resultSmem).size(), std::get<1>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<2>(resultSmem).size(), std::get<2>(resultNaive).size());

	for(unsigned int i = 0; i < std::get<2>(resultSmem).size(); i++){
		EXPECT_FLOAT_EQ(std::get<0>(resultSmem).at(i), std::get<0>(resultNaive).at(i));
		EXPECT_FLOAT_EQ(std::get<1>(resultSmem).at(i), std::get<1>(resultNaive).at(i));
		EXPECT_FLOAT_EQ(std::get<2>(resultSmem).at(i), std::get<2>(resultNaive).at(i));		
	}
}



TEST(testCountSupport, test_dim384_candidates69_transactions32_Random_Smem){
	auto transactions = std::vector<std::vector<bool>>();
	auto candidates = std::vector<std::vector<bool>>();
	std::mt19937 rng(std::random_device{}());
	rng.seed(0);
	unsigned int dim = 384;
	unsigned int nCandidates = 69;
	unsigned int nTransactions = 32;

	for(int i = 0; i < nCandidates; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			candidate.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		candidates.push_back(candidate);
	}

	for(int i = 0; i < nTransactions; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			transaction.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		transactions.push_back(transaction);
	}
	
	auto resultSmem = countSupportTester(candidates, transactions, 1, 0.25, SmemCount);
	auto resultNaive = countSupportTester(candidates, transactions, 1, 0.25);

	EXPECT_FLOAT_EQ(std::get<0>(resultSmem).size(), std::get<0>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<1>(resultSmem).size(), std::get<1>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<2>(resultSmem).size(), std::get<2>(resultNaive).size());

	for(unsigned int i = 0; i < std::get<2>(resultSmem).size(); i++){
		EXPECT_FLOAT_EQ(std::get<0>(resultSmem).at(i), std::get<0>(resultNaive).at(i));
		EXPECT_FLOAT_EQ(std::get<1>(resultSmem).at(i), std::get<1>(resultNaive).at(i));
		EXPECT_FLOAT_EQ(std::get<2>(resultSmem).at(i), std::get<2>(resultNaive).at(i));		
	}
}

TEST(testCountSupport, test_dim39_candidates69_transactions32_Random_Smem){
	auto transactions = std::vector<std::vector<bool>>();
	auto candidates = std::vector<std::vector<bool>>();
	std::mt19937 rng(std::random_device{}());
	rng.seed(0);
	unsigned int dim = 39;
	unsigned int nCandidates = 69;
	unsigned int nTransactions = 32;

	for(int i = 0; i < nCandidates; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			candidate.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		candidates.push_back(candidate);
	}

	for(int i = 0; i < nTransactions; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			transaction.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		transactions.push_back(transaction);
	}
	
	auto resultSmem = countSupportTester(candidates, transactions, 1, 0.25, SmemCount);
	auto resultNaive = countSupportTester(candidates, transactions, 1, 0.25);

	EXPECT_FLOAT_EQ(std::get<0>(resultSmem).size(), std::get<0>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<1>(resultSmem).size(), std::get<1>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<2>(resultSmem).size(), std::get<2>(resultNaive).size());

	for(unsigned int i = 0; i < std::get<2>(resultSmem).size(); i++){
		EXPECT_FLOAT_EQ(std::get<0>(resultSmem).at(i), std::get<0>(resultNaive).at(i));
		EXPECT_FLOAT_EQ(std::get<1>(resultSmem).at(i), std::get<1>(resultNaive).at(i));
		EXPECT_FLOAT_EQ(std::get<2>(resultSmem).at(i), std::get<2>(resultNaive).at(i));		
	}
}



TEST(testCountSupport, test_dim192_candidates69_transactions32_Random_Smem){
	auto transactions = std::vector<std::vector<bool>>();
	auto candidates = std::vector<std::vector<bool>>();
	std::mt19937 rng(std::random_device{}());
	rng.seed(0);
	unsigned int dim = 39;
	unsigned int nCandidates = 64;
	unsigned int nTransactions = 64;

	for(int i = 0; i < nCandidates; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			candidate.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		candidates.push_back(candidate);
	}

	for(int i = 0; i < nTransactions; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			transaction.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		transactions.push_back(transaction);
	}
	
	auto resultSmem = countSupportTester(candidates, transactions, 1, 0.25, SmemCount);
	auto resultNaive = countSupportTester(candidates, transactions, 1, 0.25);

	EXPECT_FLOAT_EQ(std::get<0>(resultSmem).size(), std::get<0>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<1>(resultSmem).size(), std::get<1>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<2>(resultSmem).size(), std::get<2>(resultNaive).size());

	for(unsigned int i = 0; i < std::get<2>(resultSmem).size(); i++){
		EXPECT_FLOAT_EQ(std::get<0>(resultSmem).at(i), std::get<0>(resultNaive).at(i));
		EXPECT_FLOAT_EQ(std::get<1>(resultSmem).at(i), std::get<1>(resultNaive).at(i));
		EXPECT_FLOAT_EQ(std::get<2>(resultSmem).at(i), std::get<2>(resultNaive).at(i));		
	}
}

TEST(testCountSupport, test_dim666_candidates69_transactions32_Random_Smem){
	auto transactions = std::vector<std::vector<bool>>();
	auto candidates = std::vector<std::vector<bool>>();
	std::mt19937 rng(std::random_device{}());
	rng.seed(0);
	unsigned int dim = 666;
	unsigned int nCandidates = 64;
	unsigned int nTransactions = 64;

	for(int i = 0; i < nCandidates; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			candidate.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		candidates.push_back(candidate);
	}

	for(int i = 0; i < nTransactions; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			transaction.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		transactions.push_back(transaction);
	}
	
	auto resultSmem = countSupportTester(candidates, transactions, 1, 0.25, SmemCount);
	auto resultNaive = countSupportTester(candidates, transactions, 1, 0.25);

	EXPECT_FLOAT_EQ(std::get<0>(resultSmem).size(), std::get<0>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<1>(resultSmem).size(), std::get<1>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<2>(resultSmem).size(), std::get<2>(resultNaive).size());

	for(unsigned int i = 0; i < std::get<2>(resultSmem).size(); i++){
		EXPECT_FLOAT_EQ(std::get<0>(resultSmem).at(i), std::get<0>(resultNaive).at(i));
		EXPECT_FLOAT_EQ(std::get<1>(resultSmem).at(i), std::get<1>(resultNaive).at(i));
		EXPECT_FLOAT_EQ(std::get<2>(resultSmem).at(i), std::get<2>(resultNaive).at(i));		
	}
}


TEST(testCountSupport, test6666_Smem_2){
	auto transactions = std::vector<std::vector<bool>>();
	auto candidates = std::vector<std::vector<bool>>();
	unsigned int n = 6666; 

	for(int i = 0; i < n; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < n; j++){
			candidate.push_back(i==j);
			transaction.push_back(i==j);
		}
		candidates.push_back(candidate);
		transactions.push_back(transaction);
	}

	auto result = countSupportTester(candidates, transactions, 1, 0.25, SmemCount);
	auto support = std::get<0>(result);
	auto score = std::get<1>(result);
	auto toBeDeleted = std::get<2>(result);

	EXPECT_FLOAT_EQ(support.size(), n);
	EXPECT_FLOAT_EQ(score.size(), n);
	EXPECT_FLOAT_EQ(toBeDeleted.size(), n);
	for(int i = 0; i < n; i++){
		ASSERT_EQ(support.at(i), 1) <<i;
		EXPECT_FLOAT_EQ(score.at(i), mu(1,1, 0.25))<<i;
		EXPECT_FLOAT_EQ(toBeDeleted.at(i), false)<<i;
	}
}





TEST(testCountSupport, SLOW_test_largeRandom_compare){
	// CountSupport Dim: 4096 numberOfCandidates: 2048 numPoints: 512      
	auto transactions = std::vector<std::vector<bool>>();
	auto candidates = std::vector<std::vector<bool>>();
	std::mt19937 rng(std::random_device{}());
	rng.seed(0);
	unsigned int dim = 4096;
	unsigned int nCandidates = 2048;
	unsigned int nTransactions = 512;

	for(int i = 0; i < nCandidates; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			candidate.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		candidates.push_back(candidate);
	}

	for(int i = 0; i < nTransactions; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			transaction.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		transactions.push_back(transaction);
	}
	
	auto resultSmem = countSupportTester(candidates, transactions, 1, 0.25, SmemCount);
	auto resultNaive = countSupportTester(candidates, transactions, 1, 0.25);

	EXPECT_FLOAT_EQ(std::get<0>(resultSmem).size(), std::get<0>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<1>(resultSmem).size(), std::get<1>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<2>(resultSmem).size(), std::get<2>(resultNaive).size());

	for(unsigned int i = 0; i < std::get<2>(resultSmem).size(); i++){
		EXPECT_FLOAT_EQ(std::get<0>(resultSmem).at(i), std::get<0>(resultNaive).at(i));
		EXPECT_FLOAT_EQ(std::get<1>(resultSmem).at(i), std::get<1>(resultNaive).at(i));
		EXPECT_FLOAT_EQ(std::get<2>(resultSmem).at(i), std::get<2>(resultNaive).at(i));		
	}
}



TEST(testCountSupport, SLOW_test_largeRandom_compare2){
	// CountSupport Dim: 4096 numberOfCandidates: 2048 numPoints: 512      
	auto transactions = std::vector<std::vector<bool>>();
	auto candidates = std::vector<std::vector<bool>>();
	std::mt19937 rng(std::random_device{}());
	rng.seed(0);
	unsigned int dim = 4096*4;
	unsigned int nCandidates = 2048;
	unsigned int nTransactions = 512;

	for(int i = 0; i < nCandidates; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			candidate.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		candidates.push_back(candidate);
	}

	for(int i = 0; i < nTransactions; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			transaction.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		transactions.push_back(transaction);
	}
	
	auto resultSmem = countSupportTester(candidates, transactions, 1, 0.25, SmemCount);
	auto resultNaive = countSupportTester(candidates, transactions, 1, 0.25);

	EXPECT_FLOAT_EQ(std::get<0>(resultSmem).size(), std::get<0>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<1>(resultSmem).size(), std::get<1>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<2>(resultSmem).size(), std::get<2>(resultNaive).size());

	for(unsigned int i = 0; i < std::get<2>(resultSmem).size(); i++){
		EXPECT_FLOAT_EQ(std::get<0>(resultSmem).at(i), std::get<0>(resultNaive).at(i));
		EXPECT_FLOAT_EQ(std::get<1>(resultSmem).at(i), std::get<1>(resultNaive).at(i));
		EXPECT_FLOAT_EQ(std::get<2>(resultSmem).at(i), std::get<2>(resultNaive).at(i));		
	}
}




TEST(testCountSupport, SLOW_test_largeRandom_compare3){
	// CountSupport Dim: 4096 numberOfCandidates: 2048 numPoints: 512      
	auto transactions = std::vector<std::vector<bool>>();
	auto candidates = std::vector<std::vector<bool>>();
	std::mt19937 rng(std::random_device{}());
	rng.seed(0);
	unsigned int dim = 4096*4;
	unsigned int nCandidates = 2048*4;
	unsigned int nTransactions = 512;

	for(int i = 0; i < nCandidates; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			candidate.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		candidates.push_back(candidate);
	}

	for(int i = 0; i < nTransactions; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			transaction.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		transactions.push_back(transaction);
	}
	
	auto resultSmem = countSupportTester(candidates, transactions, 1, 0.25, SmemCount);
	auto resultNaive = countSupportTester(candidates, transactions, 1, 0.25);

	EXPECT_FLOAT_EQ(std::get<0>(resultSmem).size(), std::get<0>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<1>(resultSmem).size(), std::get<1>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<2>(resultSmem).size(), std::get<2>(resultNaive).size());

	for(unsigned int i = 0; i < std::get<2>(resultSmem).size(); i++){
		EXPECT_FLOAT_EQ(std::get<0>(resultSmem).at(i), std::get<0>(resultNaive).at(i));
		EXPECT_FLOAT_EQ(std::get<1>(resultSmem).at(i), std::get<1>(resultNaive).at(i));
		EXPECT_FLOAT_EQ(std::get<2>(resultSmem).at(i), std::get<2>(resultNaive).at(i));		
	}
}



TEST(testCountSupport, SLOW_test_largeRandom_compare4){
	// CountSupport Dim: 4096 numberOfCandidates: 2048 numPoints: 512      
	auto transactions = std::vector<std::vector<bool>>();
	auto candidates = std::vector<std::vector<bool>>();
	std::mt19937 rng(std::random_device{}());
	rng.seed(0);
	unsigned int dim = 4096*4;
	unsigned int nCandidates = 2048*4;
	unsigned int nTransactions = 512*4;

	for(int i = 0; i < nCandidates; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			candidate.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		candidates.push_back(candidate);
	}

	for(int i = 0; i < nTransactions; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			transaction.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		transactions.push_back(transaction);
	}
	
	auto resultSmem = countSupportTester(candidates, transactions, 1, 0.25, SmemCount);
	auto resultNaive = countSupportTester(candidates, transactions, 1, 0.25);

	EXPECT_FLOAT_EQ(std::get<0>(resultSmem).size(), std::get<0>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<1>(resultSmem).size(), std::get<1>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<2>(resultSmem).size(), std::get<2>(resultNaive).size());

	for(unsigned int i = 0; i < std::get<2>(resultSmem).size(); i++){
		EXPECT_FLOAT_EQ(std::get<0>(resultSmem).at(i), std::get<0>(resultNaive).at(i));
		EXPECT_FLOAT_EQ(std::get<1>(resultSmem).at(i), std::get<1>(resultNaive).at(i));
		EXPECT_FLOAT_EQ(std::get<2>(resultSmem).at(i), std::get<2>(resultNaive).at(i));		
	}
}



TEST(testCountSupport, SLOW_test_largeRandom_compare5){
	// CountSupport Dim: 4096 numberOfCandidates: 2048 numPoints: 512      
	auto transactions = std::vector<std::vector<bool>>();
	auto candidates = std::vector<std::vector<bool>>();
	std::mt19937 rng(std::random_device{}());
	rng.seed(0);
	unsigned int dim = 4096*4;
	unsigned int nCandidates = 2048*4;
	unsigned int nTransactions = 4;

	for(int i = 0; i < nCandidates; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			candidate.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		candidates.push_back(candidate);
	}

	for(int i = 0; i < nTransactions; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			transaction.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		transactions.push_back(transaction);
	}
	
	auto resultSmem = countSupportTester(candidates, transactions, 1, 0.25, SmemCount);
	auto resultNaive = countSupportTester(candidates, transactions, 1, 0.25);

	EXPECT_FLOAT_EQ(std::get<0>(resultSmem).size(), std::get<0>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<1>(resultSmem).size(), std::get<1>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<2>(resultSmem).size(), std::get<2>(resultNaive).size());

	for(unsigned int i = 0; i < std::get<2>(resultSmem).size(); i++){
		EXPECT_FLOAT_EQ(std::get<0>(resultSmem).at(i), std::get<0>(resultNaive).at(i));
		EXPECT_FLOAT_EQ(std::get<1>(resultSmem).at(i), std::get<1>(resultNaive).at(i));
		EXPECT_FLOAT_EQ(std::get<2>(resultSmem).at(i), std::get<2>(resultNaive).at(i));		
	}
}




TEST(testCountSupport, SLOW_test_largeRandom_compare6){
	// CountSupport Dim: 4096 numberOfCandidates: 2048 numPoints: 512      
	auto transactions = std::vector<std::vector<bool>>();
	auto candidates = std::vector<std::vector<bool>>();
	std::mt19937 rng(std::random_device{}());
	rng.seed(0);
	unsigned int dim = 4096*4;
	unsigned int nCandidates = 4;
	unsigned int nTransactions = 4*32;

	for(int i = 0; i < nCandidates; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			candidate.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		candidates.push_back(candidate);
	}

	for(int i = 0; i < nTransactions; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			transaction.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		transactions.push_back(transaction);
	}
	
	auto resultSmem = countSupportTester(candidates, transactions, 1, 0.25, SmemCount);
	auto resultNaive = countSupportTester(candidates, transactions, 1, 0.25);

	EXPECT_FLOAT_EQ(std::get<0>(resultSmem).size(), std::get<0>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<1>(resultSmem).size(), std::get<1>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<2>(resultSmem).size(), std::get<2>(resultNaive).size());

	for(unsigned int i = 0; i < std::get<2>(resultSmem).size(); i++){
		EXPECT_FLOAT_EQ(std::get<0>(resultSmem).at(i), std::get<0>(resultNaive).at(i));
		EXPECT_FLOAT_EQ(std::get<1>(resultSmem).at(i), std::get<1>(resultNaive).at(i));
		EXPECT_FLOAT_EQ(std::get<2>(resultSmem).at(i), std::get<2>(resultNaive).at(i));		
	}
}



TEST(testCountSupport, test_largeRandom_compare7){
	// CountSupport Dim: 4096 numberOfCandidates: 2048 numPoints: 512      
	auto transactions = std::vector<std::vector<bool>>();
	auto candidates = std::vector<std::vector<bool>>();
	std::mt19937 rng(std::random_device{}());
	rng.seed(0);
	unsigned int dim = 512;
	unsigned int nCandidates = 256;
	unsigned int nTransactions = 256;

	for(int i = 0; i < nCandidates; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			candidate.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		candidates.push_back(candidate);
	}

	for(int i = 0; i < nTransactions; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			transaction.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);
		}
		transactions.push_back(transaction);
	}
	
	auto resultSmem = countSupportTester(candidates, transactions, 1, 0.25, SmemCount);
	auto resultNaive = countSupportTester(candidates, transactions, 1, 0.25);

	EXPECT_FLOAT_EQ(std::get<0>(resultSmem).size(), std::get<0>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<1>(resultSmem).size(), std::get<1>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<2>(resultSmem).size(), std::get<2>(resultNaive).size());

	for(unsigned int i = 0; i < std::get<2>(resultSmem).size(); i++){
		EXPECT_FLOAT_EQ(std::get<0>(resultSmem).at(i), std::get<0>(resultNaive).at(i));
		EXPECT_FLOAT_EQ(std::get<1>(resultSmem).at(i), std::get<1>(resultNaive).at(i));
		EXPECT_FLOAT_EQ(std::get<2>(resultSmem).at(i), std::get<2>(resultNaive).at(i));		
	}
}


TEST(testCountSupport, test_largeRandom_compare8){
	// CountSupport Dim: 4096 numberOfCandidates: 2048 numPoints: 512      
	auto transactions = std::vector<std::vector<bool>>();
	auto candidates = std::vector<std::vector<bool>>();
	std::mt19937 rng(std::random_device{}());
	rng.seed(0);
	unsigned int dim = 512;
	unsigned int nCandidates = 256;
	unsigned int nTransactions = 256;

	for(int i = 0; i < nCandidates; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			candidate.push_back(((i+j) % 10) == 0);
		}
		candidates.push_back(candidate);
	}

	for(int i = 0; i < nTransactions; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			if(j > dim){
				transaction.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);	
			}else{
				bool value = (j) == i;
				if(value){
					transaction.push_back(1);
				}else{
					transaction.push_back(0);
				}	
			}
		}
		transactions.push_back(transaction);
	}
	
	auto resultSmem = countSupportTester(candidates, transactions, 1, 0.25, SmemCount);
	auto resultNaive = countSupportTester(candidates, transactions, 1, 0.25);

	EXPECT_FLOAT_EQ(std::get<0>(resultSmem).size(), std::get<0>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<1>(resultSmem).size(), std::get<1>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<2>(resultSmem).size(), std::get<2>(resultNaive).size());

	for(unsigned int i = 0; i < std::get<2>(resultSmem).size(); i++){
		EXPECT_FLOAT_EQ(std::get<0>(resultSmem).at(i), std::get<0>(resultNaive).at(i));
		EXPECT_FLOAT_EQ(std::get<1>(resultSmem).at(i), std::get<1>(resultNaive).at(i));
		EXPECT_FLOAT_EQ(std::get<2>(resultSmem).at(i), std::get<2>(resultNaive).at(i));		
	}
}

TEST(testCountSupport, test_largeRandom_compare9){
	// CountSupport Dim: 4096 numberOfCandidates: 2048 numPoints: 512      
	auto transactions = std::vector<std::vector<bool>>();
	auto candidates = std::vector<std::vector<bool>>();
	std::mt19937 rng(std::random_device{}());
	rng.seed(0);
	unsigned int dim = 256;
	unsigned int nCandidates = 256;
	unsigned int nTransactions = 1;

	for(int i = 0; i < nCandidates; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			candidate.push_back(((i+j) % 10) == 0);
		}
		candidates.push_back(candidate);
	}

	for(int i = 0; i < nTransactions; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			if(j > dim){
				transaction.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);	
			}else{
				bool value = (j) == i;
				if(value){
					transaction.push_back(1);
				}else{
					transaction.push_back(0);
				}	
			}
		}
		transactions.push_back(transaction);
	}
	
	auto resultSmem = countSupportTester(candidates, transactions, 1, 0.25, SmemCount);
	auto resultNaive = countSupportTester(candidates, transactions, 1, 0.25);

	EXPECT_FLOAT_EQ(std::get<0>(resultSmem).size(), std::get<0>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<1>(resultSmem).size(), std::get<1>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<2>(resultSmem).size(), std::get<2>(resultNaive).size());

	for(unsigned int i = 0; i < std::get<2>(resultSmem).size(); i++){
		EXPECT_FLOAT_EQ(std::get<0>(resultSmem).at(i), std::get<0>(resultNaive).at(i));
		EXPECT_FLOAT_EQ(std::get<1>(resultSmem).at(i), std::get<1>(resultNaive).at(i));
		EXPECT_FLOAT_EQ(std::get<2>(resultSmem).at(i), std::get<2>(resultNaive).at(i));		
	}
}


TEST(testCountSupport, test_largeRandom_compare10){
	// CountSupport Dim: 4096 numberOfCandidates: 2048 numPoints: 512      
	auto transactions = std::vector<std::vector<bool>>();
	auto candidates = std::vector<std::vector<bool>>();
	std::mt19937 rng(std::random_device{}());
	rng.seed(0);
	unsigned int dim = 32;
	unsigned int nCandidates = 1;
	unsigned int nTransactions = 1024*2;

	for(int i = 0; i < nCandidates; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			candidate.push_back(((i+j) % 10) == 0);
		}
		candidates.push_back(candidate);
	}

	for(int i = 0; i < nTransactions; i++){
		auto candidate = std::vector<bool>();
		auto transaction = std::vector<bool>();
		for(int j = 0; j < dim; j++){
			if(j > dim){
				transaction.push_back((std::uniform_int_distribution<int>{}(rng)) & 1);	
			}else{
				bool value = (j) == i;
				if(value){
					transaction.push_back(1);
				}else{
					transaction.push_back(0);
				}	
			}
		}
		transactions.push_back(transaction);
	}
	
	auto resultSmem = countSupportTester(candidates, transactions, 1, 0.25, SmemCount);
	auto resultNaive = countSupportTester(candidates, transactions, 1, 0.25);

	EXPECT_FLOAT_EQ(std::get<0>(resultSmem).size(), std::get<0>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<1>(resultSmem).size(), std::get<1>(resultNaive).size());
	EXPECT_FLOAT_EQ(std::get<2>(resultSmem).size(), std::get<2>(resultNaive).size());

	for(unsigned int i = 0; i < std::get<2>(resultSmem).size(); i++){
		EXPECT_FLOAT_EQ(std::get<0>(resultSmem).at(i), std::get<0>(resultNaive).at(i));
		EXPECT_FLOAT_EQ(std::get<1>(resultSmem).at(i), std::get<1>(resultNaive).at(i));
		EXPECT_FLOAT_EQ(std::get<2>(resultSmem).at(i), std::get<2>(resultNaive).at(i));		
	}
}

