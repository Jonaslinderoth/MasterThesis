#include <gtest/gtest.h>
#include "../src/MineClus/MuApriori/MuApriori.h"
#include <vector>
#include <boost/dynamic_bitset.hpp>
#include <random>

TEST(testMuApriori, testConstructor){
	std::vector<boost::dynamic_bitset<>>* itemSet = new std::vector<boost::dynamic_bitset<>>;
	auto a = MuApriori(itemSet, 1);
	EXPECT_EQ(a.getBeta(), 0.25);
	SUCCEED();
}



TEST(testMuApriori, testConstructor2){
	std::vector<boost::dynamic_bitset<>>* itemSet = new std::vector<boost::dynamic_bitset<>>;
	boost::dynamic_bitset<> point = boost::dynamic_bitset<>();
	point.push_back(0);
	itemSet->push_back(point);
	auto a = MuApriori(itemSet, 1);
	EXPECT_EQ(a.getBeta(), 0.25);
	SUCCEED();
}



TEST(testMuApriori, testCreateInitialCandidates){
	std::vector<boost::dynamic_bitset<>>* itemSet = new std::vector<boost::dynamic_bitset<>>;

	boost::dynamic_bitset<> point = boost::dynamic_bitset<>();
	point.push_back(0);
	itemSet->push_back(point);
	auto a = MuApriori(itemSet, 1);
	EXPECT_EQ(a.getBeta(), 0.25);
	a.createInitialCandidates();
	auto b = a.getBest();
	EXPECT_EQ(b, nullptr);
	SUCCEED();
}

TEST(testMuApriori, testCreateInitialCandidates2){
	std::vector<boost::dynamic_bitset<>>* itemSet = new std::vector<boost::dynamic_bitset<>>;

	boost::dynamic_bitset<> point = boost::dynamic_bitset<>();
	point.push_back(0);
	point.push_back(1);
	itemSet->push_back(point);
	auto a = MuApriori(itemSet, 1);
	EXPECT_EQ(a.getBeta(), 0.25);
	a.createInitialCandidates();
	auto b = a.getBest();
	EXPECT_NE(b , nullptr);
	EXPECT_EQ(b->support, 1);
	EXPECT_EQ(b->score, a.mu(1,1));
	SUCCEED();
}


TEST(testMuApriori, testCreateInitialCandidates3){
	std::vector<boost::dynamic_bitset<>>* itemSet = new std::vector<boost::dynamic_bitset<>>;

	{boost::dynamic_bitset<> point = boost::dynamic_bitset<>();
	point.push_back(0);
	point.push_back(1);
	point.push_back(1);
	itemSet->push_back(point);}

	{boost::dynamic_bitset<> point = boost::dynamic_bitset<>();
	point.push_back(0);
	point.push_back(0);
	point.push_back(1);
	itemSet->push_back(point);}
	
	auto a = MuApriori(itemSet, 1);
	EXPECT_EQ(a.getBeta(), 0.25);
	auto b = a.createInitialCandidates();
	EXPECT_EQ(b->size(), 2);
	EXPECT_EQ(b->at(0)->support, 1);
	EXPECT_EQ(b->at(0)->score, a.mu(1,1));
	EXPECT_EQ(b->at(1)->support, 2);
	EXPECT_EQ(b->at(1)->score, a.mu(2,1));
	SUCCEED();
}


TEST(testMuApriori, testFindBest1){
	std::vector<boost::dynamic_bitset<>>* itemSet = new std::vector<boost::dynamic_bitset<>>;
	boost::dynamic_bitset<> point1 = boost::dynamic_bitset<>();
	boost::dynamic_bitset<> point2 = boost::dynamic_bitset<>();
	point1.push_back(0);
	point1.push_back(1);
	itemSet->push_back(point1);
	point2.push_back(1);
	point2.push_back(0);
	itemSet->push_back(point2);
	auto a = MuApriori(itemSet, 1);
	EXPECT_EQ(a.getBeta(), 0.25);
	a.findBest(1);
	auto b = a.getBest();
	EXPECT_NE(b , nullptr);
	EXPECT_EQ(b->support, 1);
	EXPECT_EQ(b->score, a.mu(1,1));	
}

TEST(testMuApriori, testFindBest2){
	std::vector<boost::dynamic_bitset<>>* itemSet = new std::vector<boost::dynamic_bitset<>>;

	boost::dynamic_bitset<> point1 = boost::dynamic_bitset<>();
	boost::dynamic_bitset<> point2 = boost::dynamic_bitset<>();
	point1.push_back(0);
	point1.push_back(1);
	itemSet->push_back(point1);

	point2.push_back(1);
	point2.push_back(0);
	itemSet->push_back(point2);
	
	auto a = MuApriori(itemSet, 1);
	EXPECT_EQ(a.getBeta(), 0.25);
	a.findBest(2);
	auto b = a.getBest();
	EXPECT_NE(b , nullptr);
	EXPECT_EQ(b->support, 1);
	EXPECT_EQ(b->score, a.mu(1,1));
}


TEST(testMuApriori, testFindBest3){
	std::vector<boost::dynamic_bitset<>>* itemSet = new std::vector<boost::dynamic_bitset<>>;

	boost::dynamic_bitset<> point1 = boost::dynamic_bitset<>();
	boost::dynamic_bitset<> point2 = boost::dynamic_bitset<>();
	point1.push_back(0);
	point1.push_back(1);
	itemSet->push_back(point1);

	point2.push_back(0);
	point2.push_back(1);
	itemSet->push_back(point2);

	
	auto a = MuApriori(itemSet, 0);
	EXPECT_EQ(a.getBeta(), 0.25);
	a.findBest(1);
	auto b = a.getBest();
	EXPECT_NE(b, nullptr);
	EXPECT_EQ(b->support, 2);
	EXPECT_EQ(b->score, a.mu(2,1));
	EXPECT_EQ(b->item[0], 0);
	EXPECT_EQ(b->item[1], 1);
}



TEST(testMuApriori, testFindBest4){
	std::vector<boost::dynamic_bitset<>>* itemSet = new std::vector<boost::dynamic_bitset<>>;

	boost::dynamic_bitset<> point1 = boost::dynamic_bitset<>();
	boost::dynamic_bitset<> point2 = boost::dynamic_bitset<>();
	boost::dynamic_bitset<> point3 = boost::dynamic_bitset<>();
	point1.push_back(0);
	point1.push_back(1);
	itemSet->push_back(point1);

	point2.push_back(0);
	point2.push_back(1);
	itemSet->push_back(point2);

	point3.push_back(1);
	point3.push_back(0);
	itemSet->push_back(point3);
	
	auto a = MuApriori(itemSet, 0);
	EXPECT_EQ(a.getBeta(), 0.25);
	a.findBest(1);
	auto b = a.getBest();
	EXPECT_NE(b, nullptr);
	EXPECT_EQ(b->support, 2);
	EXPECT_EQ(b->score, a.mu(2,1));
	EXPECT_EQ(b->item[0], 0);
	EXPECT_EQ(b->item[1], 1);
}




TEST(testMuApriori, testFindBest5){
	std::vector<boost::dynamic_bitset<>>* itemSet = new std::vector<boost::dynamic_bitset<>>;

	boost::dynamic_bitset<> point1 = boost::dynamic_bitset<>();
	boost::dynamic_bitset<> point2 = boost::dynamic_bitset<>();
	boost::dynamic_bitset<> point3 = boost::dynamic_bitset<>();
	point1.push_back(0);
	point1.push_back(1);
	point1.push_back(1);
	point1.push_back(1);
	itemSet->push_back(point1);

	point2.push_back(0);
	point2.push_back(1);
	point2.push_back(0);
	point2.push_back(1);
	itemSet->push_back(point2);

	point3.push_back(0);
	point3.push_back(1);
	point3.push_back(1);
	point3.push_back(1);
	
	itemSet->push_back(point3);
	
	auto a = MuApriori(itemSet, 1);
	EXPECT_EQ(a.getBeta(), 0.25);
	a.findBest(1);
	auto b = a.getBest();
	EXPECT_NE(b, nullptr);
	EXPECT_EQ(b->support, 2);
	EXPECT_EQ(b->score, a.mu(2,3));
	EXPECT_EQ(b->item[0], 0);
	EXPECT_EQ(b->item[1], 1);
	EXPECT_EQ(b->item[2], 1);
	EXPECT_EQ(b->item[3], 1);
}



TEST(testMuApriori, testFindBest6){
	std::vector<boost::dynamic_bitset<>>* itemSet = new std::vector<boost::dynamic_bitset<>>;

	boost::dynamic_bitset<> point1 = boost::dynamic_bitset<>();
	boost::dynamic_bitset<> point2 = boost::dynamic_bitset<>();
	boost::dynamic_bitset<> point3 = boost::dynamic_bitset<>();
	point1.push_back(0);
	point1.push_back(0);
	point1.push_back(0);
	point1.push_back(1);
	itemSet->push_back(point1);

	point2.push_back(0);
	point2.push_back(0);
	point2.push_back(0);
	point2.push_back(1);
	itemSet->push_back(point2);

	point3.push_back(1);
	point3.push_back(1);
	point3.push_back(1);
	point3.push_back(1);
	
	itemSet->push_back(point3);
	
	auto a = MuApriori(itemSet, 0);
	EXPECT_EQ(a.getBeta(), 0.25);
	a.findBest(1);
	auto b = a.getBest();
	EXPECT_NE(b, nullptr);
	EXPECT_EQ(b->support, 1);
	EXPECT_EQ(b->score, a.mu(1,4));
	EXPECT_EQ(b->item[0], 1);
	EXPECT_EQ(b->item[1], 1);
	EXPECT_EQ(b->item[2], 1);
	EXPECT_EQ(b->item[3], 1);
}



TEST(testMuApriori, testFindBest2Clusters){
	std::vector<boost::dynamic_bitset<>>* itemSet = new std::vector<boost::dynamic_bitset<>>;

	boost::dynamic_bitset<> point1 = boost::dynamic_bitset<>();
	point1.push_back(0);
	point1.push_back(1);
	itemSet->push_back(point1);

	{
		boost::dynamic_bitset<> point2 = boost::dynamic_bitset<>();
		point2.push_back(1);
		point2.push_back(0);
		itemSet->push_back(point2);
	}

	{
		boost::dynamic_bitset<> point2 = boost::dynamic_bitset<>();
		point2.push_back(1);
		point2.push_back(0);
		itemSet->push_back(point2);
	}

	
	
	auto a = MuApriori(itemSet, 0);
	EXPECT_EQ(a.getBeta(), 0.25);
	a.findBest(2);
	auto b = a.getBest();
	EXPECT_NE(b, nullptr);
	EXPECT_EQ(b->support, 2);
	EXPECT_EQ(b->score, a.mu(2,1));

}



TEST(testMuApriori, testFindBest2Clusters2){
	std::vector<boost::dynamic_bitset<>>* itemSet = new std::vector<boost::dynamic_bitset<>>;


   
	{
		boost::dynamic_bitset<> point1 = boost::dynamic_bitset<>();
		point1.push_back(0);
		point1.push_back(1);
		point1.push_back(1);
		point1.push_back(1);
		point1.push_back(1);
		point1.push_back(1);
		point1.push_back(1);
		point1.push_back(1);
		itemSet->push_back(point1);
	}

	{
		boost::dynamic_bitset<> point1 = boost::dynamic_bitset<>();
		point1.push_back(1);
		point1.push_back(1);
		point1.push_back(1);
		point1.push_back(1);
		point1.push_back(1);
		point1.push_back(1);
		point1.push_back(1);
		point1.push_back(1);
		itemSet->push_back(point1);
	}

	{
		boost::dynamic_bitset<> point1 = boost::dynamic_bitset<>();
		point1.push_back(1);
		point1.push_back(1);
		point1.push_back(1);
		point1.push_back(1);
		point1.push_back(1);
		point1.push_back(1);
		point1.push_back(1);
		point1.push_back(1);
		itemSet->push_back(point1);
	}

	
	
	auto a = MuApriori(itemSet, 1);
	EXPECT_EQ(a.getBeta(), 0.25);
	a.findBest(2);
	auto b = a.getBest();
	EXPECT_NE(b, nullptr);
	EXPECT_EQ(b->support, 2);
	EXPECT_EQ(((b->item))[0], 1);
	EXPECT_EQ(((b->item))[1], 1);
	EXPECT_EQ(((b->item))[2], 1);
	EXPECT_EQ(((b->item))[3], 1);
	EXPECT_EQ(((b->item))[4], 1);
	EXPECT_EQ(((b->item))[5], 1);
	EXPECT_EQ(((b->item))[6], 1);
	EXPECT_EQ(((b->item))[7], 1);

	EXPECT_EQ(b->score, a.mu(2,8));
	// EXPECT_EQ(b->at(1)->support, 3);
	// EXPECT_EQ(b->at(1)->score, a.mu(3,7));
	// EXPECT_EQ(((b->at(1)->item))[0], 0);
	// EXPECT_EQ(((b->at(1)->item))[1], 1);
	// EXPECT_EQ(((b->at(1)->item))[2], 1);
	// EXPECT_EQ(((b->at(1)->item))[3], 1);
	// EXPECT_EQ(((b->at(1)->item))[4], 1);
	// EXPECT_EQ(((b->at(1)->item))[5], 1);
	// EXPECT_EQ(((b->at(1)->item))[6], 1);
	// EXPECT_EQ(((b->at(1)->item))[7], 1);

}



TEST(testMuApriori, testFindBest3Clusters){
	std::vector<boost::dynamic_bitset<>>* itemSet = new std::vector<boost::dynamic_bitset<>>;


   
	{
		boost::dynamic_bitset<> point1 = boost::dynamic_bitset<>();
		point1.push_back(0);
		point1.push_back(1);
		point1.push_back(1);
		point1.push_back(1);
		point1.push_back(1);
		point1.push_back(1);
		point1.push_back(1);
		point1.push_back(1);
		itemSet->push_back(point1);
	}

	{
		boost::dynamic_bitset<> point1 = boost::dynamic_bitset<>();
		point1.push_back(1);
		point1.push_back(1);
		point1.push_back(1);
		point1.push_back(1);
		point1.push_back(1);
		point1.push_back(1);
		point1.push_back(1);
		point1.push_back(1);
		itemSet->push_back(point1);
	}

	{
		boost::dynamic_bitset<> point1 = boost::dynamic_bitset<>();
		point1.push_back(1);
		point1.push_back(1);
		point1.push_back(1);
		point1.push_back(1);
		point1.push_back(1);
		point1.push_back(1);
		point1.push_back(1);
		point1.push_back(1);
		itemSet->push_back(point1);
	}

	
	
	auto a = MuApriori(itemSet, 1);
	EXPECT_EQ(a.getBeta(), 0.25);
	a.findBest(3);
	auto b = a.getBest();
	// EXPECT_EQ(b->size(),3);
	// EXPECT_EQ(b->at(0)->support, 2);
	// EXPECT_EQ(b->at(0)->score, a.mu(2,8));
	// EXPECT_EQ(b->at(1)->support, 3);
	// EXPECT_EQ(b->at(1)->score, a.mu(3,7));
	// EXPECT_EQ(b->at(2)->support, 2);
	// EXPECT_EQ(b->at(2)->score, a.mu(2,7));
}


TEST(testMuApriori, _SLOW_testFindBest10Clusters){
	std::vector<boost::dynamic_bitset<>>* itemSet = new std::vector<boost::dynamic_bitset<>>;

	for(int i = 0; i < 1000; i++){ // number of points
		for(int j = 0; j < 10; j++){ // number of clusters
			boost::dynamic_bitset<> point = boost::dynamic_bitset<>();
			for(int k = 0; k< 10; k++){ // number of dimensions
				if(j <= k){
					point.push_back(1);
				}else{
					point.push_back(0);					
				}
			}
			itemSet->push_back(point);
		}
	}

	auto a = MuApriori(itemSet, 1);
	EXPECT_EQ(a.getBeta(), 0.25);
	a.findBest(10);
	auto b = a.getBest();
	EXPECT_NE(b, nullptr);
	EXPECT_EQ(b->support, 1000);
	// EXPECT_EQ(b->at(1)->support, 2000);
	EXPECT_EQ(b->item.count(), 10);
	// EXPECT_EQ(b->at(1)->item.count(), 9);
	// EXPECT_EQ(b->at(2)->item.count(), 9);
	// EXPECT_EQ(b->at(3)->item.count(), 9);
	// EXPECT_EQ(b->at(4)->item.count(), 9);
	// EXPECT_EQ(b->at(5)->item.count(), 9);
	// EXPECT_EQ(b->at(6)->item.count(), 9);
	// EXPECT_EQ(b->at(7)->item.count(), 9);
	// EXPECT_EQ(b->at(8)->item.count(), 9);
	// EXPECT_EQ(b->at(9)->item.count(), 9);

	
	// EXPECT_EQ(b->at(2)->support, 1000);
	// EXPECT_EQ(b->at(3)->support, 1000);
	// EXPECT_EQ(b->at(4)->support, 1000);
	// EXPECT_EQ(b->at(5)->support, 1000);
	// EXPECT_EQ(b->at(6)->support, 1000);
	// EXPECT_EQ(b->at(7)->support, 1000);
	// EXPECT_EQ(b->at(8)->support, 1000);
	// EXPECT_EQ(b->at(9)->support, 1000);

	
	// EXPECT_EQ(b->at(0)->score, a.mu(1000,10));
	// EXPECT_EQ(b->at(1)->score, a.mu(2000,9));	
	// EXPECT_EQ(b->at(2)->score, a.mu(1000,9));
	// EXPECT_EQ(b->at(3)->score, a.mu(1000,9));	
	// EXPECT_EQ(b->at(4)->score, a.mu(1000,9));	
	// EXPECT_EQ(b->at(5)->score, a.mu(1000,9));
	// EXPECT_EQ(b->at(6)->score, a.mu(1000,9));	
	// EXPECT_EQ(b->at(7)->score, a.mu(1000,9));
	// EXPECT_EQ(b->at(8)->score, a.mu(1000,9));
	// EXPECT_EQ(b->at(9)->score, a.mu(1000,9));
}


TEST(testMuApriori, _SLOWtestFindBest20Clusters){
	std::vector<boost::dynamic_bitset<>>* itemSet = new std::vector<boost::dynamic_bitset<>>;

	for(int i = 0; i < 1000; i++){ // number of points
		for(int j = 0; j < 80; j+=2){ // number of clusters
			boost::dynamic_bitset<> point = boost::dynamic_bitset<>();
			unsigned int count = 0;
			for(int k = 0; k< 80; k++){ // number of dimensions
				if(j <= k && count<10){
					count++;
					point.push_back(1);
				}else{
					point.push_back(0);					
				}
			}
			itemSet->push_back(point);
		}
	}

	auto a = MuApriori(itemSet, 0.1*itemSet->size());
	EXPECT_EQ(a.getBeta(), 0.25);
	a.findBest(20);
	auto b = a.getBest();
	EXPECT_NE(b, nullptr);
	

	
	EXPECT_EQ(b->item.count(), 4);
	// EXPECT_EQ(b->at(1)->item.count(), 4);
	// EXPECT_EQ(b->at(2)->item.count(), 4);
	// EXPECT_EQ(b->at(3)->item.count(), 4);
	// EXPECT_EQ(b->at(4)->item.count(), 4);
	// EXPECT_EQ(b->at(5)->item.count(), 4);
	// EXPECT_EQ(b->at(6)->item.count(), 4);
	// EXPECT_EQ(b->at(7)->item.count(), 4);
	// EXPECT_EQ(b->at(8)->item.count(), 4);
	// EXPECT_EQ(b->at(9)->item.count(), 4);

	EXPECT_EQ(b->support, 4000);
	// EXPECT_EQ(b->at(1)->support, 4000);	
	// EXPECT_EQ(b->at(2)->support, 4000);
	// EXPECT_EQ(b->at(3)->support, 4000);
	// EXPECT_EQ(b->at(4)->support, 4000);
	// EXPECT_EQ(b->at(5)->support, 4000);
	// EXPECT_EQ(b->at(6)->support, 4000);
	// EXPECT_EQ(b->at(7)->support, 4000);
	// EXPECT_EQ(b->at(8)->support, 4000);
	// EXPECT_EQ(b->at(9)->support, 4000);

	
	EXPECT_EQ(b->score, a.mu(4000,4));
	// EXPECT_EQ(b->at(1)->score, a.mu(4000,4));	
	// EXPECT_EQ(b->at(2)->score, a.mu(4000,4));
	// EXPECT_EQ(b->at(3)->score, a.mu(4000,4));	
	// EXPECT_EQ(b->at(4)->score, a.mu(4000,4));	
	// EXPECT_EQ(b->at(5)->score, a.mu(4000,4));
	// EXPECT_EQ(b->at(6)->score, a.mu(4000,4));	
	// EXPECT_EQ(b->at(7)->score, a.mu(4000,4));
	// EXPECT_EQ(b->at(8)->score, a.mu(4000,4));
	// EXPECT_EQ(b->at(9)->score, a.mu(4000,4));

}


TEST(testMuApriori, _SLOWtestFindBest80Clusters_smaller){
	std::vector<boost::dynamic_bitset<>>* itemSet = new std::vector<boost::dynamic_bitset<>>;

	for(int i = 0; i < 10; i++){ // number of points
		for(int j = 0; j < 20; j+=2){ // number of clusters
			boost::dynamic_bitset<> point = boost::dynamic_bitset<>();
			unsigned int count = 0;
			for(int k = 0; k< 20; k++){ // number of dimensions
				if(j <= k && count<10){
					count++;
					point.push_back(1);
				}else{
					point.push_back(0);					
				}
			}
			itemSet->push_back(point);
		}
	}

	auto a = MuApriori(itemSet, 0.1*itemSet->size());
	EXPECT_EQ(a.getBeta(), 0.25);
	a.findBest(20);
	auto b = a.getBest();
	EXPECT_NE(b, nullptr);
	

	EXPECT_EQ(b->item.count(), 10);
	// EXPECT_EQ(b->at(1)->item.count(), 4);
	// EXPECT_EQ(b->at(2)->item.count(), 4);
	// EXPECT_EQ(b->at(3)->item.count(), 4);
	// EXPECT_EQ(b->at(4)->item.count(), 4);
	// EXPECT_EQ(b->at(5)->item.count(), 4);
	// EXPECT_EQ(b->at(6)->item.count(), 4);
	// EXPECT_EQ(b->at(7)->item.count(), 4);
	// EXPECT_EQ(b->at(8)->item.count(), 4);
	// EXPECT_EQ(b->at(9)->item.count(), 4);

	EXPECT_EQ(b->support, 10);
	// EXPECT_EQ(b->at(1)->support, 4000);	
	// EXPECT_EQ(b->at(2)->support, 4000);
	// EXPECT_EQ(b->at(3)->support, 4000);
	// EXPECT_EQ(b->at(4)->support, 4000);
	// EXPECT_EQ(b->at(5)->support, 4000);
	// EXPECT_EQ(b->at(6)->support, 4000);
	// EXPECT_EQ(b->at(7)->support, 4000);
	// EXPECT_EQ(b->at(8)->support, 4000);
	// EXPECT_EQ(b->at(9)->support, 4000);

	
	EXPECT_EQ(b->score, a.mu(10,10));
	// EXPECT_EQ(b->at(1)->score, a.mu(4000,4));	
	// EXPECT_EQ(b->at(2)->score, a.mu(4000,4));
	// EXPECT_EQ(b->at(3)->score, a.mu(4000,4));	
	// EXPECT_EQ(b->at(4)->score, a.mu(4000,4));	
	// EXPECT_EQ(b->at(5)->score, a.mu(4000,4));
	// EXPECT_EQ(b->at(6)->score, a.mu(4000,4));	
	// EXPECT_EQ(b->at(7)->score, a.mu(4000,4));
	// EXPECT_EQ(b->at(8)->score, a.mu(4000,4));
	// EXPECT_EQ(b->at(9)->score, a.mu(4000,4));

}


TEST(testMuApriori, _SUPER_SLOW_testFindBest20ClustersRandom){
	std::vector<boost::dynamic_bitset<>>* itemSet = new std::vector<boost::dynamic_bitset<>>;
    std::random_device dev;
    auto gen = std::mt19937{dev()};
	gen.seed(1);
    static auto dist = std::uniform_real_distribution<float>(0,1);

	float p = 0.5;
	
	for(int i = 0; i < 10000; i++){ // number of points
		boost::dynamic_bitset<> point = boost::dynamic_bitset<>();
		for(int j = 0; j < 50; j+=2){ // number of dims
			point.push_back((dist(gen) < p));
		}
		itemSet->push_back(point);
	}

	// would expect best clusters to be of dim 3-4 because 0.5^3 >0.1 and 0.5^4 < 0.1
	// The first couple of iterations would be pretty slow, but will not need more than 4. 

	auto a = MuApriori(itemSet, 0.1*itemSet->size());
	EXPECT_EQ(a.getBeta(), 0.25);
	a.findBest(20);
	auto b = a.getBest();
	EXPECT_NE(b, nullptr);
	

	EXPECT_EQ(b->item.count(), 3);
	// EXPECT_EQ(b->at(1)->item.count(), 3);
	// EXPECT_EQ(b->at(2)->item.count(), 3);
	// EXPECT_EQ(b->at(3)->item.count(), 3);
	// EXPECT_EQ(b->at(4)->item.count(), 3);
	// EXPECT_EQ(b->at(5)->item.count(), 3);
	// EXPECT_EQ(b->at(6)->item.count(), 3);
	// EXPECT_EQ(b->at(7)->item.count(), 3);
	// EXPECT_EQ(b->at(8)->item.count(), 3);
	// EXPECT_EQ(b->at(9)->item.count(), 3);

	EXPECT_EQ(b->support, 1350);
	// EXPECT_EQ(b->at(1)->support, 1344);	
	// EXPECT_EQ(b->at(2)->support, 1339);
	// EXPECT_EQ(b->at(3)->support, 1337);
	// EXPECT_EQ(b->at(4)->support, 1335);
	// EXPECT_EQ(b->at(5)->support, 1333);
	// EXPECT_EQ(b->at(6)->support, 1333);
	// EXPECT_EQ(b->at(7)->support, 1332);
	// EXPECT_EQ(b->at(8)->support, 1330);
	// EXPECT_EQ(b->at(9)->support, 1327);

	
	EXPECT_EQ(b->score, a.mu(1350,3));
	// EXPECT_EQ(b->at(1)->score, a.mu(1344,3));	
	// EXPECT_EQ(b->at(2)->score, a.mu(1339,3));
	// EXPECT_EQ(b->at(3)->score, a.mu(1337,3));	
	// EXPECT_EQ(b->at(4)->score, a.mu(1335,3));	
	// EXPECT_EQ(b->at(5)->score, a.mu(1333,3));
	// EXPECT_EQ(b->at(6)->score, a.mu(1333,3));	
	// EXPECT_EQ(b->at(7)->score, a.mu(1332,3));
	// EXPECT_EQ(b->at(8)->score, a.mu(1330,3));
	// EXPECT_EQ(b->at(9)->score, a.mu(1327,3));
}

