#include <gtest/gtest.h>
#include "../src/MuApriori/MuApriori.h"
#include <vector>
#include <boost/dynamic_bitset.hpp>


TEST(testMuApriori, testConstructor){
	std::vector<boost::dynamic_bitset<>*>* itemSet = new std::vector<boost::dynamic_bitset<>*>;
	auto a = MuApriori(itemSet, 1);
	EXPECT_EQ(a.getBeta(), 0.25);
	SUCCEED();
}



TEST(testMuApriori, testConstructor2){
	std::vector<boost::dynamic_bitset<>*>* itemSet = new std::vector<boost::dynamic_bitset<>*>;
	boost::dynamic_bitset<>* point = new boost::dynamic_bitset<>;
	point->push_back(0);
	itemSet->push_back(point);
	auto a = MuApriori(itemSet, 1);
	EXPECT_EQ(a.getBeta(), 0.25);
	SUCCEED();
}



TEST(testMuApriori, testCreateInitialCandidates){
	std::vector<boost::dynamic_bitset<>*>* itemSet = new std::vector<boost::dynamic_bitset<>*>;

	boost::dynamic_bitset<>* point = new boost::dynamic_bitset<>;
	point->push_back(0);
	itemSet->push_back(point);
	auto a = MuApriori(itemSet, 1);
	EXPECT_EQ(a.getBeta(), 0.25);
	auto b = a.createInitialCandidates();
	EXPECT_EQ(b->size(), point->size());
	EXPECT_EQ(b->size(), 1);
	EXPECT_EQ(b->at(0).support, 1);
	EXPECT_EQ(b->at(0).score, a.mu(1,1));
	
	SUCCEED();
}



TEST(testMuApriori, testFindBest1){
	std::vector<boost::dynamic_bitset<>*>* itemSet = new std::vector<boost::dynamic_bitset<>*>;
	boost::dynamic_bitset<>* point1 = new boost::dynamic_bitset<>;
	boost::dynamic_bitset<>* point2 = new boost::dynamic_bitset<>;
	point1->push_back(0);
	point1->push_back(1);
	itemSet->push_back(point1);
	point2->push_back(1);
	point2->push_back(0);
	itemSet->push_back(point2);
	auto a = MuApriori(itemSet, 0);
	EXPECT_EQ(a.getBeta(), 0.25);
	auto b = a.findBest(1);
	EXPECT_EQ(b->size(),1);
	EXPECT_EQ(b->at(0).support, 1);
	EXPECT_EQ(b->at(0).score, a.mu(1,1));	
}

TEST(testMuApriori, testFindBest2){
	std::vector<boost::dynamic_bitset<>*>* itemSet = new std::vector<boost::dynamic_bitset<>*>;

	boost::dynamic_bitset<>* point1 = new boost::dynamic_bitset<>;
	boost::dynamic_bitset<>* point2 = new boost::dynamic_bitset<>;
	point1->push_back(0);
	point1->push_back(1);
	itemSet->push_back(point1);

	point2->push_back(1);
	point2->push_back(0);
	itemSet->push_back(point2);
	
	auto a = MuApriori(itemSet, 0);
	EXPECT_EQ(a.getBeta(), 0.25);
	auto b = a.findBest(2);
	EXPECT_EQ(b->size(),2);
	EXPECT_EQ(b->at(0).support, 1);
	EXPECT_EQ(b->at(0).score, a.mu(1,1));
	EXPECT_EQ(b->at(1).support, 1);
	EXPECT_EQ(b->at(1).score, a.mu(1,1));
}


TEST(testMuApriori, testFindBest3){
	std::vector<boost::dynamic_bitset<>*>* itemSet = new std::vector<boost::dynamic_bitset<>*>;

	boost::dynamic_bitset<>* point1 = new boost::dynamic_bitset<>;
	boost::dynamic_bitset<>* point2 = new boost::dynamic_bitset<>;
	point1->push_back(0);
	point1->push_back(1);
	itemSet->push_back(point1);

	point2->push_back(0);
	point2->push_back(1);
	itemSet->push_back(point2);
	
	auto a = MuApriori(itemSet, 0);
	EXPECT_EQ(a.getBeta(), 0.25);
	auto b = a.findBest(1);
	EXPECT_EQ(b->size(),1);
	EXPECT_EQ(b->at(0).support, 2);
	EXPECT_EQ(b->at(0).score, a.mu(2,1));
	EXPECT_EQ(b->at(0).item[0], 0);
	EXPECT_EQ(b->at(0).item[1], 1);
}



TEST(testMuApriori, testFindBest4){
	std::vector<boost::dynamic_bitset<>*>* itemSet = new std::vector<boost::dynamic_bitset<>*>;

	boost::dynamic_bitset<>* point1 = new boost::dynamic_bitset<>;
	boost::dynamic_bitset<>* point2 = new boost::dynamic_bitset<>;
	boost::dynamic_bitset<>* point3 = new boost::dynamic_bitset<>;
	point1->push_back(0);
	point1->push_back(1);
	itemSet->push_back(point1);

	point2->push_back(0);
	point2->push_back(1);
	itemSet->push_back(point2);

	point3->push_back(1);
	point3->push_back(0);
	itemSet->push_back(point3);
	
	auto a = MuApriori(itemSet, 0);
	EXPECT_EQ(a.getBeta(), 0.25);
	auto b = a.findBest(1);
	EXPECT_EQ(b->size(),1);
	EXPECT_EQ(b->at(0).support, 2);
	EXPECT_EQ(b->at(0).score, a.mu(2,1));
	EXPECT_EQ(b->at(0).item[0], 0);
	EXPECT_EQ(b->at(0).item[1], 1);
}
