#include <gtest/gtest.h>
#include <iostream>
#include "../src/DOC_GPU/ArgMax.h"
#include "../src/DOC/DOC.h"
#include <random>
#include <cstdlib>      // std::rand, std::srand
#include <algorithm>    // std::random_shuffle


TEST(testArgMaxGPU, testSimple){
	std::vector<float>* scores = new std::vector<float>;
	int k = 0;
	for(int i = 0; i < 8; i++){
		scores->push_back(i+1);
		k+=i+1;
	}
	auto c = argMax(scores);
	EXPECT_EQ(c, 7);
}


TEST(testArgMaxGPU, testSimple2){
	std::vector<float>* scores = new std::vector<float>;
	for(int i = 0; i < 8; i++){
		scores->push_back(8-i);
	}
	auto c = argMax(scores);
	EXPECT_EQ(c, 0);
}


TEST(testArgMaxGPU, testSimple3){
	std::vector<float>* scores = new std::vector<float>;
	for(int i = 0; i < 16; i++){
		scores->push_back(16-i);
	}
	auto c = argMax(scores);
	EXPECT_EQ(c, 0);
}

TEST(testArgMaxGPU, testSimple4){
	std::vector<float>* scores = new std::vector<float>;
	for(int i = 0; i < 16; i++){
		scores->push_back(i);
	}
	auto c = argMax(scores);
	EXPECT_EQ(c, 15);
}



TEST(testArgMaxGPU, testSimple5){
	std::vector<float>* scores = new std::vector<float>{0,9,5,3,2,9,11,4,4,4,4,4,5,6};
	auto c = argMax(scores);
	EXPECT_EQ(c, 6);
}



TEST(testArgMaxGPU, testLarge){
	std::vector<float>* scores = new std::vector<float>;
	int n = 1024*2;
	for(int i = 0; i < n; i++){
		scores->push_back(i);
	}
	auto c = argMax(scores);
	EXPECT_EQ(c, n-1);
}



TEST(testArgMaxGPU, testLarge2){
	std::vector<float>* scores = new std::vector<float>;
	int n = 4096;
	for(int i = 0; i < n; i++){
		scores->push_back(i);
	}
	auto c = argMax(scores);
	EXPECT_EQ(c, n-1);
}


TEST(testArgMaxGPU, testLarge3){
	std::vector<float>* scores = new std::vector<float>;
	int n = 4096*2;
	for(int i = 0; i < n; i++){
		scores->push_back(i);
	}
	auto c = argMax(scores);
	EXPECT_EQ(c, n-1);
}


TEST(testArgMaxGPU, testLarge4){
	std::vector<float>* scores = new std::vector<float>;
	int n = 53687;
	for(int i = 0; i < n; i++){
		scores->push_back(i);
	}
	auto c = argMax(scores);
	EXPECT_EQ(c, n-1);
}


TEST(testArgMaxGPU, testLarge5){
	std::vector<float>* scores = new std::vector<float>;
	int n = 53687;
	for(int i = 0; i < n; i++){
		scores->push_back(i);
	}
	
	std::srand ( 0 );
	std::random_shuffle ( scores->begin(), scores->end(), [](int i){ return std::rand()%i;});

	
	int max_idx = (std::distance(scores->begin(), max_element(scores->begin(), scores->end())));
	
	auto c = argMax(scores);
	EXPECT_EQ(c, max_idx);
}

TEST(testArgMaxGPU, testLarge6){
	std::vector<float>* scores = new std::vector<float>;
	int n = 536870;
	for(int i = 0; i < n; i++){
		scores->push_back(i);
	}
	
	std::srand ( 0 );
	std::random_shuffle ( scores->begin(), scores->end(), [](int i){ return std::rand()%i;});

	
	int max_idx = (std::distance(scores->begin(), max_element(scores->begin(), scores->end())));
	
	auto c = argMax(scores);
	EXPECT_EQ(c, max_idx);
}



TEST(testArgMaxGPU, testWidthBound){
	std::vector<float>* scores = new std::vector<float>;
	int n = 4096*2;
	for(int i = 0; i < n; i++){
		scores->push_back(i);
	}
	auto c = argMaxBound(scores, 100);
	EXPECT_EQ(c, 100);
}



TEST(testArgMaxGPU, testWidthBound2){
	std::vector<float>* scores = new std::vector<float>;
	int n = 4096*2;
	for(int i = 0; i < n; i++){
		scores->push_back(i);
	}
	auto c = argMaxBound(scores, 99999999);
	EXPECT_EQ(c, n-1);
}
