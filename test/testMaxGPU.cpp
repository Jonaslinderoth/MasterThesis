#include <gtest/gtest.h>
#include <iostream>
#include "../src/DOC_GPU/HyperCube.h"
#include "../src/DOC/DOC.h"
#include <random>

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
