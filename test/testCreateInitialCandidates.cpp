#include <gtest/gtest.h>
#include "../src/MineClus/MineClusKernels.h"
#include "testingTools.h"


TEST(testCreateInitialCandidates, test2){
	auto res = createInitialCandidatesTester(2);

	EXPECT_EQ(res.size(), 2);
	EXPECT_EQ(readBit(res.at(0), 0), 1);
	EXPECT_EQ(readBit(res.at(0), 1), 0);
	
	EXPECT_EQ(readBit(res.at(1), 0), 0);
	EXPECT_EQ(readBit(res.at(1), 1), 1);	
	
}

TEST(testCreateInitialCandidates, test2_2){
	auto res = createInitialCandidatesTester(2);

	EXPECT_EQ(res.size(), 2);
	EXPECT_EQ(readBit(res.at(0), 0), 1);
	EXPECT_EQ(readBit(res.at(0), 1), 0);
	EXPECT_EQ(readBit(res.at(0), 2), 0);
	
	EXPECT_EQ(readBit(res.at(1), 0), 0);
	EXPECT_EQ(readBit(res.at(1), 1), 1);
	EXPECT_EQ(readBit(res.at(1), 2), 0);	
	
}


TEST(testCreateInitialCandidates, test3){
	auto res = createInitialCandidatesTester(3);

	EXPECT_EQ(res.size(), 3);
	EXPECT_EQ(readBit(res.at(0), 0), 1);
	EXPECT_EQ(readBit(res.at(0), 1), 0);
	EXPECT_EQ(readBit(res.at(0), 2), 0);
	
	EXPECT_EQ(readBit(res.at(1), 0), 0);
	EXPECT_EQ(readBit(res.at(1), 1), 1);
	EXPECT_EQ(readBit(res.at(1), 2), 0);	

	EXPECT_EQ(readBit(res.at(2), 0), 0);
	EXPECT_EQ(readBit(res.at(2), 1), 0);
	EXPECT_EQ(readBit(res.at(2), 2), 1);	
}

TEST(testCreateInitialCandidates, test33){
	auto res = createInitialCandidatesTester(33);

	for(int i = 0; i < 33; i++){
		for(int j = 0; j < 2; j++){
			for(int k = 0; k < 32; k++){
				if(j == 1 && k >= 1){break;};
				//std::cout << readBit(res.at(j*33+i), k);
				EXPECT_EQ(readBit(res.at(j*33+i), k), i == k+32*j);
			}
		}
	}
}


TEST(testCreateInitialCandidates, test128){
	auto res = createInitialCandidatesTester(128);



	for(int i = 0; i < 128; i++){
		for(int j = 0; j < 4; j++){
			for(int k = 0; k < 32; k++){
				//std::cout << readBit(res.at(j*128+i), k);
				EXPECT_EQ(readBit(res.at(j*128+i), k), i == k+32*j);
			}
		}
	}
	

	EXPECT_EQ(res.size(), 512);

}
