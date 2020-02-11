#include <gtest/gtest.h>
#include "../src/testingTools/DataGeneratorBuilder.h"
#include "../src/dataReader/Cluster.h"
#include "../src/dataReader/DataReader.h"
#include "../src/testingTools/MetaDataFileReader.h"
#include <iostream>


TEST(dataGenerationTests, testMakingAfileConstant){
	DataGeneratorBuilder dgb;
	Cluster small;
	small.setAmmount(10);
	small.addDimension(constant,{0,0},{0,0,1},1);
	small.addDimension(constant,{0,0},{0,0,1},1);
	dgb.addCluster(small);
	dgb.build();
	DataReader dr;
	SUCCEED();
	EXPECT_EQ(dr.getDimensions(), 2);
	EXPECT_EQ(dr.getSize(), 10);
}

TEST(dataGenerationTests, testMakingAfileUniform){
	DataGeneratorBuilder dgb;
	Cluster small;
	small.setAmmount(10000);
	small.addDimension();
	small.addDimension();
	dgb.addCluster(small);
	dgb.build();
	DataReader dr;
	SUCCEED();
	EXPECT_EQ(dr.getDimensions(), 2);
	EXPECT_EQ(dr.getSize(), 10000);
	bool happen = false;
	while(dr.isThereANextPoint()){
		if(dr.nextPoint()->at(0) > 50){
			happen = true;
		}
	}
	EXPECT_TRUE(happen);

}

TEST(dataGenerationTests, testOutLiers){
	DataGeneratorBuilder dgb;
	Cluster small;
	small.setAmmount(10);
	small.addDimension(constant,{0,0},{0,0},100,0);
	small.setOutLierPercentage(50);
	Cluster big;
	big.setAmmount(20);
	big.setOutLierPercentage(25);
	big.addDimension(constant,{0,0},{0,0},100,0);
	dgb.addCluster(big);
	dgb.addCluster(small);
	dgb.build();

	DataReader dr;
	SUCCEED();
	EXPECT_EQ(dr.getDimensions(), 1);
	EXPECT_EQ(dr.getSize(), 30);
	MetaDataFileReader mdfr;
	EXPECT_EQ(mdfr.getClusters().size(), 3);
}

TEST(dataGenerationTests, testUBuilder){
	DataGeneratorBuilder dgb;

	if(!dgb.buildUClusters(1000,5,15,5,3)){
		std::cout << "not good" << std::endl;
	}

	SUCCEED();

}


TEST(dataGenerationTests, testUBuilderVariance){
	DataGeneratorBuilder dgb;
	Cluster small;
	small.setAmmount(100000);
	small.addDimension(normalDistribution,{0,0},{50,15},21);
	small.setOutLierPercentage(0);
	dgb.addCluster(small);

	Cluster big;
	big.setAmmount(100000);
	big.addDimension(normalDistribution,{0,0},{50,15},21);
	big.setOutLierPercentage(0);
	dgb.addCluster(big);

	dgb.build();
	SUCCEED();

}

TEST(dataGenerationTests, testUBuilderVarianceMultipleDimensions){
	DataGeneratorBuilder dgb;
	Cluster small;
	small.setAmmount(30);
	small.addDimension(normalDistribution,{0,0},{50,15},21);
	small.addDimension();
	small.addDimension();
	small.addDimension();
	small.setOutLierPercentage(0);
	dgb.addCluster(small);

	Cluster big;
	big.setAmmount(30);
	big.addDimension(normalDistribution,{0,0},{50,15},21);
	big.addDimension(normalDistribution,{0,0},{50,15},21);
	big.addDimension(normalDistribution,{0,0},{50,15},21);
	big.setOutLierPercentage(0);
	dgb.addCluster(big);

	Cluster medium;
	medium.setAmmount(30);
	medium.addDimension(normalDistribution,{0,0},{50,15},21);
	medium.addDimension(normalDistribution,{0,0},{50,15},21);
	medium.addDimension(normalDistribution,{0,0},{50,15},21);
	medium.addDimension(normalDistribution,{0,0},{50,15},21);
	dgb.build();
	SUCCEED();

}
