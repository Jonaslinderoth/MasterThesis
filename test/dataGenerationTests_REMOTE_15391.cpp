#include <gtest/gtest.h>
#include "../src/testingTools/DataGeneratorBuilder.h"
#include "../src/dataReader/Cluster.h"
#include "../src/dataReader/DataReader.h"
#include "../src/testingTools/MetaDataFileReader.h"
#include <iostream>

class dataGenerationTests : public ::testing::Test {
public:
  // Per-test-suite set-up.
  // Called before the first test in this test suite.
  // Can be omitted if not needed.
  static void SetUpTestCase() {
	  if(system("mkdir testData")){
		  
	  };
  }

  // Per-test-suite tear-down.
  // Called after the last test in this test suite.
  // Can be omitted if not needed.
  static void TearDownTestCase() {
	  if(system("rm -r testData")){
			  
		  };
  }


	virtual void SetUp() {
	}



};
auto dir = "testData";

TEST_F(dataGenerationTests, testMakingAfileConstant){
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

TEST_F(dataGenerationTests, testMakingAfileUniform){
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

TEST_F(dataGenerationTests, testOutLiers){
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

TEST_F(dataGenerationTests, testUBuilder){
	DataGeneratorBuilder dgb;

	bool res = dgb.buildUClusters("test1",1000,5,15,5,5,0);
	EXPECT_TRUE(res);
	SUCCEED();

}


TEST_F(dataGenerationTests, testUBuilderVariance){
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

TEST_F(dataGenerationTests, testUBuilderVarianceMultipleDimensions){
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

	dgb.build();
	SUCCEED();

}

TEST_F(dataGenerationTests, testDimensionWithMutipleClusters){
	DataGeneratorBuilder dgb;
	Cluster small;
	small.setAmmount(5);
	small.addDimension(normalDistribution,{0,0},{50,15},21);
	small.setOutLierPercentage(0);
	dgb.addCluster(small);


	Cluster big;
	big.setAmmount(5);
	big.addDimension(normalDistribution,{0,0},{50,15},21);
	big.addDimension(normalDistribution,{0,0},{50,15},21);
	big.addDimension(normalDistribution,{0,0},{50,15},21);
	big.setOutLierPercentage(0);
	dgb.addCluster(big);
	dgb.build();

	DataReader* dr = new DataReader();
	unsigned int count = 0;
	while(dr->isThereANextPoint()){
		dr->nextPoint();
		count++;
	}


	SUCCEED();
	EXPECT_TRUE(count == 10);
}


TEST_F(dataGenerationTests, testFixFileName){
	DataGeneratorBuilder dgb;
	dgb.setFileName("testData/test2");
	Cluster small;
	small.setAmmount(5);
	small.addDimension(normalDistribution,{0,0},{50,15},21);
	small.setOutLierPercentage(0);
	dgb.addCluster(small);


	Cluster big;
	big.setAmmount(5);
	big.addDimension(normalDistribution,{0,0},{50,15},21);
	big.addDimension(normalDistribution,{0,0},{50,15},21);
	big.addDimension(normalDistribution,{0,0},{50,15},21);
	big.setOutLierPercentage(0);
	dgb.addCluster(big);
	dgb.build();

	DataReader* dr = new DataReader("testData/test2");
	unsigned int count = 0;
	while(dr->isThereANextPoint()){
		dr->nextPoint();
		count++;
	}

	SUCCEED();
	EXPECT_TRUE(count == 10);
}


TEST_F(dataGenerationTests, testMultipleNormalDistibuitionsInOneDimension){
	DataGeneratorBuilder dgb;
	Cluster small;
	small.setAmmount(1000);
	small.addDimension(normalDistribution,{0,100},{50,1,3},21);
	small.addDimension();
	dgb.addCluster(small);

	Cluster big;
	big.setAmmount(100000);
	big.addDimension(normalDistribution,{0,100},{50,1,3},21);
	big.addDimension(normalDistribution,{0,100},{50,1,3},21);
	dgb.addCluster(big);

	dgb.build();
	DataReader* dr = new DataReader();
	unsigned int count = 0;
	while(dr->isThereANextPoint()){
		std::vector<float>* point = dr->nextPoint();

		//std::cout << point->at(1) << std::endl;
	}
	SUCCEED();
}


TEST_F(dataGenerationTests, testMGqBuilder){
	DataGeneratorBuilder dgb;
	bool res = dgb.buildMGqClusters("test1",2,500,1,1,1,0);

	DataReader* dr = new DataReader("test1");
	unsigned int count = 0;
	while(dr->isThereANextPoint()){
		std::vector<float>* point = dr->nextPoint();

		//std::cout << point->at(0) << std::endl;
	}

	SUCCEED();
}

TEST_F(dataGenerationTests, testMGqBuilder2){
	DataGeneratorBuilder dgb;
	dgb.buildMGqClusters("test1",2,500,2,1,1,0,0);

	DataReader* dr = new DataReader("test1");
	unsigned int count = 0;
	while(dr->isThereANextPoint()){
		std::vector<float>* point = dr->nextPoint();

		//std::cout << point->at(0) << std::endl;
	}

	SUCCEED();
}


TEST_F(dataGenerationTests, testBuilder3){
	DataGeneratorBuilder dgb;
	dgb.buildMGqClusters("test1",2,5,5,10,5,0,1);

	DataReader* dr = new DataReader("test1");
	unsigned int count = 0;
	while(dr->isThereANextPoint()){
		std::vector<float>* point = dr->nextPoint();
		/*
		for(std::vector<float>::iterator iter = point->begin() ; iter != point->end() ; ++iter){
			std::cout << std::to_string(*iter) ;

			std::cout << std::endl;
		}*/
		//std::cout << point->at(0) << std::endl;
	}

	SUCCEED();
}

TEST_F(dataGenerationTests, testOverWrite){
	DataGeneratorBuilder dgb;
	Cluster small;
	small.setAmmount(100);
	small.addDimension();
	dgb.addCluster(small);
	dgb.build();

	DataGeneratorBuilder dgb2;
	small.setAmmount(50);
	dgb2.addCluster(small);
	dgb2.build(false);

	DataReader* dr = new DataReader();
	unsigned int count = 0;
	while(dr->isThereANextPoint()){
		dr->nextPoint();
		count++;
	}
	SUCCEED();
	EXPECT_TRUE(count == 100);
}
