#include "../src/DOC_GPU/DOCGPUnified.h"
#include "../src/Fast_DOCGPU/Fast_DOCGPUnified.h"
#include "../src/DOC_GPU/DOCGPUnified.h"
#include "../src/MineClusGPU/MineClusGPUnified.h"
#include <gtest/gtest.h>
#include "../src/testingTools/DataGeneratorBuilder.h"
#include "../src/dataReader/Cluster.h"
#include "../src/dataReader/DataReader.h"


class testLargeDatasets : public ::testing::Test {
public:
	// Per-test-suite set-up.
	// Called before the first test in this test suite.
	// Can be omitted if not needed.
	static void SetUpTestCase() {
		if(system("mkdir -p test/testData")){

		};

		{
			DataGeneratorBuilder dgb;
			dgb.setSeed(1);
			Cluster small;
			small.setAmmount(400000);
			for(unsigned int j = 0; j < 1000; j++){
				if(j%100 == 0){
					small.addDimension(normalDistribution, {-10000,10000}, {50,2});						
				}else{
					small.addDimension(uniformDistribution, {-10000,10000});						
				}
			}
	
			dgb.addCluster(small);
			Cluster small2;
			small2.setAmmount(100000);
			for(unsigned int j = 0; j < 1000; j++){
				if(j%110 == 0){
					small2.addDimension(normalDistribution, {-10000,10000}, {50,2});						
				}else{
					small2.addDimension(uniformDistribution, {-10000,10000});						
				}
			}
			dgb.addCluster(small2);

			dgb.setFileName("test/testData/test2GB");
			dgb.build(false);
			std::cout << "done" << std::endl;
		}

		{
			DataGeneratorBuilder dgb;
			dgb.setSeed(1);
			Cluster small;
			small.setAmmount(800000);
			for(unsigned int j = 0; j < 1000; j++){
				if(j%100 == 0){
					small.addDimension(normalDistribution, {-10000,10000}, {50,2});						
				}else{
					small.addDimension(uniformDistribution, {-10000,10000});						
				}
			}
	
			dgb.addCluster(small);
			Cluster small2;
			small2.setAmmount(200000);
			for(unsigned int j = 0; j < 1000; j++){
				if(j%110 == 0){
					small2.addDimension(normalDistribution, {-10000,10000}, {50,2});						
				}else{
					small2.addDimension(uniformDistribution, {-10000,10000});						
				}
			}
			dgb.addCluster(small2);

			dgb.setFileName("test/testData/test4GB");
			dgb.build(false);
			std::cout << "done" << std::endl;
		}



		{
			DataGeneratorBuilder dgb;
			dgb.setSeed(1);
			{
				Cluster small;
				small.setAmmount(800000);
				for(unsigned int j = 0; j < 1000; j++){
					if(j%100 == 0){
						small.addDimension(normalDistribution, {-10000,10000}, {50,2});						
					}else{
						small.addDimension(uniformDistribution, {-10000,10000});						
					}
				}
				dgb.addCluster(small);
			}
			{
				Cluster small;
				small.setAmmount(800000);
				for(unsigned int j = 0; j < 1000; j++){
					if(j%110 == 0){
						small.addDimension(normalDistribution, {-10000,10000}, {50,2});						
					}else{
						small.addDimension(uniformDistribution, {-10000,10000});						
					}
				}
				dgb.addCluster(small);
			}
			{
				Cluster small2;
				small2.setAmmount(400000);
				for(unsigned int j = 0; j < 1000; j++){
					if(j%120 == 0){
						small2.addDimension(normalDistribution, {-10000,10000}, {50,2});						
					}else{
						small2.addDimension(uniformDistribution, {-10000,10000});						
					}
				}
				dgb.addCluster(small2);
			}

			dgb.setFileName("test/testData/test8GB");
			dgb.build(false);
			std::cout << "done" << std::endl;
		}
	}

	// Per-test-suite tear-down.
	// Called after the last test in this test suite.
	// Can be omitted if not needed.
	static void TearDownTestCase() {

	}


	virtual void SetUp() {
	}



};

TEST_F(testLargeDatasets, _SUPER_SLOW_testDOC2GB){
	std::cout << "hello " << std::endl;
	DataReader* dr = new DataReader("test/testData/test2GB");
	std::cout << "hello " << std::endl;
	DOCGPUnified* d = new DOCGPUnified(dr);
	d->setNumberOfSamples(1024); // upperbound to 1024
	auto res = d->findKClusters(2);

	EXPECT_EQ(res.at(0).first->size(), 400000);
	EXPECT_EQ(res.at(1).first->size(), 100000);

	EXPECT_EQ(res.at(0).second->size(), 10);
	EXPECT_EQ(res.at(1).second->size(), 9);	
	delete d;
}


TEST_F(testLargeDatasets, _SUPER_SLOW_testFastDOC2GB){
	DataReader* dr = new DataReader("test/testData/test2GB");
	Fast_DOCGPUnified* d = new Fast_DOCGPUnified(dr);
	auto res = d->findKClusters(2);

	EXPECT_EQ(res.at(0).first->size(), 400000);
	EXPECT_EQ(res.at(1).first->size(), 100000);

	EXPECT_EQ(res.at(0).second->size(), 1000);
	unsigned int c = 0; 
	for(unsigned int i = 0; i < 1000; i++){
		c += res.at(0).second->at(i);
	}
	EXPECT_EQ(c, 10);
	EXPECT_EQ(res.at(1).second->size(), 1000);
	c = 0; 
	for(unsigned int i = 0; i < 1000; i++){
		c += res.at(0).second->at(i);
	}
	EXPECT_EQ(c, 9);
	delete d;
}


TEST_F(testLargeDatasets, _SUPER_SLOW_testMineClus2GB){
	DataReader* dr = new DataReader("test/testData/test2GB");
	MineClusGPUnified* d = new MineClusGPUnified(dr);
	auto res = d->findKClusters(2);

	EXPECT_EQ(res.at(0).first->size(), 400000);
	EXPECT_EQ(res.at(1).first->size(), 100000);

	EXPECT_EQ(res.at(0).second->size(), 1000);
	unsigned int c = 0; 
	for(unsigned int i = 0; i < 1000; i++){
		c += res.at(0).second->at(i);
	}
	EXPECT_EQ(c, 10);
	EXPECT_EQ(res.at(1).second->size(), 1000);
	c = 0; 
	for(unsigned int i = 0; i < 1000; i++){
		c += res.at(0).second->at(i);
	}
	EXPECT_EQ(c, 9);
	delete d;
}


TEST_F(testLargeDatasets, _SUPER_SLOW_testDOC4GB){
	std::cout << "hello " << std::endl;
	DataReader* dr = new DataReader("test/testData/test4GB");
	std::cout << "hello " << std::endl;
	DOCGPUnified* d = new DOCGPUnified(dr);
	d->setNumberOfSamples(1024); // upperbound to 1024
	auto res = d->findKClusters(2);

	EXPECT_EQ(res.at(0).first->size(), 800000);
	EXPECT_EQ(res.at(1).first->size(), 200000);

	EXPECT_EQ(res.at(0).second->size(), 1000);
	unsigned int c = 0; 
	for(unsigned int i = 0; i < 1000; i++){
		c += res.at(0).second->at(i);
	}
	EXPECT_EQ(c, 10);
	EXPECT_EQ(res.at(1).second->size(), 1000);
	c = 0; 
	for(unsigned int i = 0; i < 1000; i++){
		c += res.at(0).second->at(i);
	}
	EXPECT_EQ(c, 9);
	delete d;
}


TEST_F(testLargeDatasets, _SUPER_SLOW_testFastDOC4GB){
	DataReader* dr = new DataReader("test/testData/test4GB");
	Fast_DOCGPUnified* d = new Fast_DOCGPUnified(dr);
	auto res = d->findKClusters(2);

	EXPECT_EQ(res.at(0).first->size(), 800000);
	EXPECT_EQ(res.at(1).first->size(), 200000);

	EXPECT_EQ(res.at(0).second->size(), 10);
	EXPECT_EQ(res.at(1).second->size(), 9);
	delete d;
}


TEST_F(testLargeDatasets, _SUPER_SLOW_testMineClus4GB){
	DataReader* dr = new DataReader("test/testData/test4GB");
	MineClusGPUnified* d = new MineClusGPUnified(dr);
	auto res = d->findKClusters(2);

	EXPECT_EQ(res.at(0).first->size(), 800000);
	EXPECT_EQ(res.at(1).first->size(), 200000);

	EXPECT_EQ(res.at(0).second->size(), 10);
	EXPECT_EQ(res.at(1).second->size(), 9);
	delete d;
}



TEST_F(testLargeDatasets, _SUPER_SLOW_testDOC8GB){
	std::cout << "hello " << std::endl;
	DataReader* dr = new DataReader("test/testData/test8GB");
	std::cout << "hello " << std::endl;
	DOCGPUnified* d = new DOCGPUnified(dr);
	d->setNumberOfSamples(1024); // upperbound to 1024
	auto res = d->findKClusters(3);

	EXPECT_EQ(res.at(0).first->size(), 800000);
	EXPECT_EQ(res.at(1).first->size(), 800000);
	EXPECT_EQ(res.at(2).first->size(), 400000);

	EXPECT_EQ(res.at(0).second->size(), 10);
	EXPECT_EQ(res.at(1).second->size(), 9);
	EXPECT_EQ(res.at(2).second->size(), 8);	
	delete d;
}


TEST_F(testLargeDatasets, _SUPER_SLOW_testFastDOC8GB){
	DataReader* dr = new DataReader("test/testData/test8GB");
	Fast_DOCGPUnified* d = new Fast_DOCGPUnified(dr);
	auto res = d->findKClusters(2);

	EXPECT_EQ(res.at(0).first->size(), 800000);
	EXPECT_EQ(res.at(1).first->size(), 800000);
	EXPECT_EQ(res.at(2).first->size(), 400000);

	EXPECT_EQ(res.at(0).second->size(), 10);
	EXPECT_EQ(res.at(1).second->size(), 9);
	EXPECT_EQ(res.at(2).second->size(), 8);	
	delete d;
}


TEST_F(testLargeDatasets, _SUPER_SLOW_testMineClus8GB){
	DataReader* dr = new DataReader("test/testData/test8GB");
	MineClusGPUnified* d = new MineClusGPUnified(dr);
	auto res = d->findKClusters(2);

	EXPECT_EQ(res.at(0).first->size(), 800000);
	EXPECT_EQ(res.at(1).first->size(), 800000);
	EXPECT_EQ(res.at(2).first->size(), 400000);

	EXPECT_EQ(res.at(0).second->size(), 10);
	EXPECT_EQ(res.at(1).second->size(), 9);
	EXPECT_EQ(res.at(2).second->size(), 8);	
	delete d;
}
