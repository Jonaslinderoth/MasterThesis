#include <gtest/gtest.h>
#include <iostream>
#include "../src/DOC_GPU/DOCGPUnified.h"
#include <vector>
#include "testData.h"
#include <cmath>
#include <fstream>
#include <cstdio>
#include "../src/testingTools/DataGeneratorBuilder.h"
#include "../src/dataReader/Cluster.h"
#include "../src/dataReader/DataReader.h"
#include "../src/testingTools/MetaDataFileReader.h"
#include "testingTools.h"

    #include <unistd.h>
    #define GetCurrentDir getcwd
 

class testDOCGPUnified : public ::testing::Test {
public:
	// Per-test-suite set-up.
	// Called before the first test in this test suite.
	// Can be omitted if not needed.
	static void SetUpTestCase() {
		if(system("mkdir -p test/testData")){
		  
		};
	}

	// Per-test-suite tear-down.
	// Called after the last test in this test suite.
	// Can be omitted if not needed.
	static void TearDownTestCase() {

	}


	virtual void SetUp() {
	}



};




TEST_F(testDOCGPUnified, testConstructor3){
	auto v = new std::vector<std::vector<float>*>;
	auto v1 = new std::vector<float>;
	v1->push_back(1.1);
	v1->push_back(1.2);
	v1->push_back(1.3);
	v->push_back(v1);
	DOCGPUnified d = DOCGPUnified(v);
	d.setSeed(1);
	SUCCEED();
}





TEST_F(testDOCGPUnified, testFindCluster){
	std::vector<std::vector<float>*>* data = new std::vector<std::vector<float>*>;
	for(float i = 0; i < 200; i++){
		for(float j = 9; j < 15; j++){
			std::vector<float>* point1 = new std::vector<float>{j,i};
			data->push_back(point1);

		}
	}
	DOCGPUnified d = DOCGPUnified(data, 0.1, 0.25, 5);
	d.setSeed(1);
	auto res = d.findCluster();
	SUCCEED();




	/* int i = 0;
	   for(int i = 0; i < res.first.size(); i++){
		std::cout << res.first.at(i).at(0) << ", ";
	}
	std::cout << std::endl;

	i = 0;
	for(int i = 0; i < res.first.size(); i++){
		std::cout << res.first.at(i).at(1) << ", ";
	}
	std::cout << std::endl;
	*/

	EXPECT_TRUE(res.second->at(0));
	EXPECT_FALSE(res.second->at(1));

	int count = 0;
	EXPECT_EQ(res.first->size(), 1200);
}




TEST_F(testDOCGPUnified, testFindCluster2){
	std::vector<std::vector<float>*>* data = new std::vector<std::vector<float>*>;
	int a = 0;
	for(float i = 9; i <= 12; i++){
		for(float j = 9; j <= 12; j++){
			std::vector<float>* point1 = new std::vector<float>{i,j};
			data->push_back(point1);
			a++;
		}
	}

	int b = 0;
	for(float i = 60; i <= 65; i++){
		for(float j = 0; j <= 50; j++){
			std::vector<float>* point1 = new std::vector<float>{i,j};
			data->push_back(point1);
			b++;
		}
	}
	//std::cout << a << ", " << b << std::endl;

	DOCGPUnified d = DOCGPUnified(data, 0.1, 0.25, 5);
	d.setSeed(1);
	auto res = d.findCluster();
	SUCCEED();


	EXPECT_TRUE(res.second->at(0));
	EXPECT_FALSE(res.second->at(1));

	EXPECT_EQ(res.first->size(), 306);

	for(int i = 0; i< data->size(); i++){
		delete data->at(i);
	}
	delete data;
	delete res.first;
	delete res.second;
}



TEST_F(testDOCGPUnified, _SLOW_testFindCluster3){
	std::vector<std::vector<float>*>* data = data_4dim2cluster();

	//std::cout << a << ", " << b << std::endl;

	DOCGPUnified d = DOCGPUnified(data, 0.1, 0.25, 5);
	d.setSeed(1);
	auto res = d.findCluster();
	SUCCEED();


	EXPECT_TRUE(res.second->at(0));
	EXPECT_FALSE(res.second->at(1));
	EXPECT_FALSE(res.second->at(2));
	EXPECT_FALSE(res.second->at(3));

	EXPECT_LT(abs((int)res.first->size()-(int)numPoints_4dim2cluster()), 10);

	for(int i = 0; i< data->size(); i++){
		delete data->at(i);
	}
	delete data;
	delete res.first;
	delete res.second;
}


TEST_F(testDOCGPUnified, testFindKClusters){
	std::vector<std::vector<float>*>* data = data_4dim2cluster();


	DOCGPUnified d = DOCGPUnified(data, 0.1, 0.25, 5);
	d.setSeed(1);
	std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> > res = d.findKClusters(0);
	
	SUCCEED();
	EXPECT_EQ(res.size(), 0);

}

TEST_F(testDOCGPUnified, _SLOW_testFindKClusters2){
	std::vector<std::vector<float>*>* data = data_4dim2cluster();


	DOCGPUnified d = DOCGPUnified(data, 0.1, 0.25, 5);
	d.setSeed(1);
	std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> > res = d.findKClusters(1);
	
	SUCCEED();
	EXPECT_EQ(res.size(), 1);
	EXPECT_TRUE(res.at(0).second->at(0));
	EXPECT_FALSE(res.at(0).second->at(1));
	EXPECT_FALSE(res.at(0).second->at(2));
	EXPECT_FALSE(res.at(0).second->at(3));

	EXPECT_LT(abs((int)res.at(0).first->size()-(int)numPoints_4dim2cluster()), 10);

	for(int i = 0; i< data->size(); i++){
		delete data->at(i);
	}
	delete data;
	delete res.at(0).first;
	delete res.at(0).second;
}

// bool pointEQ(std::vector<float>* a1, std::vector<float>* a2){
// 	bool output = true;
// 	//EXPECT_EQ(a1->size(), a2->size());
// 	if(a1->size() != a2->size()){
// 		return false;
// 	}

// 	for(int j = 0; j < a2->size(); j++){
// 		auto b1 = a1->at(j);
// 		auto b2 = a2->at(j);
// 		output &= abs(b1 - b2) <= 0.0001;
// 		if(!output){
// 			break;
// 		}
// 	}
// 	//EXPECT_TRUE(output);
	
// 	return output;
// }

// bool disjoint(std::vector<std::vector<float>*>* a1, std::vector<std::vector<float>*>* a2){
// 	bool output = true;

// 	for(int i = 0; i < a1->size(); i++){
// 		for(int j = 0; j < a2->size(); j++){
// 			std::vector<float>* b1 = a1->at(i);
// 			std::vector<float>* b2 = a2->at(j);
// 			bool eq = pointEQ(b1, b2);
// 			output &= !eq;		
// 		}
// 	}
// 	return output;
// }


// bool equal(std::vector<std::vector<float>*>* a1, std::vector<std::vector<float>*>* a2){
// 	bool output = true;
// 	EXPECT_EQ(a1->size(), a2->size());
	
// 	for(int i = 0; i < a1->size(); i++){
// 		output = false;
// 		for(int j = 0; j < a2->size(); j++){
// 			auto b1 = a1->at(i);
// 			auto b2 = a2->at(j);
// 			auto eq = pointEQ(b1, b2);
// 			if(eq){output = eq; break;}
// 		}			
// 	}
// 	return output;
// }

TEST_F(testDOCGPUnified, testHelperFunctions){

	
	auto vec1 = new std::vector<std::vector<float>*>;
	auto a = new std::vector<float>{0,1,2,3,4};
	vec1->push_back(a);
	auto b = new std::vector<float>{0,1,2,3,5};
	vec1->push_back(b);

	EXPECT_FALSE(pointEQ(a,b));
	
	auto vec2 = new std::vector<std::vector<float>*>;
	auto a1 = new std::vector<float>{0,1,2,3,4};
	vec2->push_back(a1);
	auto b1 = new std::vector<float>{0,1,2,3,5};
	vec2->push_back(b1);
	
	EXPECT_TRUE(pointEQ(a,a1));

	auto vec3 = new std::vector<std::vector<float>*>;
	auto a3 = new std::vector<float>{1,1,2,3,4};
	vec3->push_back(a3);
	auto b3 = new std::vector<float>{2,1,2,3,5};
	vec3->push_back(b3);

	EXPECT_TRUE(equal(vec1, vec2));
	EXPECT_FALSE(equal(vec1, vec3));
	EXPECT_TRUE(disjoint(vec1, vec3));
	EXPECT_FALSE(disjoint(vec1, vec2));
}


TEST_F(testDOCGPUnified, _SLOW_testFindKClusters3){
	std::vector<std::vector<float>*>* data = data_4dim2cluster();


	DOCGPUnified d = DOCGPUnified(data, 0.1, 0.25, 7);
	d.setSeed(1);
	std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> > res = d.findKClusters(2);
	
	SUCCEED();
	EXPECT_EQ(res.size(), 1);
	EXPECT_TRUE(res.at(0).second->at(0));
	EXPECT_FALSE(res.at(0).second->at(1));
	EXPECT_FALSE(res.at(0).second->at(2));
	EXPECT_FALSE(res.at(0).second->at(3));


	EXPECT_LT(abs((int)res.at(0).first->size()-397), 5);

	
	for(int i = 0; i< data->size(); i++){
		delete data->at(i);
	}
	delete data;
	delete res.at(0).first;
	delete res.at(0).second;
}


TEST_F(testDOCGPUnified, testFindKClusters5){
	auto data = data_4dim2cluster();


	std::vector<std::vector<float>*>* data1 = new std::vector<std::vector<float>*>;
	std::vector<std::vector<float>*>* data2 = new std::vector<std::vector<float>*>;	
	std::vector<std::vector<float>*>* data3 = new std::vector<std::vector<float>*>;

	
	for(int i = 0; i < data->size(); i++){
		data1->push_back(data->at(i));
		data2->push_back(data->at(i));
		data3->push_back(data->at(i));
	}
	
	DOCGPUnified d1 = DOCGPUnified(data1, 0.4, 0.25, 3);
	DOCGPUnified d2 = DOCGPUnified(data2, 0.4, 0.25, 3);
	DOCGPUnified d3 = DOCGPUnified(data3, 0.4, 0.25, 3);


	d1.setSeed(1);
	d2.setSeed(1);
	d3.setSeed(1);
	
	
	std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> > res1 = d1.findKClusters(2);
	std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> > res2 = d2.findKClusters(2);
	std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> > res3 = d3.findKClusters(2);
	
	SUCCEED();
	

	equal(res1.at(0).first, res1.at(0).first);
	equal(res1.at(0).first, res2.at(0).first);
	equal(res1.at(0).first, res3.at(0).first);
}

TEST_F(testDOCGPUnified, _SLOW_testLarge5){
	std::vector<std::vector<float>*>* data = new std::vector<std::vector<float>*>;
	std::default_random_engine generator;
	generator.seed(100);
	std::uniform_real_distribution<double> distribution(50000.0,500000.0);
	std::uniform_real_distribution<double> distribution2(5.0,15.0);
	
	for(float i = 0; i < 100; i++){
		std::vector<float>* point1 = new std::vector<float>;
		point1->push_back(distribution2(generator));
		point1->push_back(distribution2(generator));
		point1->push_back(distribution2(generator));
		for(int j = 0; j < 3; j++){
			point1->push_back(distribution(generator));
		}

		data->push_back(point1);

	}
	
	DOCGPUnified d = DOCGPUnified(data);
	d.setSeed(1);
	d.setAlpha(0.1);
	d.setBeta(0.25);
	d.setWidth(15);
	auto res = d.findCluster();
	
	SUCCEED();
	EXPECT_EQ(res.second->size(),6);
	EXPECT_TRUE(res.second->at(0));
	EXPECT_TRUE(res.second->at(1));
	EXPECT_TRUE(res.second->at(2));
	for(int i = 3; i < 6;i++){
		EXPECT_FALSE(res.second->at(i));
	}
	EXPECT_EQ(res.first->size(), 100);

	/*EXPECT_EQ(res.first->size(), 100);
	for(int i = 0; i < res.first->size(); i++){
		for(int j = 0; j < 6; j++){
			std::cout << res.first->at(i)->at(j) << ", ";
		}
		std::cout <<std::endl;
		}*/
}


TEST_F(testDOCGPUnified, testFindKClusters4){

	std::vector<std::vector<float>*>* data = new std::vector<std::vector<float>*>;
	auto a = new std::vector<float>{0,1};
	data->push_back(a);


	DOCGPUnified d = DOCGPUnified(data, 0.1, 0.25, 6);
	std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> > res = d.findKClusters(2);
	
	SUCCEED();
	EXPECT_EQ(res.size(), 1);
	EXPECT_TRUE(res.at(0).second->at(0));
	EXPECT_TRUE(res.at(0).second->at(1));



	
	for(int i = 0; i< data->size(); i++){
		delete data->at(i);
	}
	delete data;
	delete res.at(0).first;
	delete res.at(0).second;
}




TEST_F(testDOCGPUnified, testFindKClusters6){
	std::vector<std::vector<float>*>* data = data_2dim2cluster();


	DOCGPUnified d = DOCGPUnified(data, 0.1, 0.25, 5);
	d.setSeed(1);
	std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> > res = d.findKClusters(2);
	
	SUCCEED();
	EXPECT_EQ(res.size(), 2);



	EXPECT_LT(abs((int)res.at(0).first->size()-numPoints_2dim2cluster2()[0]), 10);
	EXPECT_LT(abs((int)res.at(1).first->size()-numPoints_2dim2cluster2()[1]), 10);
	EXPECT_TRUE(disjoint(res.at(0).first, res.at(1).first));

	EXPECT_TRUE(res.at(0).second->at(0));
	EXPECT_TRUE(res.at(0).second->at(1));
	EXPECT_TRUE(res.at(1).second->at(0));
	EXPECT_TRUE(res.at(1).second->at(1));

	

	
	for(int i = 0; i< data->size(); i++){
		delete data->at(i);
	}
	delete data;
	delete res.at(0).first;
	delete res.at(0).second;
}

TEST_F(testDOCGPUnified, testWithDataReader){
	DataGeneratorBuilder dgb;
	dgb.setSeed(1);
	Cluster small;
	small.setAmmount(10);
	small.addDimension(uniformDistribution, {-10000,10000});
	small.addDimension(normalDistribution, {-10000,10000}, {50,2});
	dgb.addCluster(small);
	dgb.setFileName("test/testData/test1");
	dgb.build(false);
	DataReader* dr = new DataReader("test/testData/test1");

	DOCGPUnified d(dr);
	d.setSeed(1);
	std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> > res = d.findKClusters(2);
	EXPECT_FALSE(res.at(0).second->at(0));
	EXPECT_TRUE(res.at(0).second->at(1));
	EXPECT_EQ(res.at(0).first->size(), 10);
}


TEST_F(testDOCGPUnified, testWithDataReader2){
	DataGeneratorBuilder dgb;
	dgb.setSeed(1);
	Cluster small;
	small.setAmmount(1000);
	small.addDimension(uniformDistribution, {-10000,10000});
	small.addDimension(normalDistribution, {-10000,10000}, {50,2});
	dgb.addCluster(small);
	dgb.setFileName("test/testData/test2");
	dgb.build(false);
	DataReader* dr = new DataReader("test/testData/test2");

	DOCGPUnified d(dr);
	d.setSeed(1);
	std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> > res = d.findKClusters(2);
	EXPECT_EQ(res.size(), 1);
	EXPECT_FALSE(res.at(0).second->at(0));
	EXPECT_TRUE(res.at(0).second->at(1));
	EXPECT_EQ(res.at(0).first->size(), 1000);
	
}
	

TEST_F(testDOCGPUnified, SLOW_testWithDataReader3){
	DataGeneratorBuilder dgb;
	dgb.setSeed(10);
	Cluster small;
	small.setAmmount(20);
	small.addDimension(uniformDistribution, {-10000,10000});
	small.addDimension(normalDistribution, {-10000,10000}, {50,1,1});
	small.addDimension(uniformDistribution, {-10000,10000});
	dgb.addCluster(small);
	
	Cluster small2;
	small2.setAmmount(1000);
	small2.addDimension(uniformDistribution, {-10000,10000});
	small2.addDimension(normalDistribution, {-10000,10000}, {5,2,1});
	small2.addDimension(uniformDistribution, {-10000,10000});
	dgb.addCluster(small2);

	
	dgb.setFileName("test/testData/test33");
	dgb.build(true);
	DataReader* dr = new DataReader("test/testData/test33");

	DOCGPUnified d(dr);
	d.setSeed(10);
	d.setBeta(0.3);
	d.setWidth(10);
	std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> > res = d.findKClusters(2);
	EXPECT_EQ(res.at(0).first->at(0)->size(), 3);
	
	EXPECT_FALSE(res.at(0).second->at(0));
	EXPECT_TRUE(res.at(0).second->at(1));
	EXPECT_FALSE(res.at(0).second->at(2));
	EXPECT_EQ(res.at(0).first->size(), 1000);

	
	EXPECT_FALSE(res.at(1).second->at(0));
	EXPECT_TRUE(res.at(1).second->at(1));
	EXPECT_FALSE(res.at(1).second->at(2));
	EXPECT_EQ(res.at(1).first->size(), 20);
}
	

TEST_F(testDOCGPUnified, SLOW_testWithDataReader4){
	DataGeneratorBuilder dgb;
	dgb.setSeed(1);
	Cluster small;
	small.setAmmount(1000);
	small.addDimension(normalDistribution, {-10000,10000}, {50,2});
	small.addDimension(uniformDistribution, {-10000,10000});
	
	dgb.addCluster(small);
	Cluster small2;
	small2.setAmmount(20);

	small2.addDimension(uniformDistribution, {-10000,10000});
	small2.addDimension(normalDistribution, {-10000,10000}, {5,2});
	small2.addDimension(uniformDistribution, {-10000,10000});
	dgb.addCluster(small2);

	
	dgb.setFileName("test/testData/test4");
	dgb.build(false);
	DataReader* dr = new DataReader("test/testData/test4");

	DOCGPUnified d(dr);
	d.setSeed(1);
	d.setBeta(0.3);
	d.setWidth(8);
	std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> > res = d.findKClusters(2);
	EXPECT_EQ(res.at(0).first->at(0)->size(), 3);
	
	//EXPECT_FALSE(res.at(0).second->at(0));
	EXPECT_TRUE(res.at(0).second->at(1-1));
	EXPECT_FALSE(res.at(0).second->at(2-1));
	EXPECT_EQ(res.at(0).first->size(), 1000);

	
	//EXPECT_FALSE(res.at(1).second->at(0));
	EXPECT_FALSE(res.at(1).second->at(1-1));
	EXPECT_TRUE(res.at(1).second->at(2-1));
	EXPECT_EQ(res.at(1).first->size(), 20);

}


TEST_F(testDOCGPUnified, SLOW_testWithDataReader5){
	
	DataGeneratorBuilder dgb;
	dgb.setSeed(2);
	Cluster small;
	small.setAmmount(100);
	small.addDimension(normalDistribution, {-10000,10000}, {50,2});
	small.addDimension(uniformDistribution, {-10000,10000});
	small.addDimension(normalDistribution, {-10000,10000}, {50,2});
	small.addDimension(uniformDistribution, {-10000,10000});
	small.addDimension(normalDistribution, {-10000,10000}, {50,2});
	small.addDimension(uniformDistribution, {-10000,10000});
	
	dgb.addCluster(small);
	Cluster small2;
	small2.setAmmount(20);

	small2.addDimension(uniformDistribution, {-10000,10000});
	small2.addDimension(uniformDistribution, {-10000,10000});
	small2.addDimension(uniformDistribution, {-10000,10000});
	small2.addDimension(uniformDistribution, {-10000,10000});
	small2.addDimension(uniformDistribution, {-10000,10000});
    small2.addDimension(uniformDistribution, {-10000,10000});

	dgb.addCluster(small2);

	
	dgb.setFileName("test/testData/test5");
	dgb.build(false);

	DataReader* dr = new DataReader("test/testData/test5");

	DOCGPUnified d(dr);
	d.setSeed(2);
	d.setWidth(8);

	std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> > res = d.findKClusters(2);
	EXPECT_EQ(res.at(0).first->at(0)->size(), 6);
	
	//EXPECT_FALSE(res.at(0).second->at(0));
	EXPECT_TRUE(res.at(0).second->at(0));
	EXPECT_FALSE(res.at(0).second->at(1));
	EXPECT_TRUE(res.at(0).second->at(2));
	EXPECT_FALSE(res.at(0).second->at(3));
	EXPECT_TRUE(res.at(0).second->at(4));
	EXPECT_FALSE(res.at(0).second->at(5));
	EXPECT_EQ(res.at(0).first->size(), 100);

	
	EXPECT_FALSE(res.at(1).second->at(0));
	EXPECT_FALSE(res.at(1).second->at(1));
	EXPECT_FALSE(res.at(1).second->at(2));
	EXPECT_FALSE(res.at(1).second->at(3));
	EXPECT_FALSE(res.at(1).second->at(4));
	EXPECT_FALSE(res.at(1).second->at(5));
	EXPECT_EQ(res.at(1).first->size(), 20);
	
}
