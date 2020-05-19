#include <gtest/gtest.h>
#include <vector>
#include <math.h>
#include "../src/testingTools/DataGeneratorBuilder.h"
#include "../src/dataReader/Cluster.h"
#include "../src/dataReader/DataReader.h"
#include "../src/testingTools/MetaDataFileReader.h"
#include "../src/testingTools/RandomFunction.h"
#include "../src/DOC_GPU/DOCGPU.h"
#include "../src/Fast_DOCGPU/Fast_DOCGPU.h"
#include "../src/MineClusGPU/MineClusGPU.h"




bool pointEq(std::vector<float>* a, std::vector<float>* b){
	bool result = true;
	result &= a->size() == b->size();
	for(unsigned int i = 0; i < a->size(); i++){
		result &= abs(a->at(i) - b->at(i)) <= 0.000001;
		if (!result) return result;
	}
	return result;
}

bool pointInCluster(std::vector<std::vector<float>*>* cluster, std::vector<float>* point){
	for(unsigned int i = 0; i < cluster->size(); i++){
		if(pointEq(cluster->at(i), point)){
			return true;
		}
	}
	return false;
}


std::vector<std::vector<unsigned int>> confusion(std::vector<std::vector<std::vector<float>*>*>* labels,std::vector<std::vector<std::vector<float>*>*>* clusters){
	std::vector<std::vector<unsigned int>> result = std::vector<std::vector<unsigned int>>(
		labels->size(),
		std::vector<unsigned int>(clusters->size(),0));

	for(unsigned int i = 0; i < labels->size(); i++){ // for each cluster
		for(unsigned int l = 0; l < clusters->size(); l++){ // for each cluster
			//entry i,l in confusion matrix
			unsigned int count = 0;
			for(unsigned int j = 0; j < clusters->at(l)->size(); j++){
				count += pointInCluster(labels->at(i), clusters->at(l)->at(j));
			}
			result.at(i).at(l) = count;
		}		
	}
	return result;
};

std::vector<std::vector<unsigned int>> confusion(std::vector<std::vector<std::vector<float>*>*>* labels,
												 std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*>> clusters){
	std::vector<std::vector<unsigned int>> result = std::vector<std::vector<unsigned int>>(
		labels->size(),
		std::vector<unsigned int>(clusters.size(),0));

	for(unsigned int i = 0; i < labels->size(); i++){ // for each cluster
		for(unsigned int l = 0; l < clusters.size(); l++){ // for each cluster
			//entry i,l in confusion matrix
			unsigned int count = 0;
			for(unsigned int j = 0; j < clusters.at(l).first->size(); j++){
				count += pointInCluster(labels->at(i), clusters.at(l).first->at(j));
			}
			result.at(i).at(l) = count;
		}		
	}
	return result;
};

float accuracy(std::vector<std::vector<unsigned int>> confusion){
	float result = 0;
	unsigned int trues = 0;
	unsigned int total = 0;
	for(unsigned int i = 0; i < confusion.size(); i++){ // for each label
		unsigned int maxIndex = 0; 
		for(unsigned int l = 0; l < confusion.at(i).size(); l++){ // for each cluster
			if(confusion.at(i).at(l) > confusion.at(i).at(maxIndex)){
				maxIndex = l;
			}
			total += confusion.at(i).at(l);
		}

		trues += confusion.at(i).at(maxIndex);
	}
	result += (float)trues/total;
	return result;	
}



std::vector<std::vector<std::vector<float>*>*>* getCluster(std::string path){
	auto dr = new DataReader(path);
	auto mdr = new MetaDataFileReader(path);

	std::vector<std::vector<std::vector<float>*>*>* labels = new std::vector<std::vector<std::vector<float>*>*>;
	for(unsigned int i = 0; i < mdr->getClusterLines().size(); i++){
		labels->push_back(new std::vector<std::vector<float>*>);		
	}
	while(dr->isThereANextPoint()){
		labels->at(mdr->nextCheat())->push_back(dr->nextPoint());
	}
	delete dr;
	delete mdr;
	return labels;
}




TEST(testAccuracy, testSimple){
	auto clusters = new std::vector<std::vector<std::vector<float>*>*>;
	auto labels = new std::vector<std::vector<std::vector<float>*>*>;

	{
		auto point1 = new std::vector<float>{1,2};
		auto point2 = new std::vector<float>{2,3};
		auto cluster1 = new std::vector<std::vector<float>*>{point1,point2};
		auto point3 = new std::vector<float>{3,4};
		auto point4 = new std::vector<float>{4,5};
		auto cluster2 = new std::vector<std::vector<float>*>{point3,point4};
		clusters->push_back(cluster1);
		clusters->push_back(cluster2);
	}

	{
		auto point1 = new std::vector<float>{1,2};
		auto point2 = new std::vector<float>{2,3};
		auto cluster1 = new std::vector<std::vector<float>*>{point1,point2};
		auto point3 = new std::vector<float>{3,4};
		auto point4 = new std::vector<float>{4,5};
		auto cluster2 = new std::vector<std::vector<float>*>{point3,point4};
		labels->push_back(cluster1);
		labels->push_back(cluster2);
	}

	auto res = confusion(labels, clusters);

	EXPECT_EQ(res.at(0).at(0), 2);
	EXPECT_EQ(res.at(1).at(1), 2);
	
	EXPECT_EQ(res.at(0).at(1), 0);
	EXPECT_EQ(res.at(1).at(0), 0);

	EXPECT_FLOAT_EQ(accuracy(res), 1);
}


TEST(testAccuracy, testSimple2){
	auto clusters = new std::vector<std::vector<std::vector<float>*>*>;
	auto labels = new std::vector<std::vector<std::vector<float>*>*>;

	{
		auto point1 = new std::vector<float>{1,2};
		auto point2 = new std::vector<float>{2,3};
		auto cluster1 = new std::vector<std::vector<float>*>{point1,point2};
		auto point3 = new std::vector<float>{3,4};
		auto point4 = new std::vector<float>{4,5};
		auto cluster2 = new std::vector<std::vector<float>*>{point3,point4};
		auto point5 = new std::vector<float>{5,6};
		auto point6 = new std::vector<float>{6,7};
		auto cluster3 = new std::vector<std::vector<float>*>{point5,point6};
		
		clusters->push_back(cluster1);
		clusters->push_back(cluster2);
		clusters->push_back(cluster3);
	}

	{
		auto point1 = new std::vector<float>{1,2};
		auto point2 = new std::vector<float>{2,3};
		auto cluster1 = new std::vector<std::vector<float>*>{point1,point2};
		auto point3 = new std::vector<float>{3,4};
		auto point4 = new std::vector<float>{4,5};
		auto cluster2 = new std::vector<std::vector<float>*>{point3,point4};
		auto point6 = new std::vector<float>{6,7};
		auto point5 = new std::vector<float>{5,6};
		auto cluster3 = new std::vector<std::vector<float>*>{point5,point6};
		labels->push_back(cluster1);
		labels->push_back(cluster2);
		labels->push_back(cluster3);
	}

	auto res = confusion(labels, clusters);

	EXPECT_EQ(res.at(0).at(0), 2);
	EXPECT_EQ(res.at(1).at(1), 2);
	EXPECT_EQ(res.at(2).at(2), 2);
	
	EXPECT_EQ(res.at(0).at(1), 0);
	EXPECT_EQ(res.at(0).at(2), 0);
	EXPECT_EQ(res.at(1).at(0), 0);
	EXPECT_EQ(res.at(1).at(2), 0);
	EXPECT_EQ(res.at(2).at(0), 0);
	EXPECT_EQ(res.at(2).at(1), 0);
	
	EXPECT_FLOAT_EQ(accuracy(res), 1);
}


TEST(testAccuracy, testSimple3){
	auto clusters = new std::vector<std::vector<std::vector<float>*>*>;
	auto labels = new std::vector<std::vector<std::vector<float>*>*>;

	{
		auto point1 = new std::vector<float>{1,2};
		auto point2 = new std::vector<float>{2,3};
		auto cluster1 = new std::vector<std::vector<float>*>{point1,point2};
		auto point3 = new std::vector<float>{3,4};
		auto point4 = new std::vector<float>{4,5};
		auto cluster2 = new std::vector<std::vector<float>*>{point3,point4};
		auto point5 = new std::vector<float>{5,6};
		auto point6 = new std::vector<float>{6,7};
		auto cluster3 = new std::vector<std::vector<float>*>{point5,point6};
		
		clusters->push_back(cluster1);
		clusters->push_back(cluster2);
		clusters->push_back(cluster3);
	}

	{
		auto point2 = new std::vector<float>{2,3};
		auto point1 = new std::vector<float>{1,2};
		auto point1_1 = new std::vector<float>{9,9};
		auto cluster1 = new std::vector<std::vector<float>*>{point1,point2,point1_1};
		auto point3 = new std::vector<float>{3,4};
		auto point4 = new std::vector<float>{4,5};
		auto cluster2 = new std::vector<std::vector<float>*>{point3,point4};
		auto point6 = new std::vector<float>{6,7};
		auto point5 = new std::vector<float>{5,6};
		auto cluster3 = new std::vector<std::vector<float>*>{point5,point6};
		labels->push_back(cluster1);
		labels->push_back(cluster2);
		labels->push_back(cluster3);
	}

	auto res = confusion(labels, clusters);

	EXPECT_EQ(res.at(0).at(0), 2);
	EXPECT_EQ(res.at(1).at(1), 2);
	EXPECT_EQ(res.at(2).at(2), 2);
	
	EXPECT_EQ(res.at(0).at(1), 0);
	EXPECT_EQ(res.at(0).at(2), 0);
	EXPECT_EQ(res.at(1).at(0), 0);
	EXPECT_EQ(res.at(1).at(2), 0);
	EXPECT_EQ(res.at(2).at(0), 0);
	EXPECT_EQ(res.at(2).at(1), 0);

	EXPECT_FLOAT_EQ(accuracy(res), 1);
}


TEST(testAccuracy, testSimple4){
	auto clusters = new std::vector<std::vector<std::vector<float>*>*>;
	auto labels = new std::vector<std::vector<std::vector<float>*>*>;

	{
		auto point1 = new std::vector<float>{1,2};
		auto point2 = new std::vector<float>{2,3};
		auto cluster1 = new std::vector<std::vector<float>*>{point1,point2};
		auto point3 = new std::vector<float>{3,4};
		auto point4 = new std::vector<float>{4,5};
		auto cluster2 = new std::vector<std::vector<float>*>{point3,point4};
		auto point5 = new std::vector<float>{5,6};
		auto point6 = new std::vector<float>{6,7};
		auto point1_1 = new std::vector<float>{9,9};
		auto cluster3 = new std::vector<std::vector<float>*>{point5,point6,point1_1};
		
		clusters->push_back(cluster1);
		clusters->push_back(cluster2);
		clusters->push_back(cluster3);
	}

	{
		auto point2 = new std::vector<float>{2,3};
		auto point1 = new std::vector<float>{1,2};
		auto point1_1 = new std::vector<float>{9,9};
		auto cluster1 = new std::vector<std::vector<float>*>{point1,point2,point1_1};
		auto point3 = new std::vector<float>{3,4};
		auto point4 = new std::vector<float>{4,5};
		auto cluster2 = new std::vector<std::vector<float>*>{point3,point4};
		auto point6 = new std::vector<float>{6,7};
		auto point5 = new std::vector<float>{5,6};
		auto cluster3 = new std::vector<std::vector<float>*>{point5,point6};
		labels->push_back(cluster1);
		labels->push_back(cluster2);
		labels->push_back(cluster3);
	}

	auto res = confusion(labels, clusters);

	EXPECT_EQ(res.at(0).at(0), 2);
	EXPECT_EQ(res.at(1).at(1), 2);
	EXPECT_EQ(res.at(2).at(2), 2);
	
	EXPECT_EQ(res.at(0).at(1), 0);
	EXPECT_EQ(res.at(0).at(2), 1);
	EXPECT_EQ(res.at(1).at(0), 0);
	EXPECT_EQ(res.at(1).at(2), 0);
	EXPECT_EQ(res.at(2).at(0), 0);
	EXPECT_EQ(res.at(2).at(1), 0);

	EXPECT_FLOAT_EQ(accuracy(res), ((float)6/7));
}


TEST(testAccuracy, testDataReader){
	DataGeneratorBuilder dgb;
	dgb.setSeed(0);
	bool res2 = dgb.buildUClusters("test2222",20,2,15,2,1,0);
	EXPECT_TRUE(res2);

	auto dr = new DataReader("test2222");
	EXPECT_EQ(dr->getSize(), 40);
	auto mdr = new MetaDataFileReader("test2222");

	std::vector<std::vector<std::vector<float>*>*>* labels = new std::vector<std::vector<std::vector<float>*>*>;
	labels->push_back(new std::vector<std::vector<float>*>);
	labels->push_back(new std::vector<std::vector<float>*>);
	
	while(dr->isThereANextPoint()){
		labels->at(mdr->nextCheat())->push_back(dr->nextPoint());
	}
	auto res = confusion(labels, labels);
	EXPECT_FLOAT_EQ(accuracy(res), 1);
	
}


TEST(testAccuracy, testDataReader2){
	DataGeneratorBuilder dgb;
	dgb.setSeed(0);
	bool res2 = dgb.buildUClusters("test2222",20,2,15,10,2,0, true);
	EXPECT_TRUE(res2);

	auto dr = new DataReader("test2222");
	EXPECT_EQ(dr->getSize(), 40);
	auto mdr = new MetaDataFileReader("test2222");

	std::vector<std::vector<std::vector<float>*>*>* labels = new std::vector<std::vector<std::vector<float>*>*>;
	labels->push_back(new std::vector<std::vector<float>*>);
	labels->push_back(new std::vector<std::vector<float>*>);
	while(dr->isThereANextPoint()){
		labels->at(mdr->nextCheat())->push_back(dr->nextPoint());
	}

	
	DOCGPU d = DOCGPU(new DataReader("test2222"));
	d.setWidth(15);
	d.setSeed(1);
	auto cluster = d.findKClusters(5);
	EXPECT_EQ(cluster.size(), 2);
	EXPECT_EQ(cluster.at(0).first->size(), 20);
	EXPECT_EQ(cluster.at(1).first->size(), 20);
	
	auto res = confusion(labels, cluster);
	EXPECT_FLOAT_EQ(accuracy(res), 1);
	
}


TEST(testAccuracy, SLOW_testDataReader3){
	DataGeneratorBuilder dgb;
	dgb.setSeed(0);
	bool res2 = dgb.buildUClusters("test2222",200,10,15,10,2,0, true);
	EXPECT_TRUE(res2);

	auto dr = new DataReader("test2222");
	EXPECT_EQ(dr->getSize(), 2000);
	auto mdr = new MetaDataFileReader("test2222");

	std::vector<std::vector<std::vector<float>*>*>* labels = new std::vector<std::vector<std::vector<float>*>*>;
	labels->push_back(new std::vector<std::vector<float>*>);
	labels->push_back(new std::vector<std::vector<float>*>);
	labels->push_back(new std::vector<std::vector<float>*>);
	labels->push_back(new std::vector<std::vector<float>*>);
	labels->push_back(new std::vector<std::vector<float>*>);
	labels->push_back(new std::vector<std::vector<float>*>);
	labels->push_back(new std::vector<std::vector<float>*>);
	labels->push_back(new std::vector<std::vector<float>*>);
	labels->push_back(new std::vector<std::vector<float>*>);
	labels->push_back(new std::vector<std::vector<float>*>);
	while(dr->isThereANextPoint()){
		labels->at(mdr->nextCheat())->push_back(dr->nextPoint());
	}

	
	DOCGPU d = DOCGPU(new DataReader("test2222"));
	d.setWidth(15);
	d.setSeed(1);
	auto cluster = d.findKClusters(10);
	EXPECT_EQ(cluster.size(), 10);
	
	auto res = confusion(labels, cluster);

	for(unsigned int i = 0; i < res.size(); i++){
		for(unsigned int j = 0; j < res.at(i).size(); j++){
			std::cout << res.at(i).at(j) << " ";
		}
		std::cout << std::endl;
	}
	

	
	EXPECT_GT(accuracy(res), 0.99);
	
}


TEST(testAccuracy, SLOW_testDataReader4){
	DataGeneratorBuilder dgb;
	dgb.setSeed(0);
	bool res2 = dgb.buildUClusters("test2222",200,10,15,10,2,0, true);
	EXPECT_TRUE(res2);

	auto labels = getCluster("test2222");

	
	DOCGPU d = DOCGPU(new DataReader("test2222"));
	d.setWidth(15);
	d.setSeed(1);
	auto cluster = d.findKClusters(10);
	EXPECT_EQ(cluster.size(), 10);
	
	auto res = confusion(labels, cluster);

	for(unsigned int i = 0; i < res.size(); i++){
		for(unsigned int j = 0; j < res.at(i).size(); j++){
			std::cout << res.at(i).at(j) << " ";
		}
		std::cout << std::endl;
	}
	
	EXPECT_GT(accuracy(res), 0.99);
	
 }



TEST(testAccuracy, SLOW_testDataReader4_2){
	DataGeneratorBuilder dgb;
	dgb.setSeed(0);
	bool res2 = dgb.buildUClusters("test2222",200,10,15,10,2,0, true);
	EXPECT_TRUE(res2);

	auto labels = getCluster("test2222");

	
	DOCGPU d = DOCGPU(new DataReader("test2222"));
	d.setWidth(15);
	d.setSeed(1);
	d.setNumberOfSamples(1024*4);
	auto cluster = d.findKClusters(10);
	EXPECT_EQ(cluster.size(), 10);
	
	auto res = confusion(labels, cluster);

	for(unsigned int i = 0; i < res.size(); i++){
		for(unsigned int j = 0; j < res.at(i).size(); j++){
			std::cout << res.at(i).at(j) << " ";
		}
		std::cout << std::endl;
	}
	
	EXPECT_GT(accuracy(res), 0.99);
	
 }

TEST(testAccuracy, SLOW_testDataReader5){
	DataGeneratorBuilder dgb;
	dgb.setSeed(0);
	bool res2 = dgb.buildUClusters("test2222",200,10,15,10,2,0, true);
	EXPECT_TRUE(res2);

	auto labels = getCluster("test2222");

	
	Fast_DOCGPU d = Fast_DOCGPU(new DataReader("test2222"));
	d.setWidth(15);
	d.setSeed(1);
	auto cluster = d.findKClusters(10);
	EXPECT_EQ(cluster.size(), 10);
	
	auto res = confusion(labels, cluster);

	for(unsigned int i = 0; i < res.size(); i++){
		for(unsigned int j = 0; j < res.at(i).size(); j++){
			std::cout << res.at(i).at(j) << " ";
		}
		std::cout << std::endl;
	}
	
	EXPECT_GT(accuracy(res), 0.92);
	
 }

TEST(testAccuracy, SLOW_testDataReader6){
	DataGeneratorBuilder dgb;
	dgb.setSeed(0);
	bool res2 = dgb.buildUClusters("test2222",200,10,15,10,2,0, true);
	EXPECT_TRUE(res2);

	auto labels = getCluster("test2222");

	
	MineClusGPU d = MineClusGPU(new DataReader("test2222"));
	d.setWidth(15);
	d.setSeed(1);
	auto cluster = d.findKClusters(10);
	EXPECT_EQ(cluster.size(), 4);
	
	auto res = confusion(labels, cluster);

	for(unsigned int i = 0; i < res.size(); i++){
		for(unsigned int j = 0; j < res.at(i).size(); j++){
			std::cout << res.at(i).at(j) << " ";
		}
		std::cout << std::endl;
	}
	EXPECT_GT(accuracy(res), 0.99);
	
 }
