#include <gtest/gtest.h>
#include "../src/MineClus/MineClus.h"
TEST(testMineClus, testFindClusterSimple1){
	auto data = new std::vector<std::vector<float>*>;
	{
		auto point = new std::vector<float>{1,10000};
		data->push_back(point);
	}
	{
		auto point = new std::vector<float>{2,10000};
		data->push_back(point);
	}
	{
		auto point = new std::vector<float>{3,10000};
		data->push_back(point);
	}
	{
		auto point = new std::vector<float>{4,10000};
		data->push_back(point);
	}
	{
		auto point = new std::vector<float>{10000,1};
		data->push_back(point);
	}

	auto c = new MineClus(data);
	c->setSeed(2);
	auto res = c->findCluster();
	
	EXPECT_EQ(res.first->size(), 4);
	EXPECT_EQ(res.second->size(), 2);
	EXPECT_TRUE(res.second->at(0));
	EXPECT_TRUE(res.second->at(1));
}



TEST(testMineClus, testFindClusterlarge1){
	unsigned int dim = 10;
	unsigned int numPoints = 1000;
	auto data =  new std::vector<std::vector<float>*>();

	std::default_random_engine generator;
	generator.seed(1);
	std::normal_distribution<float> cluster(5.0,2.0);
	std::uniform_int_distribution<> outlier(-10000,10000);
	
	
	for(int i = 0; i < numPoints; i++){
		auto point = new std::vector<float>;
		for(int j = 0; j < dim; j++){
			if(j%2 == 0){
				point->push_back(cluster(generator));
			}else{
				point->push_back(outlier(generator));
			}
		}
		data->push_back(point);
	}

		auto c = new MineClus(data);
	c->setSeed(2);
	auto res = c->findCluster();
	
	EXPECT_EQ(res.first->size(), 1000);
	EXPECT_EQ(res.second->size(), 10);
	EXPECT_TRUE(res.second->at(0));
	EXPECT_FALSE(res.second->at(1));
	EXPECT_TRUE(res.second->at(2));
	EXPECT_FALSE(res.second->at(3));
	EXPECT_TRUE(res.second->at(4));
	EXPECT_FALSE(res.second->at(5));
	EXPECT_TRUE(res.second->at(6));
	EXPECT_FALSE(res.second->at(7));
	EXPECT_TRUE(res.second->at(8));
	EXPECT_FALSE(res.second->at(9));
	
}



TEST(testMineClus, testFindClusterlarge2withOutliers){
	unsigned int dim = 10;
	unsigned int numPoints = 1000;
	auto data =  new std::vector<std::vector<float>*>();

	std::default_random_engine generator;
	generator.seed(1);
	std::normal_distribution<float> cluster(5.0,2.0);
	std::uniform_int_distribution<> outlier(-10000,10000);
	
	
	for(int i = 0; i < numPoints; i++){
		auto point = new std::vector<float>;
		for(int j = 0; j < dim; j++){
			if(j%2 == 0){
				point->push_back(cluster(generator));
			}else{
				point->push_back(outlier(generator));
			}
		}
		data->push_back(point);
	}

	for(int i = 0; i < numPoints*0.1; i++){
		auto point = new std::vector<float>;
		for(int j = 0; j < dim; j++){
			point->push_back(outlier(generator));
		}
		data->push_back(point);
	}	

	auto c = new MineClus(data);
	c->setSeed(2);
	auto res = c->findCluster();
	
	EXPECT_EQ(res.first->size(), 1000);
	EXPECT_EQ(res.second->size(), 10);
	EXPECT_TRUE(res.second->at(0));
	EXPECT_FALSE(res.second->at(1));
	EXPECT_TRUE(res.second->at(2));
	EXPECT_FALSE(res.second->at(3));
	EXPECT_TRUE(res.second->at(4));
	EXPECT_FALSE(res.second->at(5));
	EXPECT_TRUE(res.second->at(6));
	EXPECT_FALSE(res.second->at(7));
	EXPECT_TRUE(res.second->at(8));
	EXPECT_FALSE(res.second->at(9));
	
}




TEST(testMineClus, _SLOW_testFindClusterlarge3withOutliers){
	unsigned int dim = 20;
	unsigned int numPoints = 10000;
	auto data =  new std::vector<std::vector<float>*>();

	std::default_random_engine generator;
	generator.seed(1);
	std::normal_distribution<float> cluster(5.0,2.0);
	std::uniform_int_distribution<> outlier(-10000,10000);
	
	
	for(int i = 0; i < numPoints; i++){
		auto point = new std::vector<float>;
		for(int j = 0; j < dim; j++){
			if(j%2 == 0){
				point->push_back(cluster(generator));
			}else{
				point->push_back(outlier(generator));
			}
		}
		data->push_back(point);
	}

	for(int i = 0; i < numPoints*0.1; i++){
		auto point = new std::vector<float>;
		for(int j = 0; j < dim; j++){
			point->push_back(outlier(generator));
		}
		data->push_back(point);
	}	

	auto c = new MineClus(data);
	c->setSeed(2);
	auto res = c->findCluster();
	
	EXPECT_EQ(res.first->size(), 10000);
	EXPECT_EQ(res.second->size(), 20);
	EXPECT_TRUE(res.second->at(0));
	EXPECT_FALSE(res.second->at(1));
	EXPECT_TRUE(res.second->at(2));
	EXPECT_FALSE(res.second->at(3));
	EXPECT_TRUE(res.second->at(4));
	EXPECT_FALSE(res.second->at(5));
	EXPECT_TRUE(res.second->at(6));
	EXPECT_FALSE(res.second->at(7));
	EXPECT_TRUE(res.second->at(8));
	EXPECT_FALSE(res.second->at(9));
	EXPECT_TRUE(res.second->at(10));
	EXPECT_FALSE(res.second->at(11));
	EXPECT_TRUE(res.second->at(12));
	EXPECT_FALSE(res.second->at(13));
	EXPECT_TRUE(res.second->at(14));
	EXPECT_FALSE(res.second->at(15));
	EXPECT_TRUE(res.second->at(16));
	EXPECT_FALSE(res.second->at(17));
	EXPECT_TRUE(res.second->at(18));
	EXPECT_FALSE(res.second->at(19));	
}



TEST(testMineClus, testFind2ClusterSimple1){
	auto data = new std::vector<std::vector<float>*>;
	{
		auto point = new std::vector<float>{1,10000};
		data->push_back(point);
	}
	{
		auto point = new std::vector<float>{2,10000};
		data->push_back(point);
	}
	{
		auto point = new std::vector<float>{3,10000};
		data->push_back(point);
	}
	{
		auto point = new std::vector<float>{4,10000};
		data->push_back(point);
	}
	{
		auto point = new std::vector<float>{10000,1};
		data->push_back(point);
	}
	{
		auto point = new std::vector<float>{50,10000};
		data->push_back(point);
	}
	{
		auto point = new std::vector<float>{51,10000};
		data->push_back(point);
	}
	{
		auto point = new std::vector<float>{52,10000};
		data->push_back(point);
	}

	auto c = new MineClus(data);
	c->setSeed(2);
	auto res = c->findKClusters(2);
	EXPECT_EQ(res.size(), 2);
	
	EXPECT_EQ(res.at(0).first->size(), 4);
	EXPECT_EQ(res.at(0).second->size(), 2);
	EXPECT_TRUE(res.at(0).second->at(0));
	EXPECT_TRUE(res.at(0).second->at(1));

	EXPECT_EQ(res.at(1).first->size(), 3);
	EXPECT_EQ(res.at(1).second->size(), 2);
	EXPECT_TRUE(res.at(1).second->at(0));
	EXPECT_TRUE(res.at(1).second->at(1));
}


TEST(testMineClus, _SLOW_testFind5Clusters){
	unsigned int dim = 20;
	auto data =  new std::vector<std::vector<float>*>();

	std::default_random_engine generator;
	generator.seed(1);
	std::normal_distribution<float> cluster1(5.0,2.0);
	std::normal_distribution<float> cluster2(50.0,2.0);
	std::normal_distribution<float> cluster3(500.0,2.0);
	std::normal_distribution<float> cluster4(-50.0,2.0);
	std::normal_distribution<float> cluster5(-500.0,2.0);
	std::uniform_int_distribution<> outlier(-10000,10000);
	
	
	for(int i = 0; i < 100; i++){
		auto point = new std::vector<float>;
		for(int j = 0; j < dim; j++){
			if(j%2 == 0){
				point->push_back(cluster1(generator));
			}else{
				point->push_back(outlier(generator));
			}
		}
		data->push_back(point);
	}

	for(int i = 0; i < 110; i++){
		auto point = new std::vector<float>;
		for(int j = 0; j < dim; j++){
			if(j%3 == 0){
				point->push_back(cluster2(generator));
			}else{
				point->push_back(outlier(generator));
			}
		}
		data->push_back(point);
	}

	for(int i = 0; i < 120; i++){
		auto point = new std::vector<float>;
		for(int j = 0; j < dim; j++){
			if(j%4 == 0){
				point->push_back(cluster3(generator));
			}else{
				point->push_back(outlier(generator));
			}
		}
		data->push_back(point);
	}

	for(int i = 0; i < 140; i++){
		auto point = new std::vector<float>;
		for(int j = 0; j < dim; j++){
			if(j%5 == 0){
				point->push_back(cluster4(generator));
			}else{
				point->push_back(outlier(generator));
			}
		}
		data->push_back(point);
	}

	for(int i = 0; i < 130; i++){
		auto point = new std::vector<float>;
		for(int j = 0; j < dim; j++){
			if((j+1)%5 == 0){
				point->push_back(cluster5(generator));
			}else{
				point->push_back(outlier(generator));
			}
		}
		data->push_back(point);
	}
	
	for(int i = 0; i < data->size()*0.1; i++){
		auto point = new std::vector<float>;
		for(int j = 0; j < dim; j++){
			point->push_back(outlier(generator));
		}
		data->push_back(point);
	}	

	auto c = new MineClus(data);
	c->setSeed(2);
	auto res = c->findKClusters(5);
	EXPECT_EQ(res.size(), 5);

	unsigned int a = 0;
	
	EXPECT_EQ(res.at(a).first->size(), 100);
	EXPECT_EQ(res.at(a).second->size(), 20);
	EXPECT_TRUE(res.at(a).second->at(0));
	EXPECT_FALSE(res.at(a).second->at(1));
	EXPECT_TRUE(res.at(a).second->at(2));
	EXPECT_FALSE(res.at(a).second->at(3));
	EXPECT_TRUE(res.at(a).second->at(4));
	EXPECT_FALSE(res.at(a).second->at(5));
	EXPECT_TRUE(res.at(a).second->at(6));
	EXPECT_FALSE(res.at(a).second->at(7));
	EXPECT_TRUE(res.at(a).second->at(8));
	EXPECT_FALSE(res.at(a).second->at(9));
	EXPECT_TRUE(res.at(a).second->at(10));
	EXPECT_FALSE(res.at(a).second->at(11));
	EXPECT_TRUE(res.at(a).second->at(12));
	EXPECT_FALSE(res.at(a).second->at(13));
	EXPECT_TRUE(res.at(a).second->at(14));
	EXPECT_FALSE(res.at(a).second->at(15));
	EXPECT_TRUE(res.at(a).second->at(16));
	EXPECT_FALSE(res.at(a).second->at(17));
	EXPECT_TRUE(res.at(a).second->at(18));
	EXPECT_FALSE(res.at(a).second->at(19));

	a++;

	EXPECT_EQ(res.at(a).first->size(), 110);
	EXPECT_EQ(res.at(a).second->size(), 20);
	EXPECT_TRUE(res.at(a).second->at(0));
	EXPECT_FALSE(res.at(a).second->at(1));
	EXPECT_FALSE(res.at(a).second->at(2));
	EXPECT_TRUE(res.at(a).second->at(3));
	EXPECT_FALSE(res.at(a).second->at(4));
	EXPECT_FALSE(res.at(a).second->at(5));
	EXPECT_TRUE(res.at(a).second->at(6));
	EXPECT_FALSE(res.at(a).second->at(7));
	EXPECT_FALSE(res.at(a).second->at(8));
	EXPECT_TRUE(res.at(a).second->at(9));
	EXPECT_FALSE(res.at(a).second->at(10));
	EXPECT_FALSE(res.at(a).second->at(11));
	EXPECT_TRUE(res.at(a).second->at(12));
	EXPECT_FALSE(res.at(a).second->at(13));
	EXPECT_FALSE(res.at(a).second->at(14));
	EXPECT_TRUE(res.at(a).second->at(15));
	EXPECT_FALSE(res.at(a).second->at(16));
	EXPECT_FALSE(res.at(a).second->at(17));
	EXPECT_TRUE(res.at(a).second->at(18));
	EXPECT_FALSE(res.at(a).second->at(19));


	a++;

	EXPECT_EQ(res.at(a).first->size(), 120);
	EXPECT_EQ(res.at(a).second->size(), 20);
	EXPECT_TRUE(res.at(a).second->at(0));
	EXPECT_FALSE(res.at(a).second->at(1));
	EXPECT_FALSE(res.at(a).second->at(2));
	EXPECT_FALSE(res.at(a).second->at(3));
	EXPECT_TRUE(res.at(a).second->at(4));
	EXPECT_FALSE(res.at(a).second->at(5));
	EXPECT_FALSE(res.at(a).second->at(6));
	EXPECT_FALSE(res.at(a).second->at(7));
	EXPECT_TRUE(res.at(a).second->at(8));
	EXPECT_FALSE(res.at(a).second->at(9));
	EXPECT_FALSE(res.at(a).second->at(10));
	EXPECT_FALSE(res.at(a).second->at(11));
	EXPECT_TRUE(res.at(a).second->at(12));
	EXPECT_FALSE(res.at(a).second->at(13));
	EXPECT_FALSE(res.at(a).second->at(14));
	EXPECT_FALSE(res.at(a).second->at(15));
	EXPECT_TRUE(res.at(a).second->at(16));
	EXPECT_FALSE(res.at(a).second->at(17));
	EXPECT_FALSE(res.at(a).second->at(18));
	EXPECT_FALSE(res.at(a).second->at(19));



	a++;

	EXPECT_EQ(res.at(a).first->size(), 140);
	EXPECT_EQ(res.at(a).second->size(), 20);
	EXPECT_TRUE(res.at(a).second->at(0));
	EXPECT_FALSE(res.at(a).second->at(1));
	EXPECT_FALSE(res.at(a).second->at(2));
	EXPECT_FALSE(res.at(a).second->at(3));
	EXPECT_FALSE(res.at(a).second->at(4));
	EXPECT_TRUE(res.at(a).second->at(5));
	EXPECT_FALSE(res.at(a).second->at(6));
	EXPECT_FALSE(res.at(a).second->at(7));
	EXPECT_FALSE(res.at(a).second->at(8));
	EXPECT_FALSE(res.at(a).second->at(9));
	EXPECT_TRUE(res.at(a).second->at(10));
	EXPECT_FALSE(res.at(a).second->at(11));
	EXPECT_FALSE(res.at(a).second->at(12));
	EXPECT_FALSE(res.at(a).second->at(13));
	EXPECT_FALSE(res.at(a).second->at(14));
	EXPECT_TRUE(res.at(a).second->at(15));
	EXPECT_FALSE(res.at(a).second->at(16));
	EXPECT_FALSE(res.at(a).second->at(17));
	EXPECT_FALSE(res.at(a).second->at(18));
	EXPECT_FALSE(res.at(a).second->at(19));


	a++;

	EXPECT_EQ(res.at(a).first->size(), 130);
	EXPECT_EQ(res.at(a).second->size(), 20);
	EXPECT_FALSE(res.at(a).second->at(0));
	EXPECT_FALSE(res.at(a).second->at(1));
	EXPECT_FALSE(res.at(a).second->at(2));
	EXPECT_FALSE(res.at(a).second->at(3));
	EXPECT_TRUE(res.at(a).second->at(4));
	EXPECT_FALSE(res.at(a).second->at(5));
	EXPECT_FALSE(res.at(a).second->at(6));
	EXPECT_FALSE(res.at(a).second->at(7));
	EXPECT_FALSE(res.at(a).second->at(8));
	EXPECT_TRUE(res.at(a).second->at(9));
	EXPECT_FALSE(res.at(a).second->at(10));
	EXPECT_FALSE(res.at(a).second->at(11));
	EXPECT_FALSE(res.at(a).second->at(12));
	EXPECT_FALSE(res.at(a).second->at(13));
	EXPECT_TRUE(res.at(a).second->at(14));
	EXPECT_FALSE(res.at(a).second->at(15));
	EXPECT_FALSE(res.at(a).second->at(16));
	EXPECT_FALSE(res.at(a).second->at(17));
	EXPECT_FALSE(res.at(a).second->at(18));
	EXPECT_TRUE(res.at(a).second->at(19));	
}
