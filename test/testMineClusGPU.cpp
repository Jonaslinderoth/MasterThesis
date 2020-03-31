#include <gtest/gtest.h>
#include "testingTools.h"
#include <random>
#include "../src/MineClusGPU/MineClusGPU.h"
#include <vector>



TEST(testMineClusGPU, testSetup){
	auto data = new std::vector<std::vector<float>*>;
	auto point1 = new std::vector<float>({1,2});
	data->push_back(point1);

	for(int i = 0; i < 20; i++){
		auto point = new std::vector<float>({1000,20000});
		data->push_back(point);
	}

	for(int i = 0; i < 5; i++){
		auto point = new std::vector<float>({1,2});
		data->push_back(point);
	}

	
	auto c = MineClusGPU(data);
	c.setSeed(1);

	auto res = c.findKClusters(1);

	EXPECT_EQ(res.size(), 1);
	EXPECT_EQ(res.at(0).second->at(0), 1);
	EXPECT_EQ(res.at(0).second->at(1), 1);

	EXPECT_EQ(res.at(0).first->size(), 20);
	for(int i = 0; i < 20; i++){
		EXPECT_EQ(res.at(0).first->at(i)->at(0), 1000);
		EXPECT_EQ(res.at(0).first->at(i)->at(1), 20000);		
	}


	
}


TEST(testMineClusGPU, testSetup2){
	auto data = new std::vector<std::vector<float>*>;
	auto point1 = new std::vector<float>({1,2});
	data->push_back(point1);

	for(int i = 0; i < 20; i++){
		auto point = new std::vector<float>({1000,2});
		data->push_back(point);
	}

	
	auto c = MineClusGPU(data);
	c.setSeed(1);

	auto res = c.findKClusters(1);

	EXPECT_EQ(res.size(), 1);
	EXPECT_EQ(res.at(0).second->size(), 2);
	EXPECT_EQ(res.at(0).second->at(0), 1);
	EXPECT_EQ(res.at(0).second->at(1), 1);
	
	EXPECT_EQ(res.at(0).first->size(), 20);
	for(int i = 0; i < 20; i++){
		EXPECT_EQ(res.at(0).first->at(i)->at(0), 1000);
		EXPECT_EQ(res.at(0).first->at(i)->at(1), 2);		
	}
	
}


TEST(testMineClusGPU, testSetup3){
	auto data = new std::vector<std::vector<float>*>;
	auto point1 = new std::vector<float>({1,2});
	data->push_back(point1);

	for(int i = 0; i < 20; i++){
		auto point = new std::vector<float>({1,2});
		data->push_back(point);
	}

	
	auto c = MineClusGPU(data);
	c.setSeed(1);

	auto res = c.findKClusters(1);

	EXPECT_EQ(res.size(), 1);
	EXPECT_EQ(res.at(0).second->at(0), 1);
	EXPECT_EQ(res.at(0).second->at(1), 1);
	EXPECT_EQ(res.at(0).first->size(), 21);
	
}



TEST(testMineClusGPU, test3dims_1){
	auto data = new std::vector<std::vector<float>*>;
	auto point1 = new std::vector<float>({1,2,3});
	data->push_back(point1);

	for(int i = 0; i < 20; i++){
		auto point = new std::vector<float>({1,2,3});
		data->push_back(point);
	}

	
	auto c = MineClusGPU(data);
	c.setSeed(1);

	auto res = c.findKClusters(1);

	EXPECT_EQ(res.size(), 1);
	EXPECT_EQ(res.at(0).second->size(), 3);
	EXPECT_EQ(res.at(0).second->at(0), 1);
	EXPECT_EQ(res.at(0).second->at(1), 1);
	EXPECT_EQ(res.at(0).second->at(2), 1);
}

TEST(testMineClusGPU, test3dims_2){
	auto data = new std::vector<std::vector<float>*>;
	auto point1 = new std::vector<float>({1,2,3});
	data->push_back(point1);

	for(int i = 0; i < 20; i++){
		auto point = new std::vector<float>({1,2,99999});
		data->push_back(point);
	}
	for(int i = 0; i < 20; i++){
		auto point = new std::vector<float>({1,2,9});
		data->push_back(point);
	}
	
	auto c = MineClusGPU(data);
	c.setSeed(1);

	auto res = c.findKClusters(1);

	EXPECT_EQ(res.size(), 1);
	EXPECT_EQ(res.at(0).second->size(), 3);
	EXPECT_EQ(res.at(0).second->at(0), 1);
	EXPECT_EQ(res.at(0).second->at(1), 1);
	EXPECT_EQ(res.at(0).second->at(2), 1);

	EXPECT_EQ(res.at(0).first->size(), 21);	

	
}

TEST(testMineClusGPU, test10Dims){
	auto data = new std::vector<std::vector<float>*>;
	auto point1 = new std::vector<float>({1,2,3,4,5,6,7,8,9,10});
	data->push_back(point1);

	for(int i = 0; i < 20; i++){
		auto point = new std::vector<float>({1,2,3,4,5,6,7,8,999,10});
		data->push_back(point);
	}

	for(int i = 0; i < 145; i++){
		auto point = new std::vector<float>({1,2,3,999+20*i,5,6,7,8,999,10});
		data->push_back(point);
	}
	
	auto c = MineClusGPU(data);
	c.setSeed(1);


	auto res = c.findKClusters(1);

	EXPECT_EQ(res.size(), 1);
	EXPECT_EQ(res.at(0).first->size(), 165);
	EXPECT_EQ(res.at(0).second->size(), 10);

	
	EXPECT_EQ(res.at(0).second->at(0), 1);
	EXPECT_EQ(res.at(0).second->at(1), 1);
	EXPECT_EQ(res.at(0).second->at(2), 1);
	EXPECT_EQ(res.at(0).second->at(3), 0);
	EXPECT_EQ(res.at(0).second->at(4), 1);
	EXPECT_EQ(res.at(0).second->at(5), 1);
	EXPECT_EQ(res.at(0).second->at(6), 1);
	EXPECT_EQ(res.at(0).second->at(7), 1);
	EXPECT_EQ(res.at(0).second->at(8), 1);
	EXPECT_EQ(res.at(0).second->at(9), 1);





	for(int i = 0-1; i < 20-1; i ++){
		EXPECT_EQ(res.at(0).first->at(i+1)->at(0), 1);
		EXPECT_EQ(res.at(0).first->at(i+1)->at(1), 2);
		EXPECT_EQ(res.at(0).first->at(i+1)->at(2), 3);
		EXPECT_EQ(res.at(0).first->at(i+1)->at(3), 4);
		EXPECT_EQ(res.at(0).first->at(i+1)->at(4), 5);
		EXPECT_EQ(res.at(0).first->at(i+1)->at(5), 6);
		EXPECT_EQ(res.at(0).first->at(i+1)->at(6), 7);
		EXPECT_EQ(res.at(0).first->at(i+1)->at(7), 8);
		EXPECT_EQ(res.at(0).first->at(i+1)->at(8), 999);
		EXPECT_EQ(res.at(0).first->at(i+1)->at(9), 10);		
	}

	for(int i = 20-1; i < 20+145-1; i ++){
		EXPECT_EQ(res.at(0).first->at(i+1)->at(0), 1);
		EXPECT_EQ(res.at(0).first->at(i+1)->at(1), 2);
		EXPECT_EQ(res.at(0).first->at(i+1)->at(2), 3);
		EXPECT_EQ(res.at(0).first->at(i+1)->at(3), 999+(20*(i-19)));
		EXPECT_EQ(res.at(0).first->at(i+1)->at(4), 5);
		EXPECT_EQ(res.at(0).first->at(i+1)->at(5), 6);
		EXPECT_EQ(res.at(0).first->at(i+1)->at(6), 7);
		EXPECT_EQ(res.at(0).first->at(i+1)->at(7), 8);
		EXPECT_EQ(res.at(0).first->at(i+1)->at(8), 999);
		EXPECT_EQ(res.at(0).first->at(i+1)->at(9), 10);		
	}
}


TEST(testMineClusGPU, _SLOW_test66Dims){
	std::default_random_engine gen;
	gen.seed(0);
	std::normal_distribution<double> cluster1(100.0,2.0);
	std::normal_distribution<double> cluster2(1000.0,2.0);
	std::normal_distribution<double> noise(100.0,200000.0);
	auto data = new std::vector<std::vector<float>*>;
	{
		for(int i = 0; i < 300; i++){
			auto point = new std::vector<float>;
			for(int j = 0; j < 66; j++){
				if(j % 6 == 0){
					point->push_back(cluster1(gen));					
				}else{
					point->push_back(noise(gen));					
				}

			}
			data->push_back(point);
		}
	}


	{
		for(int i = 0; i < 200; i++){
			auto point = new std::vector<float>;
			for(int j = 0; j < 66; j++){
				if(j % 8 == 0){
					point->push_back(cluster2(gen));					
				}else{
					point->push_back(noise(gen));					
				}

			}
			data->push_back(point);
		}
	}

	auto c = MineClusGPU(data);
	c.setSeed(2);


	auto res = c.findKClusters(1);

	EXPECT_EQ(res.size(),1);
	EXPECT_EQ(res.at(0).first->size(),300);
	EXPECT_EQ(res.at(0).second->size(),66);


	for(int j = 0; j < 66; j++){
		if(j % 6 == 0){
			EXPECT_EQ(res.at(0).second->at(j), 1) << "j: " << j;					
		}else{
			EXPECT_EQ(res.at(0).second->at(j), 0) << "j: " << j;
		}
	}
}

TEST(testMineClusGPU, DISABLED_SLOW_test32Dims){
	std::default_random_engine gen;
	gen.seed(0);
	std::normal_distribution<float> cluster1(100.0,2.0);
	std::normal_distribution<float> cluster2(1000.0,2.0);
	std::normal_distribution<float> noise(100.0,200000.0);
	auto data = new std::vector<std::vector<float>*>;
	{
		for(int i = 0; i < 300; i++){
			auto point = new std::vector<float>;
			for(int j = 0; j < 32; j++){
				if(j % 8 == 0){
					point->push_back(cluster1(gen));					
				}else{
					point->push_back(noise(gen));					
				}

			}
			data->push_back(point);
		}
	}


	{
		for(int i = 0; i < 100; i++){
			auto point = new std::vector<float>;
			for(int j = 0; j < 32; j++){
				if(j % 10 == 0){
					point->push_back(cluster2(gen));					
				}else{
					point->push_back(noise(gen));					
				}

			}
			data->push_back(point);
		}
	}

	auto c = MineClusGPU(data);
	c.setSeed(2);


	auto res = c.findKClusters(1);

	EXPECT_EQ(res.size(),1);
	EXPECT_EQ(res.at(0).first->size(),300);
	EXPECT_EQ(res.at(0).second->size(),32);


	for(int j = 0; j < 32; j++){
		if(j % 8 == 0){
			EXPECT_EQ(res.at(0).second->at(j), 1) << "j: " << j;					
		}else{
			EXPECT_EQ(res.at(0).second->at(j), 0) << "j: " << j;
		}
	}
}


TEST(testMineClusGPU, _SLOW_test30Dims){
	std::default_random_engine gen;
	gen.seed(0);
	std::normal_distribution<float> cluster1(100.0,2.0);
	std::normal_distribution<float> cluster2(1000.0,2.0);
	std::normal_distribution<float> noise(100.0,200000.0);
	auto data = new std::vector<std::vector<float>*>;
	{
		for(int i = 0; i < 300; i++){
			auto point = new std::vector<float>;
			for(int j = 0; j < 30; j++){
				if(j % 5 == 0){
					point->push_back(cluster1(gen));					
				}else{
					point->push_back(noise(gen));					
				}

			}
			data->push_back(point);
		}
	}


	{
		for(int i = 0; i < 100; i++){
			auto point = new std::vector<float>;
			for(int j = 0; j < 30; j++){
				if(j % 10 == 0){
					point->push_back(cluster2(gen));					
				}else{
					point->push_back(noise(gen));					
				}

			}
			data->push_back(point);
		}
	}

	auto c = MineClusGPU(data);
	c.setSeed(2);


	auto res = c.findKClusters(1);

	EXPECT_EQ(res.size(),1);
	EXPECT_EQ(res.at(0).first->size(),300);
	EXPECT_EQ(res.at(0).second->size(),30);

	int count = 0;
	for(int j = 0; j < 30; j++){
		if(j % 5 == 0){
			EXPECT_EQ(res.at(0).second->at(j), 1) << "j: " << j;
			count++;
		}else{
			EXPECT_EQ(res.at(0).second->at(j), 0) << "j: " << j;
		}
	}
	EXPECT_EQ(count, 6);
}
