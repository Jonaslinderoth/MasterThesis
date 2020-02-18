#include <gtest/gtest.h>
#include "../src/DOC_GPU/DOCGPU.h"
#include "../src/DOC_GPU/DOCGPU_Kernels.h"
#include <vector>
#include <math.h>
TEST(testFindClusterGPU, testSimple){
	std::vector<std::vector<float>*>* data = new std::vector<std::vector<float>*>;
	{
		auto point = new std::vector<float>{10,10};
		data->push_back(point);
	}
	{
		auto point = new std::vector<float>{0,10};
		data->push_back(point);
	}
	{
		auto point = new std::vector<float>{10,0};
		data->push_back(point);
	}
	{
		auto point = new std::vector<float>{0,0};
		data->push_back(point);
	}
	DOCGPU d = DOCGPU(data);
	std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> res = d.findCluster();
	EXPECT_EQ(res.first->size(),4);
	EXPECT_TRUE(res.first->at(0));
	EXPECT_TRUE(res.first->at(1));
	EXPECT_TRUE(res.first->at(2));
	EXPECT_TRUE(res.first->at(3));
	
	EXPECT_EQ(res.second->size(),2);
	EXPECT_TRUE(res.second->at(0));
	EXPECT_TRUE(res.second->at(1));
}


TEST(testFindClusterGPU, testSimple1){
	std::vector<std::vector<float>*>* data = new std::vector<std::vector<float>*>;
	{
		auto point = new std::vector<float>{100,100};
		data->push_back(point);
	}
	{
		auto point = new std::vector<float>{0,100};
		data->push_back(point);
	}
	{
		auto point = new std::vector<float>{100,00};
		data->push_back(point);
	}
	{
		auto point = new std::vector<float>{0,0};
		data->push_back(point);
	}
	DOCGPU d = DOCGPU(data);
	d.setSeed(1);
	std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> res = d.findCluster();
	EXPECT_EQ(res.first->size(),1);
	//std::cout << res.first->at(0)->at(0) << ", " << res.first->at(0)->at(1) << std::endl;
	//std::cout << res.first->at(1)->at(0) << ", " << res.first->at(1)->at(1) << std::endl;
	
	EXPECT_EQ(res.second->size(),2);
	EXPECT_TRUE(res.second->at(0));
	EXPECT_TRUE(res.second->at(1));
}


TEST(testFindClusterGPU, testSimple2){
	std::vector<std::vector<float>*>* data = new std::vector<std::vector<float>*>;
	{
		auto point = new std::vector<float>{100,100};
		data->push_back(point);
	}
	{
		auto point = new std::vector<float>{0,100};
		data->push_back(point);
	}
	{
		auto point = new std::vector<float>{100,00};
		data->push_back(point);
	}
	{
		auto point = new std::vector<float>{0,0};
		data->push_back(point);
	}
	DOCGPU d = DOCGPU(data);
	d.setSeed(1);
	d.setAlpha(0.1);
	d.setBeta(0.11);
	std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> res = d.findCluster();
	EXPECT_EQ(res.first->size(),1);
	//std::cout << res.first->at(0)->at(0) << ", " << res.first->at(0)->at(1) << std::endl;
	//std::cout << res.first->at(1)->at(0) << ", " << res.first->at(1)->at(1) << std::endl;

	
	EXPECT_EQ(res.second->size(),2);
	EXPECT_TRUE(res.second->at(0));
	EXPECT_TRUE(res.second->at(1));
}




TEST(testFindClusterGPU, testSimple3){
	std::vector<std::vector<float>*>* data = new std::vector<std::vector<float>*>;
	{
		auto point = new std::vector<float>{1,1000};
		data->push_back(point);
	}
	{
		auto point = new std::vector<float>{0,100};
		data->push_back(point);
	}
	{
		auto point = new std::vector<float>{1,-100};
		data->push_back(point);
	}
	{
		auto point = new std::vector<float>{0,-1000};
		data->push_back(point);
	}
	DOCGPU d = DOCGPU(data);
	d.setSeed(1);
	std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> res = d.findCluster();
	EXPECT_EQ(res.first->size(),4);

	
	EXPECT_EQ(res.second->size(),2);
	EXPECT_TRUE(res.second->at(0));
	EXPECT_FALSE(res.second->at(1));
}




TEST(testFindClusterGPU, testSimple4){
	std::vector<std::vector<float>*>* data = new std::vector<std::vector<float>*>;
	{
		auto point = new std::vector<float>{1,1000};
		data->push_back(point);
	}
	{
		auto point = new std::vector<float>{0,100};
		data->push_back(point);
	}
	{
		auto point = new std::vector<float>{1,-100};
		data->push_back(point);
	}
	{
		auto point = new std::vector<float>{0,-1000};
		data->push_back(point);
	}
	{
		auto point = new std::vector<float>{1000,-10000};
		data->push_back(point);
	}
	DOCGPU d = DOCGPU(data);
	d.setSeed(1);
	std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> res = d.findCluster();
	EXPECT_EQ(res.first->size(),4);

	
	EXPECT_EQ(res.second->size(),2);
	EXPECT_TRUE(res.second->at(0));
	EXPECT_FALSE(res.second->at(1));
}

TEST(testFindClusterGPU, testLarge){
	std::vector<std::vector<float>*>* data = new std::vector<std::vector<float>*>;
	for(float i = 0; i < 200; i++){
		for(float j = 9; j < 15; j++){
			std::vector<float>* point1 = new std::vector<float>{j,i};
			data->push_back(point1);

		}
	}
	DOCGPU d = DOCGPU(data);
	d.setSeed(1);
	d.setAlpha(0.1);
	d.setBeta(0.25);
	d.setWidth(5);
	auto res = d.findCluster();
	
	SUCCEED();
	EXPECT_TRUE(res.second->at(0));
	EXPECT_FALSE(res.second->at(1));

	EXPECT_EQ(res.first->size(), 1200);
}



TEST(testFindClusterGPU, testScore){
	int len = 1;
	unsigned int* cluster_sizes = (unsigned int*) malloc(len*sizeof(unsigned int));
	unsigned int* dim_sizes = (unsigned int*) malloc(len*sizeof(unsigned int));
	float* output = (float*) malloc(len*sizeof(float));
	cluster_sizes[0] = 1;
	dim_sizes[0] = 1;
	
	scoreHost(cluster_sizes, dim_sizes, output, len, 0.1, 0.25, 10);
	EXPECT_EQ(output[0], 4);
}


TEST(testFindClusterGPU, testScore2){
	int len = 2;
	unsigned int* cluster_sizes = (unsigned int*) malloc(len*sizeof(unsigned int));
	unsigned int* dim_sizes = (unsigned int*) malloc(len*sizeof(unsigned int));
	float* output = (float*) malloc(len*sizeof(float));
	cluster_sizes[0] = 1;
	dim_sizes[0] = 1;
	cluster_sizes[1] = 1;
	dim_sizes[1] = 1;
	
	scoreHost(cluster_sizes, dim_sizes, output, len,0.1, 0.25,10);
	EXPECT_EQ(output[0], 4);
	EXPECT_EQ(output[1], 4);
}



TEST(testFindClusterGPU, testScore3){



	int n = 10;
	int len = n*n;
	unsigned int* cluster_sizes = (unsigned int*) malloc(len*sizeof(unsigned int));
	unsigned int* dim_sizes = (unsigned int*) malloc(len*sizeof(unsigned int));
	float* output = (float*) malloc(len*sizeof(float));
	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			cluster_sizes[i*n+j] = i;
			dim_sizes[i*n+j] = j;
		}
	}
	
	float beta = 0.25;
	float alpha = 0.1;
	unsigned int points = len*12;
	scoreHost(cluster_sizes, dim_sizes, output, len, alpha, beta, points);


	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			int a = cluster_sizes[i*n+j];
			int b = dim_sizes[i*n+j];
			if(b < alpha*points){
 				EXPECT_EQ(0, output[i*n+j]);
			}else{
				EXPECT_FLOAT_EQ(a*pow(((float) 1/beta),b),output[i*n+j]);				
			}

		}
	}
	
	beta = 0.11;
	scoreHost(cluster_sizes, dim_sizes, output, len, alpha, beta, points);


	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			int a = cluster_sizes[i*n+j];
			int b = dim_sizes[i*n+j];
			if(b < alpha*points){
				EXPECT_EQ(0, output[i*n+j]);
			}else{
				EXPECT_FLOAT_EQ(a*pow(((float) 1/beta),b),output[i*n+j]);				
			}
		}
	}
	

}
