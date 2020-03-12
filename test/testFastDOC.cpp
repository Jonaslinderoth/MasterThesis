#include <gtest/gtest.h>
#include "../src/Fast_DOC/Fast_DOC.h"
#include "../src/Fast_DOCGPU/Fast_DOCGPU.h"


TEST(testFastDOC, testFindClusterSimple1){
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

	auto c = new Fast_DOC(data);
	c->setSeed(2);
	auto res = c->findCluster();
	
	EXPECT_EQ(c->getd0(), 2);
	EXPECT_EQ(res.first->size(), 4);
	EXPECT_EQ(res.second->size(), 2);
	EXPECT_TRUE(res.second->at(0));
	EXPECT_TRUE(res.second->at(1));
}



TEST(testFastDOC, testFindClusterSimple1GPU){
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

	auto c = new Fast_DOCGPU(data);
	c->setSeed(1);
	auto res = c->findCluster();

	EXPECT_EQ(res.first->size(), 4);
	EXPECT_EQ(res.second->size(), 2);
	EXPECT_TRUE(res.second->at(0));
	EXPECT_TRUE(res.second->at(1));
}



TEST(testFastDOC, testFindClusterSimple1GPUvsCPU){
	unsigned int dim = 10;
	unsigned int numPoints = 10;
	auto data =  new std::vector<std::vector<float>*>();
	
	
	for(int i = 0; i < numPoints; i++){
		auto point = new std::vector<float>;
		for(int j = 0; j < dim; j++){
			point->push_back(j);
		}
		data->push_back(point);
	}

	auto gpu = new Fast_DOCGPU(data);
	gpu->setSeed(1);
	auto cpu = new Fast_DOC(data);
	cpu->setSeed(1);

	EXPECT_EQ(cpu->getAlpha(), gpu->getAlpha());
	EXPECT_EQ(cpu->getBeta(), gpu->getBeta());
	EXPECT_EQ(cpu->getWidth(), gpu->getWidth());
	auto res_cpu = cpu->findCluster();
	auto res_gpu = gpu->findCluster();

	
	EXPECT_EQ(res_gpu.first->size(), res_cpu.first->size());
	EXPECT_EQ(res_gpu.second->size(), res_cpu.second->size());
	for(int i = 0; i < res_gpu.second->size(); i++){
		EXPECT_EQ(res_gpu.second->at(i), res_cpu.second->at(i));
	}

}


TEST(testFastDOC, testFindClusterSimple2GPUvsCPU){
	unsigned int dim = 10;
	unsigned int numPoints = 100;
	auto data =  new std::vector<std::vector<float>*>();
	
	
	for(int i = 0; i < numPoints; i++){
		auto point = new std::vector<float>;
		for(int j = 0; j < dim; j++){
			point->push_back(j);
		}
		data->push_back(point);
	}

	auto gpu = new Fast_DOCGPU(data);
	gpu->setSeed(1);
	auto cpu = new Fast_DOC(data);
	cpu->setSeed(1);

	EXPECT_EQ(cpu->getAlpha(), gpu->getAlpha());
	EXPECT_EQ(cpu->getBeta(), gpu->getBeta());
	EXPECT_EQ(cpu->getWidth(), gpu->getWidth());
	auto res_cpu = cpu->findCluster();
	auto res_gpu = gpu->findCluster();

	
	EXPECT_EQ(res_gpu.first->size(), res_cpu.first->size());
	EXPECT_EQ(res_gpu.second->size(), res_cpu.second->size());
	for(int i = 0; i < res_gpu.second->size(); i++){
		EXPECT_EQ(res_gpu.second->at(i), res_cpu.second->at(i));
	}

}



TEST(testFastDOC, testFindClusterSimple3GPUvsCPU){
	unsigned int dim = 100;
	unsigned int numPoints = 10;
	auto data =  new std::vector<std::vector<float>*>();
	
	
	for(int i = 0; i < numPoints; i++){
		auto point = new std::vector<float>;
		for(int j = 0; j < dim; j++){
			point->push_back(j);
		}
		data->push_back(point);
	}

	auto gpu = new Fast_DOCGPU(data);
	gpu->setSeed(1);
	auto cpu = new Fast_DOC(data);
	cpu->setSeed(1);

	EXPECT_EQ(cpu->getAlpha(), gpu->getAlpha());
	EXPECT_EQ(cpu->getBeta(), gpu->getBeta());
	EXPECT_EQ(cpu->getWidth(), gpu->getWidth());
	auto res_cpu = cpu->findCluster();
	auto res_gpu = gpu->findCluster();

	
	EXPECT_EQ(res_gpu.first->size(), res_cpu.first->size());
	EXPECT_EQ(res_gpu.second->size(), res_cpu.second->size());
	for(int i = 0; i < res_gpu.second->size(); i++){
		EXPECT_EQ(res_gpu.second->at(i), res_cpu.second->at(i));
	}

}



TEST(testFastDOC, testFindClusterGPUvsCPU_1){
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

	auto gpu = new Fast_DOCGPU(data);
	gpu->setSeed(1);
	auto cpu = new Fast_DOC(data);
	cpu->setSeed(1);

	EXPECT_EQ(cpu->getAlpha(), gpu->getAlpha());
	EXPECT_EQ(cpu->getBeta(), gpu->getBeta());
	EXPECT_EQ(cpu->getWidth(), gpu->getWidth());
	auto res_cpu = cpu->findCluster();
	auto res_gpu = gpu->findCluster();
	
	EXPECT_EQ(res_gpu.first->size(), res_cpu.first->size());
	EXPECT_EQ(res_gpu.second->size(), res_cpu.second->size());
	for(int i = 0; i < res_gpu.second->size(); i++){
		EXPECT_EQ(res_gpu.second->at(i), (i%2 == 0)) << "gpu: i: " << i << ", (i%2 == 0) " << (i%2 == 0);
		EXPECT_EQ(res_cpu.second->at(i), (i%2 == 0)) << "cpu: i: " << i << ", (i%2 == 0) " << (i%2 == 0);
		EXPECT_EQ(res_gpu.second->at(i), res_cpu.second->at(i));
	}
	for(int i = 0; i < res_gpu.first->size(); i++){
		for(int j = 0; j < res_gpu.first->at(i)->size(); j++)
			EXPECT_EQ(res_gpu.first->at(i)->at(j), res_cpu.first->at(i)->at(j));
	}

}


TEST(testFastDOC, testFindClusterGPUvsCPU_1_1){
	unsigned int dim = 10;
	unsigned int numPoints = 10;
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

	auto gpu = new Fast_DOCGPU(data);
	gpu->setSeed(3);
	auto cpu = new Fast_DOC(data);
	cpu->setSeed(3);

	EXPECT_EQ(cpu->getAlpha(), gpu->getAlpha());
	EXPECT_EQ(cpu->getBeta(), gpu->getBeta());
	EXPECT_EQ(cpu->getWidth(), gpu->getWidth());
	auto res_cpu = cpu->findCluster();
	auto res_gpu = gpu->findCluster();
	
	EXPECT_EQ(res_gpu.first->size(), res_cpu.first->size());
	EXPECT_EQ(res_gpu.second->size(), res_cpu.second->size());
	for(int i = 0; i < res_gpu.second->size(); i++){
		EXPECT_EQ(res_gpu.second->at(i), (i%2 == 0)) << "gpu: i: " << i << ", (i%2 == 0) " << (i%2 == 0);
		EXPECT_EQ(res_cpu.second->at(i), (i%2 == 0)) << "cpu: i: " << i << ", (i%2 == 0) " << (i%2 == 0);
		EXPECT_EQ(res_gpu.second->at(i), res_cpu.second->at(i));
	}
	for(int i = 0; i < res_gpu.first->size(); i++){
		for(int j = 0; j < res_gpu.first->at(i)->size(); j++)
			EXPECT_EQ(res_gpu.first->at(i)->at(j), res_cpu.first->at(i)->at(j));
	}

}

TEST(testFastDOC, testFindClusterGPUvsCPU_2){
	unsigned int dim = 100;
	unsigned int numPoints = 100;
	auto data =  new std::vector<std::vector<float>*>();

	std::default_random_engine generator;
	generator.seed(1);
	std::normal_distribution<float> cluster(5.0,2.0);
	std::uniform_int_distribution<> outlier(-10000,10000);
	
	
	for(int i = 0; i < numPoints*0.9; i++){ // 10% outlier
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

	for(int i = data->size(); i < numPoints; i++){
		auto point = new std::vector<float>;
		for(int j = 0; j < dim; j++){
			point->push_back(outlier(generator));
		}
		data->push_back(point);
	}

	auto gpu = new Fast_DOCGPU(data);
	gpu->setSeed(2);
	auto cpu = new Fast_DOC(data);
	cpu->setSeed(2);

	EXPECT_EQ(cpu->getAlpha(), gpu->getAlpha());
	EXPECT_EQ(cpu->getBeta(), gpu->getBeta());
	EXPECT_EQ(cpu->getWidth(), gpu->getWidth());
	auto res_cpu = cpu->findCluster();
	auto res_gpu = gpu->findCluster();
	
	EXPECT_EQ(res_gpu.first->size(), res_cpu.first->size());
	EXPECT_EQ(res_gpu.second->size(), res_cpu.second->size());
	for(int i = 0; i < res_gpu.second->size(); i++){
		//EXPECT_EQ(res_gpu.second->at(i), (i%2 == 0)) << "gpu: i: " << i << ", (i%2 == 0) " << (i%2 == 0);
		//EXPECT_EQ(res_cpu.second->at(i), (i%2 == 0)) << "cpu: i: " << i << ", (i%2 == 0) " << (i%2 == 0);
		EXPECT_EQ(res_gpu.second->at(i), res_cpu.second->at(i));
	}
	for(int i = 0; i < res_gpu.first->size(); i++){
		for(int j = 0; j < res_gpu.first->at(i)->size(); j++)
			EXPECT_EQ(res_gpu.first->at(i)->at(j), res_cpu.first->at(i)->at(j));
	}

}


TEST(testFastDOC, testFind2ClusterGPU){
	unsigned int dim = 10;
	unsigned int numPoints = 100;
	auto data =  new std::vector<std::vector<float>*>();

	std::default_random_engine generator;
	generator.seed(1);
	std::normal_distribution<float> cluster1(5.0, 2.0);
	std::normal_distribution<float> cluster2(500.0,2.0);
	std::uniform_int_distribution<> outlier(-10000,10000);
	
	
	for(int i = 0; i < numPoints/2; i++){
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


	for(int i = data->size(); i < numPoints; i++){
		auto point = new std::vector<float>;
		for(int j = 0; j < dim; j++){
			if(j%3 == 0){
				point->push_back(cluster1(generator));				
			}else{
				point->push_back(outlier(generator));
			}
		}
		data->push_back(point);
	}
	

	auto gpu = new Fast_DOCGPU(data);
	gpu->setSeed(1);

	auto res_gpu = gpu->findKClusters(2);

	EXPECT_EQ(res_gpu.size(), 2);

	EXPECT_EQ(res_gpu.at(0).first->size(), 50);
	EXPECT_EQ(res_gpu.at(0).second->size(), dim);
		
	EXPECT_EQ(res_gpu.at(1).first->size(), 50);
	EXPECT_EQ(res_gpu.at(1).second->size(), dim);
}


TEST(testFastDOC, testFind2ClusterCPU){
	unsigned int dim = 10;
	unsigned int numPoints = 100;
	auto data =  new std::vector<std::vector<float>*>();

	std::default_random_engine generator;
	generator.seed(1);
	std::normal_distribution<float> cluster1(5.0, 2.0);
	std::normal_distribution<float> cluster2(500.0,2.0);
	std::uniform_int_distribution<> outlier(-10000,10000);
	
	
	for(int i = 0; i < numPoints/2; i++){
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


	for(int i = data->size(); i < numPoints; i++){
		auto point = new std::vector<float>;
		for(int j = 0; j < dim; j++){
			if(j%3 == 0){
				point->push_back(cluster1(generator));				
			}else{
				point->push_back(outlier(generator));
			}
		}
		data->push_back(point);
	}
	

	auto cpu = new Fast_DOC(data);
	cpu->setSeed(1);


	auto res_cpu = cpu->findKClusters(2);

	EXPECT_EQ(res_cpu.size(), 2);

	EXPECT_EQ(res_cpu.at(0).first->size(), 50);
	EXPECT_EQ(res_cpu.at(0).second->size(), dim);

	for(int i = 0; i < dim; i++){
		EXPECT_EQ(res_cpu.at(0).second->at(i), (i%2==0));
	}

	
	EXPECT_EQ(res_cpu.at(1).first->size(), 50);
	EXPECT_EQ(res_cpu.at(1).second->size(), dim);


	for(int i = 0; i < dim; i++){
		EXPECT_EQ(res_cpu.at(1).second->at(i), (i%3==0));
	}
}




TEST(testFastDOC, testFind2ClusterGPUvsCPU){
	unsigned int dim = 10;
	unsigned int numPoints = 100;
	auto data =  new std::vector<std::vector<float>*>();

	std::default_random_engine generator;
	generator.seed(1);
	std::normal_distribution<float> cluster1(5.0, 2.0);
	std::normal_distribution<float> cluster2(500.0,2.0);
	std::uniform_int_distribution<> outlier(-10000,10000);
	
	
	for(int i = 0; i < numPoints/2; i++){
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


	for(int i = data->size(); i < numPoints; i++){
		auto point = new std::vector<float>;
		for(int j = 0; j < dim; j++){
			if(j%3 == 0){
				point->push_back(cluster1(generator));				
			}else{
				point->push_back(outlier(generator));
			}
		}
		data->push_back(point);
	}
	


	auto gpu = new Fast_DOCGPU(data);
	gpu->setSeed(1);
	auto cpu = new Fast_DOC(data);
	cpu->setSeed(1);

	EXPECT_EQ(cpu->getAlpha(), gpu->getAlpha());
	EXPECT_EQ(cpu->getBeta(), gpu->getBeta());
	EXPECT_EQ(cpu->getWidth(), gpu->getWidth());

	auto res_gpu = gpu->findKClusters(2);
	auto res_cpu = cpu->findKClusters(2);

	EXPECT_EQ(res_gpu.size(), res_cpu.size());
	EXPECT_EQ(res_gpu.size(), 2);

	EXPECT_EQ(res_gpu.at(0).first->size(), res_cpu.at(0).first->size());
	EXPECT_EQ(res_gpu.at(0).second->size(), res_cpu.at(0).second->size());
	EXPECT_EQ(res_gpu.at(0).second->size(), dim);
	
		
	EXPECT_EQ(res_gpu.at(1).first->size(), res_cpu.at(1).first->size());
	EXPECT_EQ(res_gpu.at(1).second->size(), res_cpu.at(1).second->size());
	EXPECT_EQ(res_gpu.at(1).second->size(), dim);
}
