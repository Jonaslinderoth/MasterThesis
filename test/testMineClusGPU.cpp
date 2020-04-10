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

	EXPECT_EQ(res.size(), 2);
	EXPECT_EQ(res.at(0).second->at(0), 1);
	EXPECT_EQ(res.at(0).second->at(1), 1);

	EXPECT_EQ(res.at(0).first->size(), 20);
	for(int i = 0; i < 20; i++){
		EXPECT_EQ(res.at(0).first->at(i)->at(0), 1000);
		EXPECT_EQ(res.at(0).first->at(i)->at(1), 20000);		
	}


	EXPECT_EQ(res.at(1).second->at(0), 1);
	EXPECT_EQ(res.at(1).second->at(1), 1);

	EXPECT_EQ(res.at(1).first->size(), 6);
	for(int i = 0; i < 6; i++){
		EXPECT_EQ(res.at(1).first->at(i)->at(0), 1);
		EXPECT_EQ(res.at(1).first->at(i)->at(1), 2);		
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



TEST(testMineClusGPU, testNoise){
	std::default_random_engine gen;
	gen.seed(100);
	std::normal_distribution<double> noise(100.0,200000.0);
	auto data = new std::vector<std::vector<float>*>;
	{
		for(int i = 0; i < 3000; i++){
			auto point = new std::vector<float>;
			for(int j = 0; j < 66; j++){
					point->push_back(noise(gen));					
			}
			data->push_back(point);
		}
	}


	auto c = MineClusGPU(data);
	c.setSeed(1);

	auto res = c.findKClusters(10);

	EXPECT_EQ(res.size(), 1);
	EXPECT_EQ(res.at(0).first->size(),3000);
	for(int i = 0; i < 66; i++){
		EXPECT_EQ(res.at(0).second->at(i), 0);
	}
	
	
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

	EXPECT_EQ(res.size(), 2);
	EXPECT_EQ(res.at(0).second->size(), 3);
	EXPECT_EQ(res.at(0).second->at(0), 1);
	EXPECT_EQ(res.at(0).second->at(1), 1);
	EXPECT_EQ(res.at(0).second->at(2), 1);

	EXPECT_EQ(res.at(0).first->size(), 21);
	EXPECT_EQ(res.at(0).first->at(0)->at(0), 1);
	EXPECT_EQ(res.at(0).first->at(0)->at(1), 2);
	EXPECT_EQ(res.at(0).first->at(0)->at(2), 3);		
	for(int i = 1; i < 21; i++){
		EXPECT_EQ(res.at(0).first->at(i)->at(0), 1);
		EXPECT_EQ(res.at(0).first->at(i)->at(1), 2);
		EXPECT_EQ(res.at(0).first->at(i)->at(2), 9);		
	}



	EXPECT_EQ(res.at(1).second->size(), 3);
	EXPECT_EQ(res.at(1).second->at(0), 1);
	EXPECT_EQ(res.at(1).second->at(1), 1);
	EXPECT_EQ(res.at(1).second->at(2), 1);

	EXPECT_EQ(res.at(1).first->size(), 20);
	for(int i = 0; i < 20; i++){
		EXPECT_EQ(res.at(1).first->at(i)->at(0), 1);
		EXPECT_EQ(res.at(1).first->at(i)->at(1), 2);
		EXPECT_EQ(res.at(1).first->at(i)->at(2), 99999);		
	}
	


	
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
	gen.seed(10);
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

	EXPECT_EQ(res.size(),2);
	EXPECT_EQ(res.at(0).first->size(),300);
	EXPECT_EQ(res.at(0).second->size(),66);


	for(int j = 0; j < 66; j++){
		if(j % 6 == 0){
			EXPECT_EQ(res.at(0).second->at(j), 1) << "j: " << j;					
		}else{
			EXPECT_EQ(res.at(0).second->at(j), 0) << "j: " << j;
		}
	}

	EXPECT_EQ(res.at(1).first->size(),200);
	EXPECT_EQ(res.at(1).second->size(),66);
	
}


TEST(testMineClusGPU, _SLOW_test66Dims_6clusters){
	std::default_random_engine gen;
	gen.seed(100);
	std::normal_distribution<double> cluster1(100.0,2.0);
	std::normal_distribution<double> cluster2(1000.0,2.0);
	std::normal_distribution<double> cluster3(10000.0,2.0);
	std::normal_distribution<double> cluster4(100000.0,2.0);
	std::normal_distribution<double> cluster5(1000000.0,2.0);
	std::normal_distribution<double> cluster6(10000000.0,2.0);
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

	{
		for(int i = 0; i < 190; i++){
			auto point = new std::vector<float>;
			for(int j = 0; j < 66; j++){
				if(j % 8 == 0){
					point->push_back(cluster3(gen));					
				}else{
					point->push_back(noise(gen));					
				}

			}
			data->push_back(point);
		}
	}
	{
		for(int i = 0; i < 180; i++){
			auto point = new std::vector<float>;
			for(int j = 0; j < 66; j++){
				if(j % 8 == 0){
					point->push_back(cluster4(gen));					
				}else{
					point->push_back(noise(gen));					
				}

			}
			data->push_back(point);
		}
	}

	{
		for(int i = 0; i < 170; i++){
			auto point = new std::vector<float>;
			for(int j = 0; j < 66; j++){
				if(j % 8 == 0){
					point->push_back(cluster5(gen));					
				}else{
					point->push_back(noise(gen));					
				}

			}
			data->push_back(point);
		}
	}
	auto c = MineClusGPU(data);
	c.setSeed(2);


	auto res = c.findKClusters(5);

	EXPECT_EQ(res.size(),5);
	EXPECT_EQ(res.at(0).first->size(),300);
	EXPECT_EQ(res.at(0).second->size(),66);


	for(int j = 0; j < 66; j++){
		if(j % 6 == 0){
			EXPECT_EQ(res.at(0).second->at(j), 1) << "j: " << j;					
		}else{
			EXPECT_EQ(res.at(0).second->at(j), 0) << "j: " << j;
		}
	}

	EXPECT_EQ(res.at(1).first->size(),200);
	EXPECT_EQ(res.at(1).second->size(),66);

	EXPECT_EQ(res.at(2).first->size(),190);
	EXPECT_EQ(res.at(2).second->size(),66);

	EXPECT_EQ(res.at(3).first->size(),180);
	EXPECT_EQ(res.at(3).second->size(),66);

	EXPECT_EQ(res.at(4).first->size(),170);
	EXPECT_EQ(res.at(4).second->size(),66);

}


TEST(testMineClusGPU, _SLOW_test32Dims){
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
				if(j % 11 == 0){
					point->push_back(cluster2(gen));					
				}else{
					point->push_back(noise(gen));					
				}

			}
			data->push_back(point);
		}
	}

	EXPECT_EQ(data->size(), 400);
	EXPECT_EQ(data->at(0)->size(), 32);

	auto c = MineClusGPU(data);
	c.setSeed(2);


	auto res = c.findKClusters(1);

	EXPECT_EQ(res.size(),2);
	EXPECT_EQ(res.at(0).first->size(),300);
	EXPECT_EQ(res.at(0).second->size(),32);

	// for(int j = 0; j < 32; j++){
	// 	std::cout << res.at(0).second->at(j);
	// }
	// std::cout << std::endl;
	
	for(int j = 0; j < 32; j++){
		if(j % 8 == 0){
			EXPECT_EQ(res.at(0).second->at(j), 1) << "j: " << j;					
		}else{
			EXPECT_EQ(res.at(0).second->at(j), 0) << "j: " << j;
		}
	}

	EXPECT_EQ(res.at(1).first->size(),100);
	EXPECT_EQ(res.at(1).second->size(),32);

	for(int j = 0; j < 32; j++){
		if(j % 11 == 0){
			EXPECT_EQ(res.at(1).second->at(j), 1) << "j: " << j;					
		}else{
			EXPECT_EQ(res.at(1).second->at(j), 0) << "j: " << j;
		}
	}
}


TEST(testMineClusGPU, _SLOW_test100Dims){
	std::default_random_engine gen;
	gen.seed(0);
	std::normal_distribution<float> cluster1(100.0,2.0);
	std::normal_distribution<float> cluster2(1000.0,2.0);
	std::normal_distribution<float> noise(100.0,200000.0);
	auto data = new std::vector<std::vector<float>*>;
	{
		for(int i = 0; i < 300; i++){
			auto point = new std::vector<float>;
			for(int j = 0; j < 100; j++){
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
			for(int j = 0; j < 100; j++){
				if(j % 11 == 0){
					point->push_back(cluster2(gen));					
				}else{
					point->push_back(noise(gen));					
				}

			}
			data->push_back(point);
		}
	}

	EXPECT_EQ(data->size(), 400);
	EXPECT_EQ(data->at(0)->size(), 100);

	auto c = MineClusGPU(data);
	c.setSeed(2);


	auto res = c.findKClusters(1);

	EXPECT_EQ(res.size(),2);
	EXPECT_EQ(res.at(0).first->size(),300);
	EXPECT_EQ(res.at(0).second->size(),100);

	// for(int j = 0; j < 32; j++){
	// 	std::cout << res.at(0).second->at(j);
	// }
	// std::cout << std::endl;
	
	for(int j = 0; j < 100; j++){
		if(j % 8 == 0){
			EXPECT_EQ(res.at(0).second->at(j), 1) << "j: " << j;					
		}else{
			EXPECT_EQ(res.at(0).second->at(j), 0) << "j: " << j;
		}
	}

	EXPECT_EQ(res.at(1).first->size(),100);
	EXPECT_EQ(res.at(1).second->size(),100);

	for(int j = 0; j < 100; j++){
		if(j % 11 == 0){
			EXPECT_EQ(res.at(1).second->at(j), 1) << "j: " << j;					
		}else{
			EXPECT_EQ(res.at(1).second->at(j), 0) << "j: " << j;
		}
	}
}



TEST(testMineClusGPU, _SLOW_test40Dims){
	std::default_random_engine gen;
	gen.seed(1);
	unsigned int dim = 40;
	std::normal_distribution<float> cluster1(100.0,2.0);
	std::normal_distribution<float> cluster2(1000.0,2.0);
	std::normal_distribution<float> noise(100.0,200000.0);
	auto data = new std::vector<std::vector<float>*>;
	{
		for(int i = 0; i < 300; i++){
			auto point = new std::vector<float>;
			for(int j = 0; j < dim; j++){
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
			for(int j = 0; j < dim; j++){
				if(j % 11 == 0){
					point->push_back(cluster2(gen));					
				}else{
					point->push_back(noise(gen));					
				}

			}
			data->push_back(point);
		}
	}

	EXPECT_EQ(data->size(), 400);
	EXPECT_EQ(data->at(0)->size(), dim);

	auto c = MineClusGPU(data);
	c.setSeed(1);


	auto res = c.findKClusters(1);

	EXPECT_EQ(res.size(),2);
	EXPECT_EQ(res.at(0).first->size(),300);
	EXPECT_EQ(res.at(0).second->size(),dim);

	for(int j = 0; j < dim; j++){
		if(j % 8 == 0){
			EXPECT_EQ(res.at(0).second->at(j), 1) << "j: " << j;					
		}else{
			EXPECT_EQ(res.at(0).second->at(j), 0) << "j: " << j;
		}
	}

	EXPECT_EQ(res.at(1).first->size(),100);
	EXPECT_EQ(res.at(1).second->size(),dim);

	for(int j = 0; j < dim; j++){
		if(j % 11 == 0){
			EXPECT_EQ(res.at(1).second->at(j), 1) << "j: " << j;					
		}else{
			EXPECT_EQ(res.at(1).second->at(j), 0) << "j: " << j;
		}
	}
}

TEST(testMineClusGPU, _SLOW_test20Dims){
	std::default_random_engine gen;
	gen.seed(0);
	unsigned int dim = 20;
	std::normal_distribution<float> cluster1(100.0,2.0);
	std::normal_distribution<float> cluster2(1000.0,2.0);
	std::normal_distribution<float> noise(100.0,200000.0);
	auto data = new std::vector<std::vector<float>*>;
	{
		for(int i = 0; i < 300; i++){
			auto point = new std::vector<float>;
			for(int j = 0; j < dim; j++){
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
			for(int j = 0; j < dim; j++){
				if(j % 11 == 0){
					point->push_back(cluster2(gen));					
				}else{
					point->push_back(noise(gen));					
				}

			}
			data->push_back(point);
		}
	}

	EXPECT_EQ(data->size(), 400);
	EXPECT_EQ(data->at(0)->size(), dim);

	auto c = MineClusGPU(data);
	c.setSeed(2);


	auto res = c.findKClusters(1);

	EXPECT_EQ(res.size(),2);
	EXPECT_EQ(res.at(0).first->size(),300);
	EXPECT_EQ(res.at(0).second->size(),dim);

	for(int j = 0; j < dim; j++){
		if(j % 8 == 0){
			EXPECT_EQ(res.at(0).second->at(j), 1) << "j: " << j;					
		}else{
			EXPECT_EQ(res.at(0).second->at(j), 0) << "j: " << j;
		}
	}

	EXPECT_EQ(res.at(1).first->size(),100);
	EXPECT_EQ(res.at(1).second->size(),dim);

	for(int j = 0; j < dim; j++){
		if(j % 11 == 0){
			EXPECT_EQ(res.at(1).second->at(j), 1) << "j: " << j;					
		}else{
			EXPECT_EQ(res.at(1).second->at(j), 0) << "j: " << j;
		}
	}
}

TEST(testMineClusGPU, _SLOW_test30Dims){
	std::default_random_engine gen;
	gen.seed(1000);
	std::normal_distribution<float> cluster1(100.0,2.0);
	std::normal_distribution<float> cluster2(1000.0,2.0);
	std::normal_distribution<float> noise(10000.0,200000.0);
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
	c.setSeed(3000);


	auto res = c.findKClusters(1);

	EXPECT_EQ(res.size(),2);
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


	EXPECT_EQ(res.at(1).first->size(),100);
	EXPECT_EQ(res.at(1).second->size(),30);

	count = 0;
	for(int j = 0; j < 30; j++){
		if(j % 10 == 0){
			EXPECT_EQ(res.at(1).second->at(j), 1) << "j: " << j;
			count++;
		}else{
			EXPECT_EQ(res.at(1).second->at(j), 0) << "j: " << j;
		}
	}
	EXPECT_EQ(count, 3);
}


TEST(testMineClusGPU, _SUPER_SLOW_test130Dims){
	std::default_random_engine gen;
	gen.seed(1000);
	std::normal_distribution<float> cluster1(100.0,2.0);
	std::normal_distribution<float> cluster2(1000.0,2.0);
	std::normal_distribution<float> noise(10000.0,200000.0);
	auto data = new std::vector<std::vector<float>*>;
	{
		for(int i = 0; i < 300; i++){
			auto point = new std::vector<float>;
			for(int j = 0; j < 130; j++){
				if(j % 10 == 0){
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
			for(int j = 0; j < 130; j++){
				if(j % 15 == 0){
					point->push_back(cluster2(gen));					
				}else{
					point->push_back(noise(gen));					
				}

			}
			data->push_back(point);
		}
	}

	auto c = MineClusGPU(data);
	c.setSeed(3000);


	auto res = c.findKClusters(1);

	EXPECT_EQ(res.size(),2);
	EXPECT_EQ(res.at(0).first->size(),300);
	EXPECT_EQ(res.at(0).second->size(),130);

	int count = 0;
	for(int j = 0; j < 130; j++){
		if(j % 10 == 0){
			EXPECT_EQ(res.at(0).second->at(j), 1) << "j: " << j;
			count++;
		}else{
			EXPECT_EQ(res.at(0).second->at(j), 0) << "j: " << j;
		}
	}
	EXPECT_EQ(count, 13);


	EXPECT_EQ(res.at(1).first->size(),100);
	EXPECT_EQ(res.at(1).second->size(),130);

	count = 0;
	for(int j = 0; j < 130; j++){
		if(j % 15 == 0){
			EXPECT_EQ(res.at(1).second->at(j), 1) << "j: " << j;
			count++;
		}else{
			EXPECT_EQ(res.at(1).second->at(j), 0) << "j: " << j;
		}
	}
	EXPECT_EQ(count, 9);
}




TEST(testMineClusGPU, _SUPER_SLOW_test100Dims_5clusters){
	std::default_random_engine gen;
	gen.seed(10);
	std::normal_distribution<double> cluster1(100.0,2.0);
	std::normal_distribution<double> cluster2(1000.0,2.0);
	std::normal_distribution<double> cluster3(10000.0,2.0);
	std::normal_distribution<double> cluster4(100000.0,2.0);
	std::normal_distribution<double> cluster5(1000000.0,2.0);
	std::normal_distribution<double> cluster6(10000000.0,2.0);
	std::normal_distribution<double> noise(100.0,200000.0);
	auto data = new std::vector<std::vector<float>*>;
	{
		for(int i = 0; i < 300; i++){
			auto point = new std::vector<float>;
			for(int j = 0; j < 100; j++){
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
		for(int i = 0; i < 200; i++){
			auto point = new std::vector<float>;
			for(int j = 0; j < 100; j++){
				if(j % 10 == 0){
					point->push_back(cluster2(gen));					
				}else{
					point->push_back(noise(gen));					
				}

			}
			data->push_back(point);
		}
	}

	{
		for(int i = 0; i < 190; i++){
			auto point = new std::vector<float>;
			for(int j = 0; j < 100; j++){
				if(j % 10 == 0){
					point->push_back(cluster3(gen));					
				}else{
					point->push_back(noise(gen));					
				}

			}
			data->push_back(point);
		}
	}
	{
		for(int i = 0; i < 180; i++){
			auto point = new std::vector<float>;
			for(int j = 0; j < 100; j++){
				if(j % 10 == 0){
					point->push_back(cluster4(gen));					
				}else{
					point->push_back(noise(gen));					
				}

			}
			data->push_back(point);
		}
	}

	{
		for(int i = 0; i < 170; i++){
			auto point = new std::vector<float>;
			for(int j = 0; j < 100; j++){
				if(j % 10 == 0){
					point->push_back(cluster5(gen));					
				}else{
					point->push_back(noise(gen));					
				}

			}
			data->push_back(point);
		}
	}
	auto c = MineClusGPU(data);
	c.setSeed(2);


	auto res = c.findKClusters(5);

	EXPECT_EQ(res.size(),5);
	EXPECT_EQ(res.at(0).first->size(),300);
	EXPECT_EQ(res.at(0).second->size(),100);


	for(int j = 0; j < 100; j++){
		if(j % 8 == 0){
			EXPECT_EQ(res.at(0).second->at(j), 1) << "j: " << j;					
		}else{
			EXPECT_EQ(res.at(0).second->at(j), 0) << "j: " << j;
		}
	}

	EXPECT_EQ(res.at(1).first->size(),200);
	EXPECT_EQ(res.at(1).second->size(),100);

	EXPECT_EQ(res.at(2).first->size(),190);
	EXPECT_EQ(res.at(2).second->size(),100);

	EXPECT_EQ(res.at(3).first->size(),180);
	EXPECT_EQ(res.at(3).second->size(),100);

	EXPECT_EQ(res.at(4).first->size(),170);
	EXPECT_EQ(res.at(4).second->size(),100);
}
