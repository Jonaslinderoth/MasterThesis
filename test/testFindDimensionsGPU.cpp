#include <gtest/gtest.h>
#include <iostream>
#include "../src/DOC_GPU/DOCGPU_Kernels.h"
#include "../src/DOC/DOC.h"
#include <random>


TEST(testHyperCubeGPU, testFindDimmensionsInit){
	auto data = new std::vector<std::vector<float>*>;
	auto b = new std::vector<float>{1,2,3,4};
	data->push_back(b);

	
	auto c = new std::vector<unsigned int>{0};

	auto samples = new std::vector<std::vector<unsigned int>*>;
	auto d = new std::vector<unsigned int>{0};
	samples->push_back(d);
	
	auto res = findDimmensions(data, c, samples, 1);
   
	SUCCEED();
	EXPECT_EQ(res.first->size(), 1);
}


TEST(testHyperCubeGPU, testFindDimmensions){
	auto data = new std::vector<std::vector<float>*>;
	auto samples = new std::vector<std::vector<unsigned int>*>;

	auto centroids = new std::vector<unsigned int>;
	
	
	
	auto p = new std::vector<float>{1,1,1};
	auto x1 = new std::vector<float>{1,2,1000};
	auto x2 = new std::vector<float>{2,1, 1000};

	data->push_back(p);
	data->push_back(x1);
	data->push_back(x2);
	
	centroids->push_back(0);
	auto sample = new std::vector<unsigned int>{1,2};
	samples->push_back(sample);


	
	auto res2 = findDimmensions(data, centroids, samples, 1);
	auto res = res2.first;
	SUCCEED();
	EXPECT_EQ(res->size(), 1);
	EXPECT_EQ(res->at(0)->size(), 3);
	EXPECT_TRUE(res->at(0)->at(0));
	EXPECT_TRUE(res->at(0)->at(1));
	EXPECT_FALSE(res->at(0)->at(2));


	EXPECT_EQ(res2.second->size(), 1);
	EXPECT_EQ(res2.second->at(0), 2);
}



TEST(testHyperCubeGPU, testFindDimmensions_1){

	auto data = new std::vector<std::vector<float>*>;
	auto samples = new std::vector<std::vector<unsigned int>*>;
	auto centroids = new std::vector<unsigned int>{0,1};

	auto p = new std::vector<float>{1,1,1};
	auto p2 = new std::vector<float>{10,10,10};
	data->push_back(p);
	data->push_back(p2);
	

	
	{
		auto xs = new std::vector<unsigned int>{2,3};
		auto xs2 = new std::vector<unsigned int>{2,4};
		auto x1 = new std::vector<float>{1,2,1000};
		auto x2 = new std::vector<float>{2,1, 1000};
		auto x3 = new std::vector<float>{1,-2, 1000};
		data->push_back(x1);
		data->push_back(x2);
		data->push_back(x3);
		samples->push_back(xs);
		samples->push_back(xs2);
	}

	
	auto res2 = findDimmensions(data, centroids, samples, 1);
	auto res = res2.first;
	SUCCEED();
	EXPECT_EQ(res->size(), 2);
	EXPECT_EQ(res->at(0)->size(), 3);
	EXPECT_TRUE(res->at(0)->at(0));
	EXPECT_TRUE(res->at(0)->at(1));
	EXPECT_FALSE(res->at(0)->at(2));

	EXPECT_EQ(res->at(1)->size(), 3);
	EXPECT_TRUE(res->at(1)->at(0));
	EXPECT_FALSE(res->at(1)->at(1));
	EXPECT_FALSE(res->at(1)->at(2));

	

	EXPECT_EQ(res2.second->size(), 2);
	EXPECT_EQ(res2.second->at(0), 2);
	EXPECT_EQ(res2.second->at(1), 1);
}





TEST(testHyperCubeGPU, testFindDimmensions2){
	auto data = new std::vector<std::vector<float>*>;
	auto samples = new std::vector<std::vector<unsigned int>*>;
	auto centroids = new std::vector<unsigned int>{0};

	
	auto x = new std::vector<float>{1,1,1};
	auto x1 = new std::vector<float>{1,2,1000};
	auto x2 = new std::vector<float>{2,1, 1000};
	auto x3 = new std::vector<float>{1,2,1};
	auto x4 = new std::vector<float>{2,10000, 3};
	data->push_back(x);
	data->push_back(x1);
	data->push_back(x2);
	data->push_back(x3);
	data->push_back(x4);
	auto sample1 = new std::vector<unsigned int>{1,2};
	auto sample2 = new std::vector<unsigned int>{3,4};
	samples->push_back(sample1);
	samples->push_back(sample2);


	auto res2 = findDimmensions(data, centroids, samples, 2);
	auto res = res2.first;
	SUCCEED();
	EXPECT_EQ(res->size(), 2);
	EXPECT_EQ(res->at(0)->size(), 3);
	EXPECT_TRUE(res->at(0)->at(0));
	EXPECT_TRUE(res->at(0)->at(1));
	EXPECT_FALSE(res->at(0)->at(2));
	
	EXPECT_EQ(res->at(1)->size(), 3);
	EXPECT_TRUE(res->at(1)->at(0));
	EXPECT_FALSE(res->at(1)->at(1));
	EXPECT_TRUE(res->at(1)->at(2));


	EXPECT_EQ(res2.second->size(), 2);
	EXPECT_EQ(res2.second->at(0), 2);
	EXPECT_EQ(res2.second->at(1), 2);
}

/*

TEST(testFindDimensionsGPU, testFindDimmensions3){
	auto ps = new std::vector<std::vector<float>*>;
	auto p = new std::vector<float>{1,1,1,9};
	ps->push_back(p);
	auto xss = std::vector<std::vector<std::vector<float>*>*>();
	{
		auto xs = new std::vector<std::vector<float>*>;
		auto x1 = new std::vector<float>{1,2,1000,9};
		auto x2 = new std::vector<float>{2,1, 1000,9};
		xs->push_back(x1);
		xs->push_back(x2);
		xss.push_back(xs);
	}

	for(int i = 0; i < 5; i++){
		auto xs = new std::vector<std::vector<float>*>;
		auto x1 = new std::vector<float>{1,2,1,9};
		auto x2 = new std::vector<float>{2,10000, 1,9};
		xs->push_back(x1);
		xs->push_back(x2);
		xss.push_back(xs);
	}
	
	auto res2 = findDimmensions(ps, xss, 6);
	auto res = res2.first;
	
	SUCCEED();
	EXPECT_EQ(res->size(), 6);
	
	EXPECT_EQ(res->at(0)->size(), 4);
	EXPECT_TRUE(res->at(0)->at(0));
	EXPECT_TRUE(res->at(0)->at(1));
	EXPECT_FALSE(res->at(0)->at(2));
	EXPECT_TRUE(res->at(0)->at(3));
	for(int i = 1; i < 6; i++){
		EXPECT_EQ(res->at(i)->size(), 4);
		EXPECT_TRUE(res->at(i)->at(0)) << i << ", " << 0 << " is not TRUE";
		EXPECT_FALSE(res->at(i)->at(1)) << i << ", " << 1 << " is not FALSE";
		EXPECT_TRUE(res->at(i)->at(2)) << i << ", " << 2 << " is not TRUE";
		EXPECT_TRUE(res->at(i)->at(3)) << i << ", " << 3 << " is not TRUE";
	}

	EXPECT_EQ(res2.second->size(), 6);
	EXPECT_EQ(res2.second->at(0), 3);
	EXPECT_EQ(res2.second->at(1), 3);
	EXPECT_EQ(res2.second->at(2), 3);
	EXPECT_EQ(res2.second->at(3), 3);
	EXPECT_EQ(res2.second->at(4), 3);
	EXPECT_EQ(res2.second->at(5), 3);
}


*/

TEST(testFindDimensionsGPU, testFindDimmensions4){
	auto data = new std::vector<std::vector<float>*>;
	auto samples = new std::vector<std::vector<unsigned int>*>;
	auto centroids = new std::vector<unsigned int>{0,1};

	{auto p = new std::vector<float>{1,1,1,9};
	data->push_back(p);}
	{auto p = new std::vector<float>{1000,1000,1000,9000};
	data->push_back(p);}
	auto x1 = new std::vector<float>{1,2,1000,9};
	auto x2 = new std::vector<float>{2,1, 1000,9};
	auto x3 = new std::vector<float>{1,1000,1,9};
	auto x4 = new std::vector<float>{2,1000, 1,9};
	data->push_back(x1);
	data->push_back(x2);
	data->push_back(x3);
	data->push_back(x4);

	auto sample1 = new std::vector<unsigned int>{2,3};
	auto sample2 = new std::vector<unsigned int>{4,5};
	auto sample3 = new std::vector<unsigned int>{4,5};
	auto sample4 = new std::vector<unsigned int>{4,5};
	auto sample5 = new std::vector<unsigned int>{4,5};
	auto sample6 = new std::vector<unsigned int>{4,5};

	samples->push_back(sample1);
	samples->push_back(sample2);
	samples->push_back(sample3);
	samples->push_back(sample4);
	samples->push_back(sample5);
	samples->push_back(sample6);
	
	auto res2 = findDimmensions(data, centroids, samples, 3);
	auto res = res2.first;
	SUCCEED();
	EXPECT_EQ(res->size(), 6);

	
	  std::cout << std::endl;
	for(int i = 0; i < res->size(); i++){
		for(int j = 0; j < res->at(i)->size(); j++){
			std::cout << res->at(i)->at(j) << ", ";
		}
		std::cout << std::endl;
	}
	
	
	
	EXPECT_EQ(res->at(0)->size(), 4);
	EXPECT_TRUE(res->at(0)->at(0));
	EXPECT_TRUE(res->at(0)->at(1));
	EXPECT_FALSE(res->at(0)->at(2));
	EXPECT_TRUE(res->at(0)->at(3));
	
	for(int i = 1; i < 3; i++){
		EXPECT_EQ(res->at(i)->size(), 4);
		EXPECT_TRUE(res->at(i)->at(0)) << i << ", " << 0 << " is not TRUE";
		EXPECT_FALSE(res->at(i)->at(1)) << i << ", " << 1 << " is not FALSE";
		EXPECT_TRUE(res->at(i)->at(2)) << i << ", " << 2 << " is not TRUE";
		EXPECT_TRUE(res->at(i)->at(3)) << i << ", " << 3 << " is not TRUE";
	}



	for(int i = 3; i < 6; i++){
		EXPECT_EQ(res->at(i)->size(), 4);
		EXPECT_FALSE(res->at(i)->at(0)) << i << ", " << 0 << " is not FALSE";
		EXPECT_TRUE(res->at(i)->at(1)) << i << ", " << 1 << " is not TRUE";
		EXPECT_FALSE(res->at(i)->at(2)) << i << ", " << 2 << " is not FALSE";
		EXPECT_FALSE(res->at(i)->at(3)) << i << ", " << 3 << " is not FALSE";
	}
	EXPECT_EQ(res2.second->size(), 6);
	EXPECT_EQ(res2.second->at(0), 3);
	EXPECT_EQ(res2.second->at(1), 3);
	EXPECT_EQ(res2.second->at(2), 3);
	EXPECT_EQ(res2.second->at(3), 1);
	EXPECT_EQ(res2.second->at(4), 1);
	EXPECT_EQ(res2.second->at(5), 1);

}
/*

TEST(testFindDimensionsGPU, testFindDimmensions5){
	auto ps = new std::vector<std::vector<float>*>;
	{auto p = new std::vector<float>{1,1,1,1};
		ps->push_back(p);}
	{auto p = new std::vector<float>{1000,1000,1000,1000};
	ps->push_back(p);}
	auto xss = std::vector<std::vector<std::vector<float>*>*>();
	auto xs = new std::vector<std::vector<float>*>;
	auto x1 = new std::vector<float>{1,2,1000,1};
	auto x2 = new std::vector<float>{2,1, 1000,1};
	xs->push_back(x1);
	xs->push_back(x2);
	xss.push_back(xs);

	
	auto res2 = findDimmensions(ps, xss, 1);
	auto res = res2.first;
	SUCCEED();

	
	/*std::cout << std::endl;
	for(int i = 0; i < res->size(); i++){
		for(int j = 0; j < res->at(i)->size(); j++){
			std::cout << res->at(i)->at(j) << ", ";
		}
		std::cout << std::endl;
		}
	
	
	EXPECT_EQ(res->size(), 1);
	EXPECT_EQ(res->at(0)->size(), 4);
	EXPECT_TRUE(res->at(0)->at(0));
	EXPECT_TRUE(res->at(0)->at(1));
	EXPECT_FALSE(res->at(0)->at(2));	

	//EXPECT_EQ(res->at(1)->size(), 4);
	//EXPECT_FALSE(res->at(1)->at(0));
	//EXPECT_FALSE(res->at(1)->at(1));
	//EXPECT_TRUE(res->at(1)->at(2));
}



TEST(testFindDimensionsGPU, testFindDimmensionsRandom){
	std::vector<std::vector<float>*>* ps = new std::vector<std::vector<float>*>;
	int amount_of_ps = 100;
	int number_of_samples = 200;
	int m = number_of_samples/amount_of_ps;

	int sample_size = 20;
	int point_dim = 200;
	std::default_random_engine generator;
	generator.seed(100);
	std::uniform_real_distribution<double> distribution(0.0,20.0);

	
	for(int i = 0; i < amount_of_ps; i++){
		auto p = new std::vector<float>;
		for(int j = 0; j < point_dim; j++){
			float d = distribution(generator);
			//std::cout << d << ", ";
			p->push_back(d);
		}
		//std::cout << std::endl;
		ps->push_back(p);
	}
	//std::cout << std::endl;

	std::vector<std::vector<std::vector<float>*>*> xs = std::vector<std::vector<std::vector<float>*>*>();
	for(int i = 0; i < number_of_samples; i++){
		auto pk = new std::vector<std::vector<float>*>;
		for(int j = 0; j < sample_size; j++){
			auto p = new std::vector<float>;
			for(int k = 0; k < point_dim; k++){
				float d = distribution(generator);
				p->push_back(d);
				//std::cout << d << ", ";
			}
			pk->push_back(p);
			//std::cout << std::endl;
		}
		//std::cout << std::endl;
		xs.push_back(pk);
	}

	auto res2 = findDimmensions(ps, xs, m);
	auto resGPU = res2.first;
	
	
	DOC d;
	int f = 0,t = 0;
	for(int i = 0; i < number_of_samples; i++){
			std::vector<bool>* res = d.findDimensions(ps->at(i/m), xs.at(i), 10);
			std::vector<bool>* res2 = resGPU->at(i);
			EXPECT_EQ(res->size(), res2->size());
			/*
			std::cout << "CPU: ";
			for(int k = 0; k < res->size();k++){
				std::cout << res->at(k) << ", ";
			}
			std::cout << std::endl;
			
			std::cout << "GPU: ";
			for(int k = 0; k < res->size();k++){
				std::cout << res2->at(k) << ", ";
			}
			std::cout << std::endl;
			
			
			for(int k = 0; k < res->size(); k++){
				ASSERT_EQ(res->at(k), res2->at(k)) << "Not equal at k: " << k << ", and " << i <<"th sample and " << i/m << "th centroid";
				if(res->at(k)){
					t++;
				}else{
					f++;
				}
			}
		  
			
		
	}
	//std::cout << "number of false: " << f << " number of true: " << t <<std::endl;


	
}
*/
