#include <gtest/gtest.h>
#include <iostream>
#include "../src/DOC_GPU/HyperCube.h"
#include "../src/DOC/HyperCube.h"
#include <random>
 

class testPointsContainedGPU : public ::testing::Test {
public:
  // Per-test-suite set-up.
  // Called before the first test in this test suite.
  // Can be omitted if not needed.
  static void SetUpTestCase() {
  }

  // Per-test-suite tear-down.
  // Called after the last test in this test suite.
  // Can be omitted if not needed.
  static void TearDownTestCase() {

  }


	virtual void SetUp() {
	}



};

TEST_F(testPointsContainedGPU, testSimple){
	auto a = new std::vector<std::vector<bool>*>;
	auto b = new std::vector<std::vector<float>*>;
	auto d = new std::vector<std::vector<float>*>;
	auto dd = new std::vector<float>{0};
	d->push_back(dd);
	auto c = pointsContained(a,b,d,1).first;
	
	SUCCEED();
	EXPECT_EQ(c->size(), a->size());
}

TEST_F(testPointsContainedGPU, testWithSimpleData){
	auto a = new std::vector<std::vector<bool>*>;
	auto aa = new std::vector<bool>{true, false};
	a->push_back(aa);

	auto b = new std::vector<std::vector<float>*>;
    auto bb = new std::vector<float>{9,9};
	b->push_back(bb);
	auto bbb = new std::vector<float>{111,111};
	b->push_back(bbb);

    auto centroid = new std::vector<float>{10,10};
	auto centorids = new std::vector<std::vector<float>*>;
	centorids->push_back(centroid);
	auto c1 = pointsContained(a,b,centorids,1);
	auto c = c1.first;
	SUCCEED();
	EXPECT_EQ(c->size(), a->size());
	EXPECT_TRUE(c->at(0)->at(0));
	EXPECT_FALSE(c->at(0)->at(1));

	EXPECT_EQ(c1.second->size(), 1);
	EXPECT_EQ(c1.second->at(0), 1);
}
TEST_F(testPointsContainedGPU, testWithSimpleData2){
	auto a = new std::vector<std::vector<bool>*>;
	auto aa = new std::vector<bool>{true, false, false};
	a->push_back(aa);

	auto b = new std::vector<std::vector<float>*>;
    auto bb = new std::vector<float>{9,9,9};
	b->push_back(bb);
	{auto bbb = new std::vector<float>{111,111,111};
		b->push_back(bbb);};
	{auto bbb = new std::vector<float>{-111,-111,-111};
		b->push_back(bbb);};

    auto centroid = new std::vector<float>{10,10,10};
	auto centorids = new std::vector<std::vector<float>*>;
	centorids->push_back(centroid);
	auto c1 = pointsContained(a,b,centorids,1);
	auto c = c1.first;
	SUCCEED();
	EXPECT_EQ(c->size(), a->size());
	EXPECT_TRUE(c->at(0)->at(0));
	EXPECT_FALSE(c->at(0)->at(1));
	EXPECT_FALSE(c->at(0)->at(2));


	EXPECT_EQ(c1.second->size(), 1);
	EXPECT_EQ(c1.second->at(0), 1);
}



TEST_F(testPointsContainedGPU, testWithSimpleData3){
	auto a = new std::vector<std::vector<bool>*>;
	{auto aa = new std::vector<bool>{true, false, false};
		a->push_back(aa);};
	{auto aa = new std::vector<bool>{true, false, false};
		a->push_back(aa);};
	
	auto b = new std::vector<std::vector<float>*>;
    auto bb = new std::vector<float>{9,9,9};
	b->push_back(bb);
	{auto bbb = new std::vector<float>{111,111,111};
		b->push_back(bbb);};
	{auto bbb = new std::vector<float>{-111,-111,-111};
		b->push_back(bbb);};

    auto centroid = new std::vector<float>{10,10,10};
	auto centorids = new std::vector<std::vector<float>*>;
	centorids->push_back(centroid);
	auto c1 = pointsContained(a,b,centorids,1);
	auto c = c1.first;
	SUCCEED();
	EXPECT_EQ(c->size(), a->size());
	EXPECT_TRUE(c->at(0)->at(0));
	EXPECT_FALSE(c->at(0)->at(1));
	EXPECT_FALSE(c->at(0)->at(2));

	EXPECT_TRUE(c->at(1)->at(0));
	EXPECT_FALSE(c->at(1)->at(1));
	EXPECT_FALSE(c->at(1)->at(2));


	EXPECT_EQ(c1.second->size(), 2);
	EXPECT_EQ(c1.second->at(0), 1);
	EXPECT_EQ(c1.second->at(1), 1);
}

TEST_F(testPointsContainedGPU, testWithSimpleData4){
	auto a = new std::vector<std::vector<bool>*>;
	{auto aa = new std::vector<bool>{true, false, false};
		a->push_back(aa);};
	{auto aa = new std::vector<bool>{true, false, false};
		a->push_back(aa);};
	{auto aa = new std::vector<bool>{false, true, true};
		a->push_back(aa);};	
	auto b = new std::vector<std::vector<float>*>;
    auto bb = new std::vector<float>{9,9,9};
	b->push_back(bb);
	{auto bbb = new std::vector<float>{111,111,111};
		b->push_back(bbb);};
	{auto bbb = new std::vector<float>{-111,-111,-111};
		b->push_back(bbb);};

	{auto bbb = new std::vector<float>{8,-111,-111};
		b->push_back(bbb);};
    auto centroid = new std::vector<float>{10,10,10};
	auto centorids = new std::vector<std::vector<float>*>;
	centorids->push_back(centroid);
	auto c1 = pointsContained(a,b,centorids,1);
	auto c = c1.first;
	SUCCEED();
	EXPECT_EQ(c->size(), a->size());
	EXPECT_EQ(c->at(0)->size(), 4);
	EXPECT_TRUE(c->at(0)->at(0));
	EXPECT_FALSE(c->at(0)->at(1));
	EXPECT_FALSE(c->at(0)->at(2));
	EXPECT_TRUE(c->at(0)->at(3));


	EXPECT_EQ(c->at(1)->size(), 4);
	EXPECT_TRUE(c->at(1)->at(0));
	EXPECT_FALSE(c->at(1)->at(1));
	EXPECT_FALSE(c->at(1)->at(2));
	EXPECT_TRUE(c->at(1)->at(3));
	
	EXPECT_EQ(c->at(2)->size(), 4);
	EXPECT_TRUE(c->at(2)->at(0));
	EXPECT_FALSE(c->at(2)->at(1));
	EXPECT_FALSE(c->at(2)->at(2));
	EXPECT_FALSE(c->at(2)->at(3));

	EXPECT_EQ(c1.second->size(), 3);
	EXPECT_EQ(c1.second->at(0), 2);
	EXPECT_EQ(c1.second->at(1), 2);
	EXPECT_EQ(c1.second->at(2), 1);
	
}


TEST_F(testPointsContainedGPU, testWithSimpleData5){
	auto a = new std::vector<std::vector<bool>*>;
	{auto aa = new std::vector<bool>{true, true, false};
		a->push_back(aa);};
	auto b = new std::vector<std::vector<float>*>;
    auto bb = new std::vector<float>{9,9,9};
	b->push_back(bb);
	{auto bbb = new std::vector<float>{111,111,111};
		b->push_back(bbb);};
	{auto bbb = new std::vector<float>{8,8,-111};
		b->push_back(bbb);};
	{auto bbb = new std::vector<float>{8,8,111};
		b->push_back(bbb);};
    auto centroid = new std::vector<float>{10,10,10};
	auto centorids = new std::vector<std::vector<float>*>;
	centorids->push_back(centroid);
	auto c1 = pointsContained(a,b,centorids,1);
	auto c = c1.first;
	SUCCEED();
	EXPECT_EQ(c->size(), a->size());
	EXPECT_EQ(c->at(0)->size(), 4);
	EXPECT_TRUE(c->at(0)->at(0));
	EXPECT_FALSE(c->at(0)->at(1));
	EXPECT_TRUE(c->at(0)->at(2));
	EXPECT_TRUE(c->at(0)->at(3));


	EXPECT_EQ(c1.second->size(), 1);
	EXPECT_EQ(c1.second->at(0), 3);
}


TEST_F(testPointsContainedGPU, testWithSimpleData6){
	auto a = new std::vector<std::vector<bool>*>;
	{auto aa = new std::vector<bool>{true, true, true};
		a->push_back(aa);};
	auto b = new std::vector<std::vector<float>*>;
    auto bb = new std::vector<float>{9,9,9};
	b->push_back(bb);
	{auto bbb = new std::vector<float>{111,111,111};
		b->push_back(bbb);};
	{auto bbb = new std::vector<float>{8,8,-111};
		b->push_back(bbb);};
	{auto bbb = new std::vector<float>{8,8,111};
		b->push_back(bbb);};
    auto centroid = new std::vector<float>{10,10,10};
	auto centorids = new std::vector<std::vector<float>*>;
	centorids->push_back(centroid);
	auto c1 = pointsContained(a,b,centorids,1);
	auto c = c1.first;
	SUCCEED();
	EXPECT_EQ(c->size(), a->size());
	EXPECT_EQ(c->at(0)->size(), 4);
	EXPECT_TRUE(c->at(0)->at(0));
	EXPECT_FALSE(c->at(0)->at(1));
	EXPECT_FALSE(c->at(0)->at(2));
	EXPECT_FALSE(c->at(0)->at(3));

	EXPECT_EQ(c1.second->size(), 1);
	EXPECT_EQ(c1.second->at(0), 1);
}


TEST_F(testPointsContainedGPU, testWithSimpleData7){
	auto a = new std::vector<std::vector<bool>*>;
	{auto aa = new std::vector<bool>{true, false};
		a->push_back(aa);};
	auto b = new std::vector<std::vector<float>*>;
    auto bb = new std::vector<float>{9,111};
	b->push_back(bb);
    auto centroid = new std::vector<float>{10,10};
	auto centorids = new std::vector<std::vector<float>*>;
	centorids->push_back(centroid);
	auto c1 = pointsContained(a,b,centorids,1);
	auto c = c1.first;
	SUCCEED();
	EXPECT_EQ(c->size(), a->size());
	EXPECT_EQ(c->at(0)->size(), 1);
	EXPECT_TRUE(c->at(0)->at(0));
	EXPECT_EQ(c1.second->size(), 1);
	EXPECT_EQ(c1.second->at(0), 1);

}


TEST_F(testPointsContainedGPU, testWithSimpleData8){
	auto a = new std::vector<std::vector<bool>*>;
	{auto aa = new std::vector<bool>{true, true};
		a->push_back(aa);};
	auto b = new std::vector<std::vector<float>*>;
    auto bb = new std::vector<float>{9,111};
	b->push_back(bb);
    auto centroid = new std::vector<float>{10,10};
	auto centorids = new std::vector<std::vector<float>*>;
	centorids->push_back(centroid);
	auto c1 = pointsContained(a,b,centorids,1);
	auto c = c1.first;
	SUCCEED();
	EXPECT_EQ(c->size(), a->size());
	EXPECT_EQ(c->at(0)->size(), 1);
	EXPECT_FALSE(c->at(0)->at(0));

	EXPECT_EQ(c1.second->size(), 1);
	EXPECT_EQ(c1.second->at(0), 0);
}


TEST_F(testPointsContainedGPU, testWithSimpleData9){
	auto a = new std::vector<std::vector<bool>*>;
	{auto aa = new std::vector<bool>{true};
		a->push_back(aa);};
	auto b = new std::vector<std::vector<float>*>;
    auto bb = new std::vector<float>{111};
	b->push_back(bb);
    auto centroid = new std::vector<float>{10};
	auto centorids = new std::vector<std::vector<float>*>;
	centorids->push_back(centroid);
	auto c1 = pointsContained(a,b,centorids,1);
	auto c = c1.first;
	SUCCEED();
	EXPECT_EQ(c->size(), a->size());
	EXPECT_EQ(c->at(0)->size(), 1);
	EXPECT_FALSE(c->at(0)->at(0));
	EXPECT_EQ(c1.second->size(), 1);
	EXPECT_EQ(c1.second->at(0), 0);
}


TEST_F(testPointsContainedGPU, testWithSimpleData10){
	auto a = new std::vector<std::vector<bool>*>;
	{auto aa = new std::vector<bool>{true, true};
		a->push_back(aa);};
	auto b = new std::vector<std::vector<float>*>;
    auto bb = new std::vector<float>{111,9};
	b->push_back(bb);
    auto centroid = new std::vector<float>{10,10};
	auto centorids = new std::vector<std::vector<float>*>;
	centorids->push_back(centroid);
	auto c1 = pointsContained(a,b,centorids,1);
	auto c = c1.first;
	SUCCEED();
	EXPECT_EQ(c->size(), a->size());
	EXPECT_EQ(c->at(0)->size(), 1);
	EXPECT_FALSE(c->at(0)->at(0));
	EXPECT_EQ(c1.second->size(), 1);
	EXPECT_EQ(c1.second->at(0), 0);
}


TEST_F(testPointsContainedGPU, testWithSimpleData11){
	auto a = new std::vector<std::vector<bool>*>;
	{auto aa = new std::vector<bool>{true, true, true};
		a->push_back(aa);};
	auto b = new std::vector<std::vector<float>*>;
    auto bb = new std::vector<float>{9,111, 9};
	b->push_back(bb);
    auto centroid = new std::vector<float>{10,10,10};
	auto centorids = new std::vector<std::vector<float>*>;
	centorids->push_back(centroid);
	auto c1 = pointsContained(a,b,centorids,1);
	auto c = c1.first;
	SUCCEED();
	EXPECT_EQ(c->size(), a->size());
	EXPECT_EQ(c->at(0)->size(), 1);
	EXPECT_FALSE(c->at(0)->at(0));
	EXPECT_EQ(c1.second->size(), 1);
	EXPECT_EQ(c1.second->at(0), 0);
}



TEST_F(testPointsContainedGPU, testRandomCompCPU){
	int no_dims = 100;
	int point_dim = 200;
	int data_size = 1001;
	std::default_random_engine generator;
	generator.seed(100);
	
	std::uniform_real_distribution<double> distribution(15,20);
	std::uniform_real_distribution<double> distribution2(9,26);

	std::vector<std::vector<float>*>* centorids = new std::vector<std::vector<float>*>;
	for(int i = 0; i < no_dims; i++){
		auto point = new std::vector<float>;
		for(int j = 0; j < point_dim; j++){
			point->push_back(distribution(generator));
		}
		centorids->push_back(point);
	}


	std::vector<std::vector<bool>*>* dims = new std::vector<std::vector<bool>*>;
	for(int i = 0; i < no_dims; i++){
		auto point = new std::vector<bool>;
		for(int j = 0; j < point_dim; j++){
			auto a = distribution2(generator)< 13;
			point->push_back(a);
			
		}
		dims->push_back(point);
	}

	
	
	std::vector<std::vector<float>*>* data = new std::vector<std::vector<float>*>;
	for(int i = 0; i < data_size; i++){
		auto point = new std::vector<float>;
		for(int j = 0; j < point_dim; j++){
			point->push_back(distribution2(generator));
		}
		data->push_back(point);
	}

	auto c1 = pointsContained(dims, data, centorids,1);
	auto c = c1.first;
	int t = 0, f = 0;
	for(int i = 0; i < centorids->size(); i++){
		auto cpu = HyperCube(centorids->at(i), 10, dims->at(i));
		for(int j = 0; j < data->size(); j++){
			auto d = cpu.pointContained(data->at(j));
			if(d){
				t++;
			}else{
				f++;
			}
			EXPECT_EQ(d,c->at(i)->at(j)) << "i: " << i << " j: " <<j << " : " << data->at(j);
		}
	}
	/*
	std::cout << "number of true: " << t << " number of false: " << f << std::endl;



	for(int i = 0; i < point_dim; i++){
		std::cout << centorids->at(98)->at(i) << ", ";
	}
	
	std::cout << std::endl;
	
	for(int i = 0; i < point_dim; i++){
		std::cout << data->at(747)->at(i) << ", ";
	}
	std::cout << std::endl;

	std::cout << "GPU version:" << std::endl;
	for(int i = 0; i < point_dim; i++){
		std::cout << ((!dims->at(98)->at(i))|| abs(centorids->at(98)->at(i) - data->at(747)->at(i)) < 10.0) << ", ";
	}
	std::cout << std::endl;
	int id = 0;
	std::cout << "CPU version:" << std::endl;
	for(int i = 0; i < point_dim; i++){
		

		float a = centorids->at(98)->at(i)-10.0;
		float b = centorids->at(98)->at(i)+10.0;
		float min = std::min(a,b);
		float max = std::max(a,b);
		if(not (min < data->at(747)->at(i) && max > data->at(747)->at(i))){
			std::cout << 0 << ", ";
			id = i;
		}else{
			std::cout << 1 << ", ";			
		}
		
		
	}
	std::cout << std::endl;
	std::cout << id << std::endl;

	std::cout << data->at(747)->at(id) << std::endl;
	std::cout << centorids->at(98)->at(id) << std::endl;
	std::cout << dims->at(98)->at(id) << std::endl;
	*/
	
}
