#include <gtest/gtest.h>
#include <iostream>
#include "../src/DOC_GPU/DOCGPU_Kernels.h"
#include "../src/DOC/HyperCube.h"
#include "../src/randomCudaScripts/arrayEqual.h"
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
	auto dims = new std::vector<std::vector<bool>*>;
	auto data = new std::vector<std::vector<float>*>;
	auto centroids = new std::vector<unsigned int>{0};
	
	auto dd = new std::vector<float>{0,0};
	data->push_back(dd);

	auto dim = new std::vector<bool>{true, true};
	dims->push_back(dim);
	auto c = pointsContained(dims,data,centroids,1);
	
	SUCCEED();
	EXPECT_EQ(c.first->size(), 1);
	EXPECT_EQ(c.first->at(0)->size(), 1);
	EXPECT_EQ(c.first->at(0)->at(0), true);
}

TEST_F(testPointsContainedGPU, testSimple2){
	auto dims = new std::vector<std::vector<bool>*>;
	auto data = new std::vector<std::vector<float>*>;
	auto centroids = new std::vector<unsigned int>{0};
	
	auto dd = new std::vector<float>{0,0};
	auto dd2 = new std::vector<float>{0,100};
	data->push_back(dd);
	data->push_back(dd2);

	auto dim = new std::vector<bool>{true, true};
	dims->push_back(dim);
	auto c = pointsContained(dims,data,centroids,1);
	
	SUCCEED();
	EXPECT_EQ(c.first->size(), 1);
	EXPECT_EQ(c.first->at(0)->size(), 2);
	EXPECT_EQ(c.first->at(0)->at(0), true);
	EXPECT_EQ(c.first->at(0)->at(1), false);
}


TEST_F(testPointsContainedGPU, testSimple3){
	auto dims = new std::vector<std::vector<bool>*>;
	auto data = new std::vector<std::vector<float>*>;
	auto centroids = new std::vector<unsigned int>{0,1};
	
	auto dd = new std::vector<float>{0,0};
	auto dd2 = new std::vector<float>{0,100};
	data->push_back(dd);
	data->push_back(dd2);

	auto dim = new std::vector<bool>{true, true};
	dims->push_back(dim);
	auto dim2 = new std::vector<bool>{true, true};
	dims->push_back(dim2);
	auto c = pointsContained(dims,data,centroids,1);
	
	SUCCEED();
	EXPECT_EQ(c.first->size(), 2);
	EXPECT_EQ(c.first->at(0)->size(), 2);
	EXPECT_EQ(c.first->at(0)->at(0), true);
	EXPECT_EQ(c.first->at(0)->at(1), false);

	EXPECT_EQ(c.first->at(1)->size(), 2);
	EXPECT_EQ(c.first->at(1)->at(0), false);
	EXPECT_EQ(c.first->at(1)->at(1), true);
}


/*

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

*/
TEST_F(testPointsContainedGPU, testWithSimpleData2){
	auto dims = new std::vector<std::vector<bool>*>;
	auto data = new std::vector<std::vector<float>*>;
	auto centroids = new std::vector<unsigned int>{0};

	data->push_back(new std::vector<float>{10,10,10});
	dims->push_back(new std::vector<bool>{true, false, false});
	data->push_back(new std::vector<float>{9,9,9});
	data->push_back(new std::vector<float>{111,111,111});
	data->push_back(new std::vector<float>{-111,-111,-111});


	auto c1 = pointsContained(dims,data,centroids,1);
	auto c = c1.first;
	SUCCEED();
	EXPECT_EQ(c->size(), dims->size());
	EXPECT_TRUE(c->at(0)->at(0));
	EXPECT_TRUE(c->at(0)->at(1));
	EXPECT_FALSE(c->at(0)->at(2));
	EXPECT_FALSE(c->at(0)->at(3));


	EXPECT_EQ(c1.second->size(), 1);
	EXPECT_EQ(c1.second->at(0), 2);
}


TEST_F(testPointsContainedGPU, testWithSimpleData2_2){
	auto dims = new std::vector<std::vector<bool>*>;
	auto data = new std::vector<std::vector<float>*>;
	auto centroids = new std::vector<unsigned int>{0,1};

	data->push_back(new std::vector<float>{10,10,10});
	data->push_back(new std::vector<float>{110,10,10});
	dims->push_back(new std::vector<bool>{true, false, false});
	dims->push_back(new std::vector<bool>{true, false, true});
	data->push_back(new std::vector<float>{9,9,9});
	data->push_back(new std::vector<float>{111,111,111});
	data->push_back(new std::vector<float>{-111,-111,-111});


	auto c1 = pointsContained(dims,data,centroids,1);
	auto c = c1.first;
	SUCCEED();
	EXPECT_EQ(c->size(), dims->size());
	EXPECT_TRUE(c->at(0)->at(0));
	EXPECT_FALSE(c->at(0)->at(1));
	EXPECT_TRUE(c->at(0)->at(2));
	EXPECT_FALSE(c->at(0)->at(3));
	EXPECT_FALSE(c->at(0)->at(4));

	EXPECT_FALSE(c->at(1)->at(0));
	EXPECT_TRUE(c->at(1)->at(1));
	EXPECT_FALSE(c->at(1)->at(2));
	EXPECT_FALSE(c->at(1)->at(3));
	EXPECT_FALSE(c->at(1)->at(4));

	EXPECT_EQ(c1.second->size(), 2);
	EXPECT_EQ(c1.second->at(0), 2);
	EXPECT_EQ(c1.second->at(1), 1);
}
/*


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
	auto c1 = pointsContained(a,b,centorids,2);
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
	auto c1 = pointsContained(a,b,centorids,3);
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
*/

TEST_F(testPointsContainedGPU, SUPER_SLOW_testRandomCompCPU){
	unsigned int point_dim = 100;
	unsigned int no_data = 2000;
	unsigned int no_centroids = 20;
	unsigned int no_dims = 1000;
	unsigned int m = no_dims/no_centroids;
	std::default_random_engine generator;
	generator.seed(100);
	
	std::uniform_real_distribution<double> distribution(15,20);
	std::uniform_real_distribution<double> distribution2(9,26);


	auto data = new std::vector<std::vector<float>*>;
	auto centroids = new std::vector<unsigned int>;
	auto dims = new std::vector<std::vector<bool>*>;

	for(int i = 0; i < no_data; i++){
		auto point = new std::vector<float>;
		for(int j = 0; j < point_dim; j++){
			point->push_back(distribution2(generator));
		}
		data->push_back(point);
		centroids->push_back(i);
	}
	for(int i = data->size()-1; i < no_data; i++){
		auto point = new std::vector<float>;
		for(int j = 0; j < point_dim; j++){
			point->push_back(distribution(generator));
		}
		data->push_back(point);
	}
	for(int i = 0; i < no_dims; i++){
		auto dim = new std::vector<bool>;
		for(int j = 0; j < point_dim; j++){
			dim->push_back(distribution2(generator)< 13);
		}
		dims->push_back(dim);
	}
	auto c1 = pointsContained(dims, data, centroids,m);
	
	auto c = c1.first;

	int t = 0, f = 0;
	for(int i = 0; i < dims->size(); i++){

		auto centroid = data->at(centroids->at(i/m));
		auto cpu = HyperCube(centroid, 10, dims->at(i));
		for(int j = 0; j < data->size(); j++){
			auto d = cpu.pointContained(data->at(j));
			if(d){
				t++;
			}else{
				f++;
			}
			EXPECT_EQ(d,c->at(i)->at(j)) << "i: " << i << " j: " <<j << " : " << data->at(j);
			if(d != c->at(i)->at(j)){
				break;
			}
		}
	}

}



TEST_F(testPointsContainedGPU, SUPER_SLOW_testIfDifferentPointContainedDeviceKernel){
	unsigned int point_dim = 100;
	unsigned int no_data = 4000;
	unsigned int no_centroids = 20;
	unsigned int no_dims = 1000;
	unsigned int m = no_dims/no_centroids;
	std::default_random_engine generator;
	generator.seed(100);

	std::uniform_real_distribution<double> distribution(15,20);
	std::uniform_real_distribution<double> distribution2(9,26);


	auto data = new std::vector<std::vector<float>*>;
	auto centroids = new std::vector<unsigned int>;
	auto dims = new std::vector<std::vector<bool>*>;

	for(int i = 0; i < no_data; i++){
		auto point = new std::vector<float>;
		for(int j = 0; j < point_dim; j++){
			point->push_back(distribution2(generator));
		}
		data->push_back(point);
		centroids->push_back(i);
	}
	for(int i = data->size()-1; i < no_data; i++){
		auto point = new std::vector<float>;
		for(int j = 0; j < point_dim; j++){
			point->push_back(distribution(generator));
		}
		data->push_back(point);
	}
	for(int i = 0; i < no_dims; i++){
		auto dim = new std::vector<bool>;
		for(int j = 0; j < point_dim; j++){
			dim->push_back(distribution2(generator)< 13);
		}
		dims->push_back(dim);
	}
	auto c0 = pointsContained(dims, data, centroids,m,10,0);
	auto c1 = pointsContained(dims, data, centroids,m,10,1);
	auto c2 = pointsContained(dims, data, centroids,m,10,2);
	//auto c3 = pointsContained(dims, data, centroids,m,10,3);




	EXPECT_TRUE(areTheyEqual_h(c1,c0)) << "c1 , c0 ";
	EXPECT_TRUE(areTheyEqual_h(c2,c0)) << "c2 , c0 ";
	//EXPECT_TRUE(areTheyEqual_h(c3,c0)) << "c3 , c0 ";

	auto c = c1.first;

	int t = 0, f = 0;
	for(int i = 0; i < dims->size(); i++){

		auto centroid = data->at(centroids->at(i/m));
		auto cpu = HyperCube(centroid, 10, dims->at(i));
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
}


TEST_F(testPointsContainedGPU, _SUPER_SLOW_testIfDifferentPointContainedDeviceKernelMany){


	std::mt19937 gen{0};
	gen.seed(1);
	static std::random_device rand;
	std::uniform_int_distribution<int> distSmall(6, 20);
	std::uniform_int_distribution<int> distBig(400, 4000);
	unsigned long small = 20;
	unsigned long big = 4000;


	for(unsigned long point_dim = 10 ; point_dim < 400-small ; point_dim += small){
	    for(unsigned long no_data = 100 ; no_data < 10000-big ; no_data += big){
		    for(unsigned long no_dims = 100 ; no_dims < 10000-big; no_dims += big){



				unsigned int no_centroids = 20;
				unsigned int m = ceilf((float)no_dims/(float)no_centroids);
				std::default_random_engine generator;
				generator.seed(100);

				std::uniform_real_distribution<double> distribution(15,20);
				std::uniform_real_distribution<double> distribution2(9,26);


				auto data = new std::vector<std::vector<float>*>;
				auto centroids = new std::vector<unsigned int>;
				auto dims = new std::vector<std::vector<bool>*>;

				for(int i = 0; i < no_data/2; i++){
					auto point = new std::vector<float>;
					for(int j = 0; j < point_dim; j++){
						point->push_back(distribution2(generator));
					}
					data->push_back(point);
				}

                for(int i = data->size()-1; i < no_data; i++){
                auto point = new std::vector<float>;
                for(int j = 0; j < point_dim; j++){
                point->push_back(distribution(generator));
                }
                data->push_back(point);
                }

				for(unsigned int i = 0; i < no_centroids; i++){
                    centroids->push_back(i);
				}



				for(int i = 0; i < no_dims; i++){
					auto dim = new std::vector<bool>;
					for(int j = 0; j < point_dim; j++){
						dim->push_back(distribution2(generator)< 13);
					}
					dims->push_back(dim);
				}
				auto c0 = pointsContained(dims, data, centroids,m,10,0);
				auto c1 = pointsContained(dims, data, centroids,m,10,1);
				auto c2 = pointsContained(dims, data, centroids,m,10,2);
				auto c3 = pointsContained(dims, data, centroids,m,10,3);
				//auto c4 = pointsContained(dims, data, centroids,m,10,4);
				auto c5 = pointsContained(dims, data, centroids,m,10,5,2);

				std::cout << "done with gpu" << std::endl;
				if(not areTheyEqual_h(c1,c0)){
					std::cout << "error c1" << std::endl;
				}
				if(not areTheyEqual_h(c2,c0)){
					std::cout << "error c2" << std::endl;
				}
				if(not areTheyEqual_h(c3,c0)){
					std::cout << "error c3" << std::endl;
				}
				/*
				if(not areTheyEqual_h(c4,c0)){
					std::cout << "error c4" << std::endl;
				}
				*/
				if(not areTheyEqual_h(c5,c0)){
					std::cout << "error c5" << std::endl;
				}

				EXPECT_TRUE(areTheyEqual_h(c1,c0)) << "c1 point_dim: " << point_dim << " no_dims " << no_dims << " no_data " << no_data << std::endl;
				EXPECT_TRUE(areTheyEqual_h(c2,c0)) << "c2 point_dim: " << point_dim << " no_dims " << no_dims << " no_data " << no_data << std::endl;
				EXPECT_TRUE(areTheyEqual_h(c3,c0)) << "c3 point_dim: " << point_dim << " no_dims " << no_dims << " no_data " << no_data << std::endl;
				//EXPECT_TRUE(areTheyEqual_h(c4,c0)) << "c4 point_dim: " << point_dim << " no_dims " << no_dims << " no_data " << no_data << std::endl;
				EXPECT_TRUE(areTheyEqual_h(c5,c0)) << "c5 point_dim: " << point_dim << " no_dims " << no_dims << " no_data " << no_data << std::endl;

				std::cout << "deleting now" << std::endl;
                for(int i = 0; i < c1.first->size(); i++){
                    delete c1.first->at(i);
                }
				delete c1.first;
				delete c1.second;
                for(int i = 0; i < c2.first->size(); i++){
                    delete c2.first->at(i);
                }
				delete c2.first;
				delete c2.second;
				for(int i = 0; i < c3.first->size(); i++){
					delete c3.first->at(i);
				}
				/*
				delete c3.second;
				for(int i = 0; i < c4.first->size(); i++){
					delete c4.first->at(i);
				}
				delete c4.second;
				*/
				for(int i = 0; i < c5.first->size(); i++){
					delete c5.first->at(i);
				}
				delete c5.second;

				for(int i = 0; i < data->size(); i++){
				    delete data->at(i);
				}
				delete data;

				delete centroids;
                for(int i = 0; i < dims->size(); i++){
                    delete dims->at(i);
                }
				delete dims;

				EXPECT_TRUE(areTheyEqual_h(c3,c0)) << " point_dim: " << point_dim << " no_dims " << no_dims << " no_data " << no_data << std::endl;


			}
		}
	}

}




