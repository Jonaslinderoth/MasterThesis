#include <iostream>
#include <chrono>
#include <random>
#include <iostream>
#include "src/DOC_GPU/DOCGPU.h"
#include "src/Fast_DOCGPU/Fast_DOCGPU.h"
#include "src/Clustering.h"
#include "test/testData.h"
#include <vector>
#include <assert.h>
#include "src/dataReader/Cluster.h"
#include "src/MineClusGPU/MineClusGPU.h"
#include "src/Fast_DOC/Fast_DOC.h"
#include "src/Fast_DOCGPU/Fast_DOCGPU.h"
#include "src/randomCudaScripts/arrayEqual.h"



int main ()
{

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
				/*
				EXPECT_TRUE(areTheyEqual_h(c1,c0)) << "c1 point_dim: " << point_dim << " no_dims " << no_dims << " no_data " << no_data << std::endl;
				EXPECT_TRUE(areTheyEqual_h(c2,c0)) << "c2 point_dim: " << point_dim << " no_dims " << no_dims << " no_data " << no_data << std::endl;
				EXPECT_TRUE(areTheyEqual_h(c3,c0)) << "c3 point_dim: " << point_dim << " no_dims " << no_dims << " no_data " << no_data << std::endl;
				EXPECT_TRUE(areTheyEqual_h(c4,c0)) << "c4 point_dim: " << point_dim << " no_dims " << no_dims << " no_data " << no_data << std::endl;
				EXPECT_TRUE(areTheyEqual_h(c5,c0)) << "c5 point_dim: " << point_dim << " no_dims " << no_dims << " no_data " << no_data << std::endl;
				*/
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

				//EXPECT_TRUE(areTheyEqual_h(c3,c0)) << " point_dim: " << point_dim << " no_dims " << no_dims << " no_data " << no_data << std::endl;


			}
		}
	}

}
