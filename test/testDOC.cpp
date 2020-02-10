#include <gtest/gtest.h>
#include <iostream>
#include "../src/DOC/DOC.h"
#include <vector>
#include "testData.h"
#include <cmath>



TEST(testDOC, testConstructor){
	DOC d = DOC();
	d.setSeed(1);
	SUCCEED();
	EXPECT_EQ(d.size(), 0);
}


TEST(testDOC, testConstructor2){
	DOC d = DOC(new std::vector<std::vector<float>*>);
	d.setSeed(1);
	SUCCEED();
	EXPECT_EQ(d.size(), 0);
}


TEST(testDOC, testConstructor3){
	auto v = new std::vector<std::vector<float>*>;
	auto v1 = new std::vector<float>;
	v1->push_back(1.1);
	v1->push_back(1.2);
	v1->push_back(1.3);
	v->push_back(v1);
	DOC d = DOC(v);
	d.setSeed(1);
	SUCCEED();
	EXPECT_EQ(d.size(), 1);
}

TEST(testDOC, testAddPoint){
	DOC d = DOC(new std::vector<std::vector<float>*>);
	d.setSeed(1);
	SUCCEED();
	EXPECT_EQ(d.size(), 0);
	auto v1 = new std::vector<float>;
		v1->push_back(1.1);
		v1->push_back(.2);
		v1->push_back(1.3);
	auto v2 = new std::vector<float>;
		v2->push_back(2.1);
		v2->push_back(2.2);
		v2->push_back(2.3);
	EXPECT_TRUE(d.addPoint(v1));
	EXPECT_EQ(d.size(), 1);
	EXPECT_TRUE(d.addPoint(v2));
	EXPECT_EQ(d.size(), 2);
}

TEST(testDOC, testFindDimensions){
	DOC d = DOC(new std::vector<std::vector<float>*>);
	d.setSeed(1);
	
	std::vector<float> centroid{20, 20, 20};
	std::vector<float> point1{20, 20, 20};
	std::vector<float> point2{20, 10, 20};
	std::vector<float> point3{20, 10, 20};
	std::vector<float> point4{20, 10, 10};

	auto v = new std::vector<std::vector<float>*>;
	v->push_back(&point1);
	v->push_back(&point2);
	v->push_back(&point3);
	v->push_back(&point4);

	auto res = d.findDimensions(&centroid, v,5);
	EXPECT_TRUE(res->at(0));
	EXPECT_FALSE(res->at(1));
	EXPECT_FALSE(res->at(2));
}


TEST(testDOC, testFindDimensions2){
	DOC d = DOC();
	d.setSeed(1);
	std::vector<float> centroid{20, 20, 20};
	std::vector<float> point1{20, 20, 20};
	std::vector<float> point2{20, 10, 20};
	std::vector<float> point3{20, 10, 20};
	std::vector<float> point4{20, 10, 10};
	std::vector<float> point5{10, 10, 10};

	auto v = new std::vector<std::vector<float>*>;
	v->push_back(&point1);
	v->push_back(&point2);
	v->push_back(&point3);
	v->push_back(&point4);
	v->push_back(&point5);

	auto res = d.findDimensions(&centroid, v,5);
	EXPECT_FALSE(res->at(0));
	EXPECT_FALSE(res->at(1));
	EXPECT_FALSE(res->at(2));

}


TEST(testDOC, testFindCluster){
	std::vector<std::vector<float>*>* data = new std::vector<std::vector<float>*>;
	for(float i = 0; i < 200; i++){
		for(float j = 9; j < 15; j++){
			std::vector<float>* point1 = new std::vector<float>{j,i};
			data->push_back(point1);

		}
	}
	DOC d = DOC(data, 0.1, 0.25, 5);
	d.setSeed(1);
	auto res = d.findCluster();
	SUCCEED();




	/* int i = 0;
	   for(int i = 0; i < res.first.size(); i++){
		std::cout << res.first.at(i).at(0) << ", ";
	}
	std::cout << std::endl;

	i = 0;
	for(int i = 0; i < res.first.size(); i++){
		std::cout << res.first.at(i).at(1) << ", ";
	}
	std::cout << std::endl;
	*/

	EXPECT_TRUE(res.second->at(0));
	EXPECT_FALSE(res.second->at(1));

	int count = 0;
	EXPECT_EQ(res.first->size(), 1200);
}




TEST(testDOC, testFindCluster2){
	std::vector<std::vector<float>*>* data = new std::vector<std::vector<float>*>;
	int a = 0;
	for(float i = 9; i <= 12; i++){
		for(float j = 9; j <= 12; j++){
			std::vector<float>* point1 = new std::vector<float>{i,j};
			data->push_back(point1);
			a++;
		}
	}

	int b = 0;
	for(float i = 60; i <= 65; i++){
		for(float j = 0; j <= 50; j++){
			std::vector<float>* point1 = new std::vector<float>{i,j};
			data->push_back(point1);
			b++;
		}
	}
	//std::cout << a << ", " << b << std::endl;

	DOC d = DOC(data, 0.1, 0.25, 5);
	d.setSeed(1);
	auto res = d.findCluster();
	SUCCEED();


	EXPECT_TRUE(res.second->at(0));
	EXPECT_FALSE(res.second->at(1));

	EXPECT_EQ(res.first->size(), 306);

	for(int i = 0; i< data->size(); i++){
		delete data->at(i);
	}
	delete data;
	delete res.first;
	delete res.second;
}

TEST(testDOC, testMu){
	auto d = DOC();
	EXPECT_EQ(d.mu(1,1), 4);
	EXPECT_EQ(d.mu(2,2), 32);
	EXPECT_EQ(d.mu(3,2), 48);
	EXPECT_EQ(d.mu(3,3), 192);
	EXPECT_EQ(d.mu(3,4), 768);
}


TEST(testDOC, testFindCluster3){
	std::vector<std::vector<float>*>* data = data_4dim2cluster();

	//std::cout << a << ", " << b << std::endl;

	DOC d = DOC(data, 0.1, 0.25, 5);
	d.setSeed(1);
	auto res = d.findCluster();
	SUCCEED();


	EXPECT_TRUE(res.second->at(0));
	EXPECT_FALSE(res.second->at(1));
	EXPECT_FALSE(res.second->at(2));
	EXPECT_FALSE(res.second->at(3));

	EXPECT_LT(abs((int)res.first->size()-(int)numPoints_4dim2cluster()), 10);

	for(int i = 0; i< data->size(); i++){
		delete data->at(i);
	}
	delete data;
	delete res.first;
	delete res.second;
}


TEST(tesDOC, testFindKClusters){
	std::vector<std::vector<float>*>* data = data_4dim2cluster();


	DOC d = DOC(data, 0.1, 0.25, 5);
	d.setSeed(1);
	std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> > res = d.findKClusters(0);
	
	SUCCEED();
	EXPECT_EQ(res.size(), 0);

}

TEST(tesDOC, testFindKClusters2){
	std::vector<std::vector<float>*>* data = data_4dim2cluster();


	DOC d = DOC(data, 0.1, 0.25, 5);
	d.setSeed(1);
	std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> > res = d.findKClusters(1);
	
	SUCCEED();
	EXPECT_EQ(res.size(), 1);
	EXPECT_TRUE(res.at(0).second->at(0));
	EXPECT_FALSE(res.at(0).second->at(1));
	EXPECT_FALSE(res.at(0).second->at(2));
	EXPECT_FALSE(res.at(0).second->at(3));

	EXPECT_LT(abs((int)res.at(0).first->size()-(int)numPoints_4dim2cluster()), 10);

	for(int i = 0; i< data->size(); i++){
		delete data->at(i);
	}
	delete data;
	delete res.at(0).first;
	delete res.at(0).second;
}

bool pointEQ(std::vector<float>* a1, std::vector<float>* a2){
	bool output = true;
	EXPECT_EQ(a1->size(), a2->size());
	
	for(int i = 0; i < a1->size(); i++){
		output = false;
		for(int j = 0; j < a2->size(); j++){
			auto b1 = a1->at(i);
			auto b2 = a2->at(j);
			output = b1 == b2;
			if(output){
				break;
			}
		}
		EXPECT_TRUE(output);			
	}
}

bool disjoint(std::vector<std::vector<float>*>* a1, std::vector<std::vector<float>*>* a2){
	bool output = true;

	for(int i = 0; i < a1->size(); i++){
		for(int j = 0; j < a2->size(); j++){
			auto b1 = a1->at(i);
			auto b2 = a2->at(j);
			output = b1 == b2;
			EXPECT_FALSE(b1 == b2);			
		}
	}
	return output;
}


bool equal(std::vector<std::vector<float>*>* a1, std::vector<std::vector<float>*>* a2){
	bool output = true;
	EXPECT_EQ(a1->size(), a2->size());
	
	for(int i = 0; i < a1->size(); i++){
		output = false;
		for(int j = 0; j < a2->size(); j++){
			auto b1 = a1->at(i);
			auto b2 = a2->at(j);
			output = b1 == b2;
			if(output){
				break;
			}
		}
		EXPECT_TRUE(output);			
	}
}



TEST(testDOC, testFindKClusters3){
	std::vector<std::vector<float>*>* data = data_4dim2cluster();


	DOC d = DOC(data, 0.1, 0.25, 6);
	d.setSeed(1);
	std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> > res = d.findKClusters(2);
	
	SUCCEED();
	EXPECT_EQ(res.size(), 1);
	EXPECT_TRUE(res.at(0).second->at(0));
	EXPECT_FALSE(res.at(0).second->at(1));
	EXPECT_FALSE(res.at(0).second->at(2));
	EXPECT_FALSE(res.at(0).second->at(3));


	EXPECT_LT(abs((int)res.at(0).first->size()-397), 10);

	
	for(int i = 0; i< data->size(); i++){
		delete data->at(i);
	}
	delete data;
	delete res.at(0).first;
	delete res.at(0).second;
}


TEST(testDOC, testFindKClusters5){
	auto data = data_4dim2cluster();


	std::vector<std::vector<float>*>* data1 = new std::vector<std::vector<float>*>;
	std::vector<std::vector<float>*>* data2 = new std::vector<std::vector<float>*>;	
	std::vector<std::vector<float>*>* data3 = new std::vector<std::vector<float>*>;

	
	for(int i = 0; i < data->size(); i++){
		data1->push_back(data->at(i));
		data2->push_back(data->at(i));
		data3->push_back(data->at(i));
	}
	
	DOC d1 = DOC(data1, 0.4, 0.25, 3);
	DOC d2 = DOC(data2, 0.4, 0.25, 3);
	DOC d3 = DOC(data3, 0.4, 0.25, 3);


	d1.setSeed(1);
	d2.setSeed(1);
	d3.setSeed(1);
	
	
	std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> > res1 = d1.findKClusters(2);
	std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> > res2 = d2.findKClusters(2);
	std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> > res3 = d3.findKClusters(2);
	
	SUCCEED();
	

	equal(res1.at(0).first, res1.at(0).first);
	equal(res1.at(0).first, res2.at(0).first);
	equal(res1.at(0).first, res3.at(0).first);
}




TEST(testDOC, testFindKClusters4){

	std::vector<std::vector<float>*>* data = new std::vector<std::vector<float>*>;
	auto a = new std::vector<float>{0,1};
	data->push_back(a);


	DOC d = DOC(data, 0.1, 0.25, 6);
	std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> > res = d.findKClusters(2);
	
	SUCCEED();
	EXPECT_EQ(res.size(), 1);
	EXPECT_TRUE(res.at(0).second->at(0));
	EXPECT_TRUE(res.at(0).second->at(1));



	
	for(int i = 0; i< data->size(); i++){
		delete data->at(i);
	}
	delete data;
	delete res.at(0).first;
	delete res.at(0).second;
}


/*TEST(testDOC2, test1){
	auto cluster = new std::vector<float>{1, 4, 9};
	auto data = new std::vector<float>{1,2,3,4,5,6,7,8,9};
	int head = cluster->size()-1;		
	for(int j = data->size()-1; j >=0  ;j-- ){
		if (data->at(j) == cluster->at(head)){
			auto temp = data->at(j);
			data->at(j) = data->at(data->size()-1);
			data->at(data->size()-1) = temp;
			data->pop_back();
			head--;
		}
		//std::cout << "got here3" << std::endl;
	}
	//7 2 3 8 5 6
		
	EXPECT_EQ(data->at(0), 7);
	EXPECT_EQ(data->at(1), 2);
	EXPECT_EQ(data->at(2), 3);
	EXPECT_EQ(data->at(3), 8);
	EXPECT_EQ(data->at(4), 5);
	EXPECT_EQ(data->at(5), 6);
	
	}*/
