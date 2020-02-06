#include <gtest/gtest.h>
#include <iostream>
#include "../src/DOC/HyperCube.h"
#include <vector>


TEST(testHyperCube, testConstructor){
	auto p = new std::vector<float>;
	p->push_back(0.1);
	p->push_back(1.1);
	auto c = HyperCube(p, 1.1);
	SUCCEED();
}

TEST(testHyperCube, testConstructor2){
	auto p = new std::vector<float>;
	p->push_back(0.1);
	p->push_back(1.1);
	auto d = new std::vector<bool>;
	d->push_back(true);
	d->push_back(false);
	auto c = HyperCube(p, 1.1, d);
	SUCCEED();
	EXPECT_FLOAT_EQ(c.getWidth(), 1.1);
	auto d2 = c.getDimmensions();
	EXPECT_EQ(d2->at(0),true);
	EXPECT_EQ(d2->at(1), false);
	auto d3 = c.getCentroid();
	EXPECT_FLOAT_EQ(d3->at(0), 0.1);
	EXPECT_FLOAT_EQ(d3->at(1), 1.1);
	
	EXPECT_EQ(c.getDimmension(0), true);
	EXPECT_EQ(c.getDimmension(1), false);

	c.setDimmension(true,1);
	c.setDimmension(false,0);
	EXPECT_EQ(c.getDimmension(1), true);
	EXPECT_EQ(c.getDimmension(0), false);
}


TEST(testHyperCube, testPointContained){
	auto p = new std::vector<float>;
	p->push_back(20);
	p->push_back(20);
	auto d = new std::vector<bool>;
	d->push_back(true);
	d->push_back(false);
	auto c = HyperCube(p, 5, d);
	SUCCEED();
	
	std::vector<float>* goodPoint1 = new std::vector<float>{20, 20};
	std::vector<float>* goodPoint2 = new std::vector<float>{20, 5};
	std::vector<float>* goodPoint3 = new std::vector<float>{17, 30};
	
	std::vector<float>* badPoint1 = new std::vector<float>{5, 5};
	std::vector<float>* badPoint2 = new std::vector<float>{26, 10};
	
	EXPECT_TRUE(c.pointContained(goodPoint1));
	EXPECT_TRUE(c.pointContained(goodPoint2));
	EXPECT_TRUE(c.pointContained(goodPoint3));
	
	EXPECT_FALSE(c.pointContained(badPoint1));
	EXPECT_FALSE(c.pointContained(badPoint2));
	
	c.setDimmension(false,0);
	
	EXPECT_TRUE(c.pointContained(goodPoint1));
	EXPECT_TRUE(c.pointContained(goodPoint2));
	EXPECT_TRUE(c.pointContained(goodPoint3));
	
	EXPECT_TRUE(c.pointContained(badPoint1));
	EXPECT_TRUE(c.pointContained(badPoint2));
	
	c.setDimmension(true,0);
	c.setDimmension(true,1);
	
	EXPECT_TRUE(c.pointContained(goodPoint1));
	EXPECT_FALSE(c.pointContained(goodPoint2));
	EXPECT_FALSE(c.pointContained(goodPoint3));
	
	EXPECT_FALSE(c.pointContained(badPoint1));
	EXPECT_FALSE(c.pointContained(badPoint2));
}


TEST(testHyperCube, testPointContained2){
	std::vector<float>* centroid = new std::vector<float>{20, 20, 20};
	std::vector<float>* point1 = new std::vector<float>{10, 10, 10};
	
	std::vector<bool>* dimms = new std::vector<bool>{true, false};
	float width = 5;
	auto c = HyperCube(centroid, width, dimms);
	SUCCEED();
	
	EXPECT_TRUE(c.pointContained(centroid));
	EXPECT_FALSE(c.pointContained(point1));
	
}


TEST(testHyperCube, testPointContained3){

	std::vector<float>* centroid  = new std::vector<float>{10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
	std::vector<float>* point1  = new std::vector<float> {10, 10, 10, 10, 10, 10, 10, 10, 10, 10};
	std::vector<float>* point2  = new std::vector<float>{0, 10, 10, 10, 10, 10, 10, 10, 10, 10};
	std::vector<float>* point3  = new std::vector<float>{0, 20, 30, 40, 50, 60, 70, 80, 90, 100};
	
	std::vector<bool>* dimms = new std::vector<bool>{true, false};
	float width = 5;
	auto c = HyperCube(centroid, width, dimms);
	SUCCEED();
	
	EXPECT_TRUE(c.pointContained(centroid));
	EXPECT_TRUE(c.pointContained(point1));
	EXPECT_FALSE(c.pointContained(point2));
	EXPECT_FALSE(c.pointContained(point3));
}



TEST(testHyperCube, testPointContained4){

	std::vector<float>* centroid = new std::vector<float>{-10, 20, 30, -40, 50, 60, 70, -80, -90, 100};
	std::vector<float>* point1   = new std::vector<float>{-10, 10, -10, 10, 10, 10, -10, 10, 10, 10};
	std::vector<float>* point2   = new std::vector<float>{0, -10, 10, 10, 10, 10, 10, 10, 10, 10};
	std::vector<float>* point3   = new std::vector<float>{0, 20, -30, 40, 50, 60, 70, 80, 90, 100};
	
	std::vector<bool>* dimms = new std::vector<bool>{true, false};
	float width = 5;
	auto c = HyperCube(centroid, width, dimms);
	SUCCEED();
	
	EXPECT_TRUE(c.pointContained(centroid));
	EXPECT_TRUE(c.pointContained(point1));
	EXPECT_FALSE(c.pointContained(point2));
	EXPECT_FALSE(c.pointContained(point3));
}
