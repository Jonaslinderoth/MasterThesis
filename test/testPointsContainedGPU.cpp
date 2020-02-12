#include <gtest/gtest.h>
#include <iostream>
#include "../src/DOC_GPU/HyperCube.h"
 

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
	auto d = new std::vector<float>;
	auto c = pointsContained(a,b,d);
	SUCCEED();
	EXPECT_EQ(c->size(), a->size());
}

TEST_F(testPointsContainedGPU, DISABLED_testWithSimpleData){
	//TODO
	auto a = new std::vector<std::vector<bool>*>;
	auto aa = new std::vector<bool>{true, false};
	a->push_back(aa);

	auto b = new std::vector<std::vector<float>*>;
    auto bb = new std::vector<float>{9,9};
b->push_back(bb);
auto bbb = new std::vector<float>{111,111};
b->push_back(bbb);

    auto centroid = new std::vector<float>{10,10};
	auto c = pointsContained(a,b,centroid);
	SUCCEED();
	EXPECT_EQ(c->size(), a->size());
EXPECT_EQ(c->at(0)->at(0), true);
EXPECT_EQ(c->at(0)->at(1), false);
}
