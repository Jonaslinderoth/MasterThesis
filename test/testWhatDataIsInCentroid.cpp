#include <gtest/gtest.h>
#include <vector>
#include "../src/randomCudaScripts/Utils.h"
#include "../src/DOC_GPU/whatDataInCentroid.h"


TEST(testWhatDataIsInCentroid, testSetup){

	std::vector<bool>* dims = new std::vector<bool>{0,0,1};
	std::vector<std::vector<float>*>* data = new std::vector<std::vector<float>*>;

	auto point1 = new std::vector<float>{1,1,1};
	auto point2 = new std::vector<float>{1,1,999};
	data->push_back(point1);
	data->push_back(point2);

	auto res = whatDataIsInCentroidTester(dims, data,0, 10);
	EXPECT_EQ(res->at(0), true);
	EXPECT_EQ(res->at(1), false);
}
