#include <gtest/gtest.h>
#include <vector>
#include <random>
#include "../src/randomCudaScripts/Utils.h"
#include "../src/Fast_DOCGPU/whatDataInCentroid.h"


TEST(testWhatDataIsInCentroid, testSetupNaive){

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



TEST(testWhatDataIsInCentroid, testSetupChunks){

	std::vector<bool>* dims = new std::vector<bool>{0,0,1};
	std::vector<std::vector<float>*>* data = new std::vector<std::vector<float>*>;

	auto point1 = new std::vector<float>{1,1,1};
	auto point2 = new std::vector<float>{1,1,999};
	data->push_back(point1);
	data->push_back(point2);

	auto res = whatDataIsInCentroidTester(dims, data,0, 10, ChunksContained);
	EXPECT_EQ(res->at(0), true);
	EXPECT_EQ(res->at(1), false);
}


TEST(testWhatDataIsInCentroid, testSetupFew){

	std::vector<bool>* dims = new std::vector<bool>{0,0,1};
	std::vector<std::vector<float>*>* data = new std::vector<std::vector<float>*>;

	auto point1 = new std::vector<float>{1,1,1};
	auto point2 = new std::vector<float>{1,1,999};
	data->push_back(point1);
	data->push_back(point2);

	auto res = whatDataIsInCentroidTester(dims, data,0, 10, FewDimsContained);
	EXPECT_EQ(res->at(0), true);
	EXPECT_EQ(res->at(1), false);
}

TEST(testWhatDataIsInCentroid, test4DimsChunks){

	std::vector<bool>* dims = new std::vector<bool>{0,0,1,1};
	std::vector<std::vector<float>*>* data = new std::vector<std::vector<float>*>;

	auto point1 = new std::vector<float>{1,1,1,1};
	auto point2 = new std::vector<float>{1,1,999,1};
	data->push_back(point1);
	data->push_back(point2);

	auto res = whatDataIsInCentroidTester(dims, data,0, 10, ChunksContained);
	EXPECT_EQ(res->at(0), true);
	EXPECT_EQ(res->at(1), false);
}

TEST(testWhatDataIsInCentroid, test4DimsFew){

	std::vector<bool>* dims = new std::vector<bool>{0,0,1,1};
	std::vector<std::vector<float>*>* data = new std::vector<std::vector<float>*>;

	auto point1 = new std::vector<float>{1,1,1,1};
	auto point2 = new std::vector<float>{1,1,999,1};
	data->push_back(point1);
	data->push_back(point2);

	auto res = whatDataIsInCentroidTester(dims, data,0, 10, FewDimsContained);
	EXPECT_EQ(res->at(0), true);
	EXPECT_EQ(res->at(1), false);
}

TEST(testWhatDataIsInCentroid, test7DimsChunks){

	std::vector<bool>* dims = new std::vector<bool>{0,0,1,1,0,0,1};
	std::vector<std::vector<float>*>* data = new std::vector<std::vector<float>*>;

	auto point1 = new std::vector<float>{1,1,1,1,99,99,1};
	auto point2 = new std::vector<float>{1,1,999,1,999,99,99};
	auto point3 = new std::vector<float>{1,1,1,1,-99,-99,1};
	data->push_back(point1);
	data->push_back(point2);
	data->push_back(point3);

	auto res = whatDataIsInCentroidTester(dims, data,0, 10, ChunksContained);
	EXPECT_EQ(res->at(0), true);
	EXPECT_EQ(res->at(1), false);
	EXPECT_EQ(res->at(2), true);
}


TEST(testWhatDataIsInCentroid, test8DimsChunks){

	std::vector<bool>* dims = new std::vector<bool>{0,0,1,1,0,0,1,1};
	std::vector<std::vector<float>*>* data = new std::vector<std::vector<float>*>;

	auto point1 = new std::vector<float>{1,1,1,1,99,99,1,1};
	auto point2 = new std::vector<float>{1,1,999,1,999,99,99,1};
	auto point3 = new std::vector<float>{1,1,1,1,-99,-99,1,1};
	data->push_back(point1);
	data->push_back(point2);
	data->push_back(point3);

	auto res = whatDataIsInCentroidTester(dims, data,0, 10, ChunksContained);
	EXPECT_EQ(res->at(0), true);
	EXPECT_EQ(res->at(1), false);
	EXPECT_EQ(res->at(2), true);
}


TEST(testWhatDataIsInCentroid, test9DimsChunks){

	std::vector<bool>* dims = new std::vector<bool>{0,0,1,1,0,0,1,1,1};
	std::vector<std::vector<float>*>* data = new std::vector<std::vector<float>*>;

	auto point1 = new std::vector<float>{1,1,1  ,1, 99, 99, 1,1,1};
	auto point2 = new std::vector<float>{1,1,999,1,999, 99,99,1,1};
	auto point3 = new std::vector<float>{1,1,1  ,1,-99,-99, 1,1,999};
	data->push_back(point1);
	data->push_back(point2);
	data->push_back(point3);

	auto res = whatDataIsInCentroidTester(dims, data,0, 10, ChunksContained);
	EXPECT_EQ(res->at(0), true);
	EXPECT_EQ(res->at(1), false);
	EXPECT_EQ(res->at(2), false);
}

TEST(testWhatDataIsInCentroid, test9DimsNaive){

	std::vector<bool>* dims = new std::vector<bool>{0,0,1,1,0,0,1,1,1};
	std::vector<std::vector<float>*>* data = new std::vector<std::vector<float>*>;

	auto point1 = new std::vector<float>{1,1,1  ,1, 99, 99, 1,1,1};
	auto point2 = new std::vector<float>{1,1,999,1,999, 99,99,1,1};
	auto point3 = new std::vector<float>{1,1,1  ,1,-99,-99, 1,1,999};
	data->push_back(point1);
	data->push_back(point2);
	data->push_back(point3);

	auto res = whatDataIsInCentroidTester(dims, data,0, 10, NaiveContained);
	EXPECT_EQ(res->at(0), true);
	EXPECT_EQ(res->at(1), false);
	EXPECT_EQ(res->at(2), false);
}

TEST(testWhatDataIsInCentroid, test13DimsChunks){

	std::vector<bool>* dims = new std::vector<bool>{0,0,1,1,0,0,1,1,1,0,0,0,1};
	std::vector<std::vector<float>*>* data = new std::vector<std::vector<float>*>;

	auto point1 = new std::vector<float>{1,1,1,1,99,99,1,1,1,1,1,1,1};
	auto point2 = new std::vector<float>{1,1,999,1,999,99,99,1,1,1,1,1,1};
	auto point3 = new std::vector<float>{1,1,1,1,-99,-99,1,1,9,999,999,999,1};
	data->push_back(point1);
	data->push_back(point2);
	data->push_back(point3);

	auto res = whatDataIsInCentroidTester(dims, data,0, 10, ChunksContained);
	EXPECT_EQ(res->at(0), true);
	EXPECT_EQ(res->at(1), false);
	EXPECT_EQ(res->at(2), true);
}


TEST(testWhatDataIsInCentroid, test13Dims){

	std::vector<bool>* dims = new std::vector<bool>{0,0,1,1,0,0,1,1,1,0,0,0,1};
	std::vector<std::vector<float>*>* data = new std::vector<std::vector<float>*>;

	auto point1 = new std::vector<float>{1,1,1,1,99,99,1,1,1,1,1,1,1};
	auto point2 = new std::vector<float>{1,1,999,1,999,99,99,1,1,1,1,1,1};
	auto point3 = new std::vector<float>{1,1,1,1,-99,-99,1,1,9,999,999,999,1};
	data->push_back(point1);
	data->push_back(point2);
	data->push_back(point3);

	auto res = whatDataIsInCentroidTester(dims, data,0, 10);
	EXPECT_EQ(res->at(0), true);
	EXPECT_EQ(res->at(1), false);
	EXPECT_EQ(res->at(2), true);
}




TEST(testWhatDataIsInCentroid, testRandom100points10dims){
	unsigned int numPoint = 100;
	unsigned int dim = 10;
	std::default_random_engine generator;
	generator.seed(100);
	std::uniform_real_distribution<float> distribution(-25.0,25.0);


	
	std::vector<bool>* dims = new std::vector<bool>;
	for(unsigned int i = 0; i < dim; i++){
		dims->push_back(distribution(generator) < -20);
	}
	
	std::vector<std::vector<float>*>* data = new std::vector<std::vector<float>*>;
	for(unsigned int i = 0; i < numPoint; i++){
		auto point = new std::vector<float>;
		for(unsigned int j = 0; j < dim; j++){
			point->push_back(distribution(generator));
		}
		data->push_back(point);
	}


	auto resNaive = whatDataIsInCentroidTester(dims, data,0, 10);
	auto resChunk = whatDataIsInCentroidTester(dims, data,0, 10,ChunksContained);
	auto resFew = whatDataIsInCentroidTester(dims, data,0, 10,FewDimsContained);
	EXPECT_EQ(resNaive->size(), resChunk->size());
	EXPECT_EQ(resNaive->size(), resFew->size());
	for(unsigned int i = 0; i < resNaive->size(); i++){
		EXPECT_EQ(resNaive->at(i), resChunk->at(i));
		EXPECT_EQ(resNaive->at(i), resFew->at(i));
	}
}


TEST(testWhatDataIsInCentroid, testRandom1000points100dims){
	unsigned int numPoint = 1000;
	unsigned int dim = 100;
	std::default_random_engine generator;
	generator.seed(100);
	std::uniform_real_distribution<float> distribution(-25.0,25.0);


	
	std::vector<bool>* dims = new std::vector<bool>;
	for(unsigned int i = 0; i < dim; i++){
		dims->push_back(distribution(generator) < -24);
	}
	
	std::vector<std::vector<float>*>* data = new std::vector<std::vector<float>*>;
	for(unsigned int i = 0; i < numPoint; i++){
		auto point = new std::vector<float>;
		for(unsigned int j = 0; j < dim; j++){
			point->push_back(distribution(generator));
		}
		data->push_back(point);
	}


	auto resNaive = whatDataIsInCentroidTester(dims, data,0, 10);
	auto resChunk = whatDataIsInCentroidTester(dims, data,0, 10,ChunksContained);
	auto resFew = whatDataIsInCentroidTester(dims, data,0, 10,FewDimsContained);
	EXPECT_EQ(resNaive->size(), resChunk->size());
	EXPECT_EQ(resNaive->size(), resFew->size());
	for(unsigned int i = 0; i < resNaive->size(); i++){
		EXPECT_EQ(resNaive->at(i), resChunk->at(i));
		EXPECT_EQ(resNaive->at(i), resFew->at(i));
	}
}


TEST(testWhatDataIsInCentroid, testRandom2000points100dims){
	unsigned int numPoint = 2000;
	unsigned int dim = 100;
	std::default_random_engine generator;
	generator.seed(1010);
	std::uniform_real_distribution<float> distribution(-25.0,25.0);


	
	std::vector<bool>* dims = new std::vector<bool>;
	for(unsigned int i = 0; i < dim; i++){
		dims->push_back(distribution(generator) < -24);
	}
	
	std::vector<std::vector<float>*>* data = new std::vector<std::vector<float>*>;
	for(unsigned int i = 0; i < numPoint; i++){
		auto point = new std::vector<float>;
		for(unsigned int j = 0; j < dim; j++){
			point->push_back(distribution(generator));
		}
		data->push_back(point);
	}


	auto resNaive = whatDataIsInCentroidTester(dims, data,0, 10);
	auto resChunk = whatDataIsInCentroidTester(dims, data,0, 10,ChunksContained);
	auto resFew = whatDataIsInCentroidTester(dims, data,0, 10,FewDimsContained);
	auto resLess = whatDataIsInCentroidTester(dims, data,0, 10,LessReadingContained);
	auto resBreak = whatDataIsInCentroidTester(dims, data,0, 10,LessReadingBreakContained);
	EXPECT_EQ(resNaive->size(), resChunk->size());
	EXPECT_EQ(resNaive->size(), resFew->size());
	EXPECT_EQ(resNaive->size(), resLess->size());
	EXPECT_EQ(resNaive->size(), resBreak->size());
	for(unsigned int i = 0; i < resNaive->size(); i++){
		EXPECT_EQ(resNaive->at(i), resChunk->at(i));
		EXPECT_EQ(resNaive->at(i), resFew->at(i));
		EXPECT_EQ(resNaive->at(i), resLess->at(i));
		EXPECT_EQ(resNaive->at(i), resBreak->at(i));
	}
}

TEST(testWhatDataIsInCentroid, testRandom2000points100dimsNaive){
	unsigned int numPoint = 20000;
	unsigned int dim = 1000;
	std::default_random_engine generator;
	generator.seed(1010);
	std::uniform_real_distribution<float> distribution(-25.0,25.0);


	
	std::vector<bool>* dims = new std::vector<bool>;
	for(unsigned int i = 0; i < dim; i++){
		dims->push_back(distribution(generator) < -24);
	}
	
	std::vector<std::vector<float>*>* data = new std::vector<std::vector<float>*>;
	for(unsigned int i = 0; i < numPoint; i++){
		auto point = new std::vector<float>;
		for(unsigned int j = 0; j < dim; j++){
			point->push_back(distribution(generator));
		}
		data->push_back(point);
	}


	auto resNaive = whatDataIsInCentroidTester(dims, data,0, 10);
}

TEST(testWhatDataIsInCentroid, testRandom2000points100dimsChunk){
	unsigned int numPoint = 20000;
	unsigned int dim = 1000;
	std::default_random_engine generator;
	generator.seed(1010);
	std::uniform_real_distribution<float> distribution(-25.0,25.0);


	
	std::vector<bool>* dims = new std::vector<bool>;
	for(unsigned int i = 0; i < dim; i++){
		dims->push_back(distribution(generator) < -24);
	}
	
	std::vector<std::vector<float>*>* data = new std::vector<std::vector<float>*>;
	for(unsigned int i = 0; i < numPoint; i++){
		auto point = new std::vector<float>;
		for(unsigned int j = 0; j < dim; j++){
			point->push_back(distribution(generator));
		}
		data->push_back(point);
	}


	auto resChunk = whatDataIsInCentroidTester(dims, data,0, 10,ChunksContained);
}

TEST(testWhatDataIsInCentroid, testRandom2000points100dimsFew){
	unsigned int numPoint = 20000;
	unsigned int dim = 1000;
	std::default_random_engine generator;
	generator.seed(1010);
	std::uniform_real_distribution<float> distribution(-25.0,25.0);


	
	std::vector<bool>* dims = new std::vector<bool>;
	for(unsigned int i = 0; i < dim; i++){
		dims->push_back(distribution(generator) < -24);
	}
	
	std::vector<std::vector<float>*>* data = new std::vector<std::vector<float>*>;
	for(unsigned int i = 0; i < numPoint; i++){
		auto point = new std::vector<float>;
		for(unsigned int j = 0; j < dim; j++){
			point->push_back(distribution(generator));
		}
		data->push_back(point);
	}



	auto resFew = whatDataIsInCentroidTester(dims, data,0, 10,FewDimsContained);
}


TEST(testWhatDataIsInCentroid, testRandom2000points100dimsLessReading){
	unsigned int numPoint = 20000;
	unsigned int dim = 1000;
	std::default_random_engine generator;
	generator.seed(1010);
	std::uniform_real_distribution<float> distribution(-25.0,25.0);


	
	std::vector<bool>* dims = new std::vector<bool>;
	for(unsigned int i = 0; i < dim; i++){
		dims->push_back(distribution(generator) < -24);
	}
	
	std::vector<std::vector<float>*>* data = new std::vector<std::vector<float>*>;
	for(unsigned int i = 0; i < numPoint; i++){
		auto point = new std::vector<float>;
		for(unsigned int j = 0; j < dim; j++){
			point->push_back(distribution(generator));
		}
		data->push_back(point);
	}



	auto resLess = whatDataIsInCentroidTester(dims, data,0, 10,LessReadingContained);
}

TEST(testWhatDataIsInCentroid, testRandom2000points100dimsLessReadingAndBreaking){
	unsigned int numPoint = 20000;
	unsigned int dim = 1000;
	std::default_random_engine generator;
	generator.seed(1010);
	std::uniform_real_distribution<float> distribution(-25.0,25.0);


	
	std::vector<bool>* dims = new std::vector<bool>;
	for(unsigned int i = 0; i < dim; i++){
		dims->push_back(distribution(generator) < -24);
	}
	
	std::vector<std::vector<float>*>* data = new std::vector<std::vector<float>*>;
	for(unsigned int i = 0; i < numPoint; i++){
		auto point = new std::vector<float>;
		for(unsigned int j = 0; j < dim; j++){
			point->push_back(distribution(generator));
		}
		data->push_back(point);
	}



	auto resBreak = whatDataIsInCentroidTester(dims, data,0, 10,LessReadingBreakContained);
}
