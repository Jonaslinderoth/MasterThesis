#include <gtest/gtest.h>
#include "../src/DOC_GPU/MemSolver.h"
#include "../src/DOC_GPU/DOCGPU_Kernels.h"
#include <curand.h>
TEST(testMemSolver, testSetup){
	auto res = MemSolver::computeForAllocations(5, 100000, 19, 11090, 5, 5, 10000000000);

	EXPECT_EQ(res.size_of_data, 100000*5*sizeof(float));
	EXPECT_EQ(res.size_of_centroids, 19*sizeof(float));
	EXPECT_EQ(res.size_of_randomStates, 1024*10 * sizeof(curandState));
	EXPECT_EQ(res.size_of_bestDims, 100000+1);

	size_t sum = 0;
	sum += res.size_of_data;
	sum += res.size_of_samples;
	sum += res.size_of_centroids;
	sum += res.size_of_findDim;
	sum += res.size_of_findDim_count;
	sum += res.size_of_pointsContained;
	sum += res.size_of_pointsContained_count;
	sum += res.size_of_score;
	sum += res.size_of_index;
	sum += res.size_of_randomStates;
	sum += res.size_of_bestDims;
	EXPECT_LE(sum, 10000000000);
	
}


TEST(testMemSolver, testSetup2){
	auto res = MemSolver::computeForAllocations(5, 100000, 19, 11090, 5, 5, 2000000000);

	EXPECT_EQ(res.size_of_data, 100000*5*sizeof(float));
	EXPECT_EQ(res.size_of_centroids, 19*sizeof(float));
	EXPECT_EQ(res.size_of_randomStates, 1024*10 * sizeof(curandState));
	EXPECT_EQ(res.size_of_bestDims, 100000+1);

	size_t sum = 0;
	sum += res.size_of_data;
	sum += res.size_of_samples;
	sum += res.size_of_centroids;
	sum += res.size_of_findDim;
	sum += res.size_of_findDim_count;
	sum += res.size_of_pointsContained;
	sum += res.size_of_pointsContained_count;
	sum += res.size_of_score;
	sum += res.size_of_index;
	sum += res.size_of_randomStates;
	sum += res.size_of_bestDims;
	EXPECT_LE(sum, 2000000000);

	auto res2 = MemSolver::computeForArrays(res, 5, 100000, 19, 11090, 5);
	EXPECT_FLOAT_EQ(res2.number_of_centroids_f, 1.79311);
	EXPECT_EQ(res2.number_of_centroids, 2);


	res2 = MemSolver::computeForArrays(res, 5, 1000, 19, 11090, 5);
	EXPECT_EQ(res2.number_of_centroids_f, 19);
	EXPECT_EQ(res2.number_of_centroids, 19);	
}

