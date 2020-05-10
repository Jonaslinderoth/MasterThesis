#include <gtest/gtest.h>
#include "../src/DOC_GPU/MemSolver.h"
#include "../src/Fast_DOCGPU/MemSolver_Fast_DOCUnified.h"
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


	auto res3 = MemSolver::computeCentroidSizeForAllocation(res, 5, 100000, 19, 11090, 5);
	EXPECT_FLOAT_EQ(res3, 1.6982849);


	auto res2 = MemSolver::computeCentroidSizeForAllocation(res, 5, 1000, 19, 11090, 5);
	EXPECT_FLOAT_EQ(res2, 19);
		
}

TEST(testMemSolverFastDOC, testSetup1){
	auto res = MemSolver_Fast_DOCUnified::computeForAllocations(5, 100, 19, 5^2, 5, 5, 200000);

	EXPECT_EQ(res.size_of_data, 100*5*sizeof(float));
	EXPECT_EQ(res.size_of_centroids, 19*sizeof(unsigned int));
	EXPECT_EQ(res.size_of_randomStates, 1024*2 * sizeof(curandState));
	EXPECT_EQ(res.size_of_bestDims, 101*sizeof(bool));

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
	
	EXPECT_LE(sum, 200000);

	EXPECT_EQ(res.first_number_of_centroids, 19);
	
	auto res3 = MemSolver_Fast_DOCUnified::computeCentroidSizeForAllocation(res, 5, 100, 19, 5^2, 5);
	EXPECT_FLOAT_EQ(res3, 19);


	auto res2 = MemSolver_Fast_DOCUnified::computeCentroidSizeForAllocation(res, 5, 1000, 19, 5^2, 5);
	EXPECT_FLOAT_EQ(res2, 19);

	auto res1 = MemSolver_Fast_DOCUnified::computeCentroidSizeForAllocation(res, 5, 10, 19, 5^2, 5);
	EXPECT_FLOAT_EQ(res1, 19);	
}



TEST(testMemSolverFastDOC, testSetup1_2){
	auto res = MemSolver_Fast_DOCUnified::computeForAllocations(5, 100, 19, 5^2, 5, 5, 20000);

	EXPECT_EQ(res.size_of_data, 100*5*sizeof(float));
	EXPECT_EQ(res.size_of_centroids, 19*sizeof(unsigned int));
	EXPECT_EQ(res.size_of_randomStates, 1024*2 * sizeof(curandState));
	EXPECT_EQ(res.size_of_bestDims, 101*sizeof(bool));

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
	
	EXPECT_LE(sum, 200000);

	EXPECT_EQ(res.first_number_of_centroids, 19);
	
	auto res3 = MemSolver_Fast_DOCUnified::computeCentroidSizeForAllocation(res, 5, 100, 19, 5^2, 5);
	EXPECT_FLOAT_EQ(res3, 19);


	auto res2 = MemSolver_Fast_DOCUnified::computeCentroidSizeForAllocation(res, 5, 1000, 19, 5^2, 5);
	EXPECT_FLOAT_EQ(res2, 19);

	auto res1 = MemSolver_Fast_DOCUnified::computeCentroidSizeForAllocation(res, 5, 10, 19, 5^2, 5);
	EXPECT_FLOAT_EQ(res1, 19);	
}


TEST(testMemSolverFastDOC, testSetup2){
	auto res = MemSolver_Fast_DOCUnified::computeForAllocations(5, 100000, 19, 11090, 5, 5, 2000000000);

	EXPECT_EQ(res.size_of_data, 100000*5*sizeof(float));
	EXPECT_EQ(res.size_of_centroids, 19*sizeof(float));
	EXPECT_EQ(res.size_of_randomStates, 1024*2 * sizeof(curandState));
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


	auto res3 = MemSolver_Fast_DOCUnified::computeCentroidSizeForAllocation(res, 5, 100000, 19, 11090, 5);
	EXPECT_FLOAT_EQ(res3, 19);


	auto res2 = MemSolver_Fast_DOCUnified::computeCentroidSizeForAllocation(res, 5, 1000, 19, 11090, 5);
	EXPECT_FLOAT_EQ(res2, 19);	
}

TEST(testMemSolverFastDOC, testSetup3){
	auto res = MemSolver_Fast_DOCUnified::computeForAllocations(1000, 2000000, 19, 11090, 5, 5, (size_t)4*1000*1000*1000);

	EXPECT_EQ(res.size_of_data, (size_t)2000000*1000*sizeof(float));
	EXPECT_EQ(res.size_of_centroids, 19*sizeof(float));
	EXPECT_EQ(res.size_of_randomStates, 1024*2 * sizeof(curandState));
	EXPECT_EQ(res.size_of_bestDims, 2000000+1);
	EXPECT_EQ(res.size_of_pointsContained, 2000000+1);
	

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
	EXPECT_LE(sum, (size_t)4*1000*1000*1000*2*2);


	auto res3 = MemSolver_Fast_DOCUnified::computeCentroidSizeForAllocation(res, 1000, 2000000, 19, 11090, 5);
	EXPECT_FLOAT_EQ(res3, 19);


	auto res2 = MemSolver_Fast_DOCUnified::computeCentroidSizeForAllocation(res, 1000, 2000000, 19, 11090, 5);
	EXPECT_FLOAT_EQ(res2, 19);
		
}

