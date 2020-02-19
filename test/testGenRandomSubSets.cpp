#include <gtest/gtest.h>
#include <iostream>
#include "../src/DOC_GPU/DOCGPU_Kernels.h"
#include "../src/DOC/DOC.h"
#include "../src/DOC_GPU/DOCGPU.h"




TEST(testGenRandomSubSets, testGenerateRandomArray){
	size_t size= 100;
	curandGenerator_t gen ;

	//Create pseudo - random number generator
	curandCreateGenerator(&gen ,CURAND_RNG_PSEUDO_MTGP32 );


	//seed the generator
	curandSetPseudoRandomGeneratorSeed(gen,1345);
	DOCGPU docgpu;

	unsigned int* randomNumberArray_d = docgpu.cudaRandomNumberArray( size , &gen );
	unsigned int* hostData;

	// Allocate n floats on host
	hostData = ( unsigned int*) calloc (size , sizeof ( unsigned int) );


	// Copy device memory to host
	cudaMemcpy ( hostData , randomNumberArray_d , size * sizeof ( unsigned int ) ,cudaMemcpyDeviceToHost );

	// sum result
	unsigned int sum = 0;
	for (int i = 0; i < size; i ++) {
   
		sum += hostData [i];
	}

	
	// Cleanup

	cudaFree( randomNumberArray_d );
	free( hostData );

	curandDestroyGenerator(gen);

	EXPECT_TRUE(1867906863==sum);
}






