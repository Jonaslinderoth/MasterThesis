/*
 * DOCGPU.cpp
 *
 *  Created on: Feb 14, 2020
 *      Author: mikkel
 */

#include <src/DOC_GPU/DOCGPU.h>


# define CUDA_CALL ( x) do { if (( x) != cudaSuccess ) { \
	printf (" Error at % s :% d\ n" , __FILE__ , __LINE__ ) ;\
return EXIT_FAILURE ;}} while (0)
# define CURAND_CALL ( x) do { if (( x) != CURAND_STATUS_SUCCESS ) { \
printf (" Error at % s :% d\ n" , __FILE__ , __LINE__ ) ;\
return EXIT_FAILURE ;}} while (0)

DOCGPU::DOCGPU() {
	// TODO Auto-generated constructor stub

}

unsigned int* DOCGPU::cudaRandomNumberArray(const size_t lenght ,const curandGenerator_t* gen, unsigned int* array) {
	//if the array was not given
	if(array == nullptr){
		/*allocate that array.*/
		cudaMalloc(( void **) & array , lenght * sizeof ( unsigned int )) ;
	}

	/* Generate n floats on device */
	curandGenerate( *gen , array , lenght);
	return array;
}

DOCGPU::~DOCGPU() {
	// TODO Auto-generated destructor stub
}


bool DOCGPU::generateRandomSubSets(DataReader* dataReader){

	size_t size= 100;
	curandGenerator_t gen ;

	//Create pseudo - random number generator
	curandCreateGenerator(&gen ,CURAND_RNG_PSEUDO_MTGP32 );


	//seed the generator
	curandSetPseudoRandomGeneratorSeed(gen,1345);


	unsigned int* randomNumberArray_d = cudaRandomNumberArray( size , &gen );
	unsigned int* hostData;

	// Allocate n floats on host
	hostData = ( unsigned int*) calloc (size , sizeof ( unsigned int) );


	// Copy device memory to host
	cudaMemcpy ( hostData , randomNumberArray_d , size * sizeof ( unsigned int ) ,cudaMemcpyDeviceToHost );

	// Cleanup

	cudaFree( randomNumberArray_d );
	free( hostData );

	curandDestroyGenerator(gen);

	return true;
}
