#include "pointsContainedDevice.h"
#include "../randomCudaScripts/Utils.h"
#include "../randomCudaScripts/DeleteFromArray.h"
#include "../Fast_DOCGPU/whatDataInCentroid.h"
#include <assert.h>
#include <utility>
#include <vector>
#include <iostream>
#include "../randomCudaScripts/arrayEqual.h"

/*
 * This fuction returns if the points are in the hypercube made by the centroid by using a subset of the dimensions
 * It does not use anything fancy.
 */
__global__ void pointsContainedDeviceNaive(float* data, unsigned int* centroids, bool* dims, bool* output, unsigned int* Csum_out,
									  float width, unsigned int point_dim, unsigned int no_data, unsigned int no_dims, unsigned int m){
	// one kernel for each hypercube
	unsigned int entry = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int currentCentroid = entry/m;
	if(entry < no_dims){
		//assert(currentCentroid < no_of_ps);
		// for each data point
		unsigned int Csum = 0;
		for(unsigned int j = 0; j < no_data; j++){
			// for all dimmensions in each hypercube / point
			bool d = true;
			for(unsigned int i = 0; i < point_dim; i++){
				//(not (dims[entry*point_dim+i])) ||
				unsigned int centroid_index = centroids[currentCentroid];
				//if(!(centroid_index < no_data)){
				//    printf("num_data: %u, centroid_index: %u, currentCentroid: %u \n", no_data, centroid_index, currentCentroid);
				//}
				assert(centroid_index < no_data);
				assert(entry*point_dim+i < no_dims*point_dim);
				assert(centroid_index*point_dim+i < no_data*point_dim);
				assert(j*point_dim+i < no_data*point_dim);
				const unsigned long entryDims = entry*point_dim+i;
				const float centro = data[centroid_index*point_dim+i];
				const float punto = data[j*point_dim+i];
				const float abss = abs(centro - punto);
				d &= (not (dims[entryDims])) || (abss < width);
			}
			assert(entry < no_dims);
			assert((size_t)entry*(size_t)no_data+(size_t)j < (size_t)no_dims*(size_t)no_data+(size_t)j);
			output[(size_t)entry*(size_t)no_data+(size_t)j] = d;
			Csum += d;
		}
		Csum_out[entry] = Csum;

	}
}


/*
 * This fuction returns if the points are in the hypercube made by the centroid by using a subset of the dimensions
 * It does tries to break early every .
 */
__global__ void pointsContainedDeviceNaiveBreak(float* data,
											    unsigned int* centroids,
											    bool* dims,
											    bool* output,
											    unsigned int* Csum_out,
											    float width,
											    unsigned int point_dim,
											    unsigned int no_data,
											    unsigned int no_dims,
											    unsigned int m,
											    unsigned int breakingIntervall){
	// one kernel for each hypercub
	unsigned int entry = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int currentCentroid = entry/m;
	if(entry < no_dims){
		//assert(currentCentroid < no_of_ps);
		// for each data point
		unsigned int Csum = 0;
		for(unsigned int j = 0; j < no_data; j++){
			// for all dimmensions in each hypercube / point
			bool d = true;
			for(unsigned int i = 0; i < point_dim; i++){
				//(not (dims[entry*point_dim+i])) ||
				unsigned int centroid_index = centroids[currentCentroid];
				//if(!(centroid_index < no_data)){
				//    printf("num_data: %u, centroid_index: %u, currentCentroid: %u \n", no_data, centroid_index, currentCentroid);
				//}
				
				assert(centroid_index < no_data);
				assert(entry*point_dim+i < no_dims*point_dim);
				assert(centroid_index*point_dim+i < no_data*point_dim);
				assert(j*point_dim+i < no_data*point_dim);
				const unsigned long entryDims = entry*point_dim+i;
				const float centro = data[centroid_index*point_dim+i];
				const float punto = data[j*point_dim+i];
				const float abss = abs(centro - punto);
				d &= (not (dims[entryDims])) || (abss < width);
				if(i%breakingIntervall == 0 and (not d)){
					break;

				}
			}
			assert(entry < no_dims);
			assert((size_t)entry*(size_t)no_data+(size_t)j < (size_t)no_dims*(size_t)no_data+(size_t)j);
			output[(size_t)entry*(size_t)no_data+(size_t)j] = d;
			Csum += d;
		}
		Csum_out[entry] = Csum;

	}
}

/*
 * This does the same as the naive but it moved the data to shared memory before.
 * making
 *
 */
__global__ void pointsContainedDeviceSharedMemory(float* data,
		unsigned int* centroids,
		bool* dims,
		bool* output,
		unsigned int* Csum_out,
		float width,
		const unsigned int point_dim,
		const unsigned int no_data_p,
		const unsigned int no_dims,
		const unsigned int ammountOfSamplesThatUseSameCentroid,
		const unsigned int numberOfCentroids,
		const unsigned int centroidSharedMemorySize_f,
		const unsigned int dataSharedMemorySize_f,
		const unsigned int blocksWithSameCentroid){
	extern __shared__ float sharedMemory[];

	//shared memory spit
	float* centroidSharedMemory = sharedMemory;
	float* dataSharedMemory = (float*)&sharedMemory[centroidSharedMemorySize_f];


	//we have that every blocksWithSameCentroid the centroid chances to the next one.
	const unsigned int indexOfCentroidToCentroids = blockIdx.x/blocksWithSameCentroid;
	const unsigned int indexOfCentroidInDataNoDims_f = centroids[indexOfCentroidToCentroids]*point_dim;
	//we want to move the centroid to shared memory.
	// if(threadIdx.x < point_dim){  // OBS... This kernel will not work for dims higher than the block size
	// 	const long offsetByDimension_f = threadIdx.x;
	// 	const unsigned long indexOfCentroidInData_f = indexOfCentroidInDataNoDims_f + offsetByDimension_f;
	// 	const float partOfACentroid_f = data[indexOfCentroidInData_f];
	// 	centroidSharedMemory[offsetByDimension_f] = partOfACentroid_f;
	// }

	for(unsigned int i = 0; i < ceilf((float)point_dim/blockDim.x); i++){
		const unsigned int j = i*blockDim.x+threadIdx.x;
		if(j < point_dim){
			centroidSharedMemory[j] = data[indexOfCentroidInDataNoDims_f+j];
		}
	}

	// it is okay to not have 
	

	const unsigned long howLongOnM_p = ( blockIdx.x % blocksWithSameCentroid ) * blockDim.x + threadIdx.x;
	const unsigned long whatM = blockIdx.x / blocksWithSameCentroid;
	const unsigned long offEntry_p = howLongOnM_p + whatM * ammountOfSamplesThatUseSameCentroid;
	unsigned int Csum = 0;
	//_f stand for float , and _p stand for point.

	//now we need to "work" on the data
	// times the threads will need to copy data from global to shared memory.
	const unsigned long no_data_f = no_data_p*point_dim;
	const unsigned long dataSharedMemorySize_p = dataSharedMemorySize_f/point_dim;


	for(unsigned long i_p = 0 ; i_p < no_data_p ; i_p += dataSharedMemorySize_p){
		//copy the data from global to shared memory
		for(unsigned long indexCopy_f = 0 ; indexCopy_f < dataSharedMemorySize_f ; indexCopy_f+=blockDim.x){
			const unsigned long indexInSharedMemory_f = indexCopy_f + threadIdx.x;
			const unsigned long indexInData_f = i_p*point_dim + indexCopy_f +threadIdx.x;
			if(indexInData_f < no_data_f and indexInSharedMemory_f < dataSharedMemorySize_f ){
				dataSharedMemory[indexInSharedMemory_f] = data[indexInData_f];
			}
		}
		__syncthreads();

		const unsigned long ammountOfPointsLeft = no_data_p-i_p;
		for(unsigned long indexDataSM_p = 0 ; indexDataSM_p < dataSharedMemorySize_p ; indexDataSM_p++){
			const unsigned long indexDimsNoDimension_f = offEntry_p * point_dim;
			const unsigned long indexPointDataSM_p = indexDataSM_p;
			const unsigned long indexOutput_p = offEntry_p * no_data_p + i_p + indexDataSM_p;


			if( howLongOnM_p < ammountOfSamplesThatUseSameCentroid and indexDataSM_p < ammountOfPointsLeft and offEntry_p < no_dims)
			{

				bool d = true;

				for(unsigned long dimensionIndex_f = 0 ; dimensionIndex_f < point_dim ; dimensionIndex_f++){

					const unsigned long indexDataSM_f = indexPointDataSM_p*point_dim + dimensionIndex_f;
					const unsigned long indexDims_f = indexDimsNoDimension_f + dimensionIndex_f;
					const bool dim = dims[indexDims_f];
					const float cen = centroidSharedMemory[dimensionIndex_f];
					const float dat = dataSharedMemory[indexDataSM_f];
					d &= (not (dim)) || (abs(cen - dat) < width);

				}
				output[indexOutput_p] = d;
				Csum += d;
			}

		}
		__syncthreads();

	}

	if(offEntry_p < no_dims and howLongOnM_p < ammountOfSamplesThatUseSameCentroid){
		Csum_out[offEntry_p] = Csum;

	}

}



/*
 * This does the same as the naive but it moved the data to shared memory before.
 * this contained early breaking
 *
 */
__global__ void pointsContainedDeviceSharedMemoryBreak(float* data,
													   unsigned int* centroids,
													   bool* dims,
													   bool* output,
													   unsigned int* Csum_out,
													   float width,
													   const unsigned int point_dim,
													   const unsigned int no_data_p,
													   const unsigned int no_dims,
													   const unsigned int ammountOfSamplesThatUseSameCentroid,
													   const unsigned int numberOfCentroids,
													   const unsigned int centroidSharedMemorySize_f,
													   const unsigned int dataSharedMemorySize_f,
													   const unsigned int blocksWithSameCentroid,
													   const unsigned int breakingIntervall){
	extern __shared__ float sharedMemory[];

	//shared memory spit
	float* centroidSharedMemory = sharedMemory;
	float* dataSharedMemory = (float*)&sharedMemory[centroidSharedMemorySize_f];


	//we have that every blocksWithSameCentroid the centroid chances to the next one.
	const unsigned int indexOfCentroidToCentroids = blockIdx.x/blocksWithSameCentroid;
	const unsigned int indexOfCentroidInDataNoDims_f = centroids[indexOfCentroidToCentroids]*point_dim;
	//we want to move the centroid to shared memory.
	// if(threadIdx.x < point_dim){  // OBS... This kernel will not work for dims higher than the block size
	// 	const long offsetByDimension_f = threadIdx.x;
	// 	const unsigned long indexOfCentroidInData_f = indexOfCentroidInDataNoDims_f + offsetByDimension_f;
	// 	const float partOfACentroid_f = data[indexOfCentroidInData_f];
	// 	centroidSharedMemory[offsetByDimension_f] = partOfACentroid_f;
	// }

	for(unsigned int i = 0; i < ceilf((float)point_dim/blockDim.x); i++){
		const unsigned int j = i*blockDim.x+threadIdx.x;
		if(j < point_dim){
			centroidSharedMemory[j] = data[indexOfCentroidInDataNoDims_f+j];
		}
	}

	// it is okay to not have


	const unsigned long howLongOnM_p = ( blockIdx.x % blocksWithSameCentroid ) * blockDim.x + threadIdx.x;
	const unsigned long whatM = blockIdx.x / blocksWithSameCentroid;
	const unsigned long offEntry_p = howLongOnM_p + whatM * ammountOfSamplesThatUseSameCentroid;
	unsigned int Csum = 0;
	//_f stand for float , and _p stand for point.

	//now we need to "work" on the data
	// times the threads will need to copy data from global to shared memory.
	const unsigned long no_data_f = no_data_p*point_dim;
	const unsigned long dataSharedMemorySize_p = dataSharedMemorySize_f/point_dim;


	for(unsigned long i_p = 0 ; i_p < no_data_p ; i_p += dataSharedMemorySize_p){
		//copy the data from global to shared memory
		for(unsigned long indexCopy_f = 0 ; indexCopy_f < dataSharedMemorySize_f ; indexCopy_f+=blockDim.x){
			const unsigned long indexInSharedMemory_f = indexCopy_f + threadIdx.x;
			const unsigned long indexInData_f = i_p*point_dim + indexCopy_f +threadIdx.x;
			if(indexInData_f < no_data_f and indexInSharedMemory_f < dataSharedMemorySize_f ){
				dataSharedMemory[indexInSharedMemory_f] = data[indexInData_f];
			}
		}
		__syncthreads();

		const unsigned long ammountOfPointsLeft = no_data_p-i_p;
		for(unsigned long indexDataSM_p = 0 ; indexDataSM_p < dataSharedMemorySize_p ; indexDataSM_p++){
			const unsigned long indexDimsNoDimension_f = offEntry_p * point_dim;
			const unsigned long indexPointDataSM_p = indexDataSM_p;
			const unsigned long indexOutput_p = offEntry_p * no_data_p + i_p + indexDataSM_p;


			if( howLongOnM_p < ammountOfSamplesThatUseSameCentroid and indexDataSM_p < ammountOfPointsLeft and offEntry_p < no_dims)
			{

				bool d = true;

				for(unsigned long dimensionIndex_f = 0 ; dimensionIndex_f < point_dim ; dimensionIndex_f++){

					const unsigned long indexDataSM_f = indexPointDataSM_p*point_dim + dimensionIndex_f;
					const unsigned long indexDims_f = indexDimsNoDimension_f + dimensionIndex_f;
					const bool dim = dims[indexDims_f];
					const float cen = centroidSharedMemory[dimensionIndex_f];
					const float dat = dataSharedMemory[indexDataSM_f];
					d &= (not (dim)) || (abs(cen - dat) < width);
					if(dimensionIndex_f % breakingIntervall == 0 and (not d)){
						break;
					}

				}
				output[indexOutput_p] = d;
				Csum += d;
			}

		}
		__syncthreads();

	}

	if(offEntry_p < no_dims and howLongOnM_p < ammountOfSamplesThatUseSameCentroid){
		Csum_out[offEntry_p] = Csum;

	}

}

__global__ void pointsContainedDeviceSharedMemoryFewBank(float* __restrict__ data,
		unsigned int* __restrict__ centroids,
		bool* __restrict__ dims,
		bool* __restrict__ output,
		unsigned int* __restrict__ Csum_out,
		float width,
		const unsigned long point_dim,
		const unsigned long no_data_p,
		const unsigned long no_dims,
		const unsigned long ammountOfSamplesThatUseSameCentroid,
		const unsigned long numberOfCentroids,
		const unsigned long centroidSharedMemorySize_f,
		const unsigned long dataSharedMemorySize_f,
		const unsigned long blocksWithSameCentroid){
	extern __shared__ float sharedMemory[];

	//shared memory spit
	float* centroidSharedMemory = sharedMemory;
	float* dataSharedMemory = (float*)&sharedMemory[centroidSharedMemorySize_f];

	//we have that every blocksWithSameCentroid the centroid chances to the next one.
	const unsigned int indexOfCentroidToCentroids = blockIdx.x/blocksWithSameCentroid;
	const unsigned int indexOfCentroidInDataNoDims_f = centroids[indexOfCentroidToCentroids]*point_dim;
	//we want to move the centroid to shared memory.

	// // This only allows for medoids of upt to block
	// if(threadIdx.x < point_dim){
	// 	const long offsetByDimension_f = threadIdx.x;
	// 	const unsigned long indexOfCentroidInData_f = indexOfCentroidInDataNoDims_f + offsetByDimension_f;
	// 	const float partOfACentroid_f = data[indexOfCentroidInData_f];
	// 	centroidSharedMemory[offsetByDimension_f] = partOfACentroid_f;
	// }

	
	for(unsigned int i = 0; i < ceilf((float)point_dim/blockDim.x); i++){
		const unsigned int j = i*blockDim.x+threadIdx.x;
		if(j < point_dim){
			centroidSharedMemory[j] = data[indexOfCentroidInDataNoDims_f+j];
		}
	}
	//now i have the centroid in shared memory. but its going to cause bank conflicts :( , TODO fix this.

	const unsigned long howLongOnM_p = ( blockIdx.x % blocksWithSameCentroid ) * blockDim.x + threadIdx.x;
	const unsigned long offEntry_p = howLongOnM_p + ( blockIdx.x / blocksWithSameCentroid ) * ammountOfSamplesThatUseSameCentroid;
	unsigned int Csum = 0;
	//_f stand for float , and _p stand for point.

	//now we need to "work" on the data
	// times the threads will need to copy data from global to shared memory.
	const unsigned long no_data_f = no_data_p*point_dim;
	const unsigned long dataSharedMemorySize_p = dataSharedMemorySize_f/point_dim;
	//assert(dataSharedMemorySize_f%point_dim == 0);



	for(unsigned long i_p = 0 ; i_p < no_data_p ; i_p += dataSharedMemorySize_p){
		//copy the data from global to shared memory
		for(unsigned long indexCopy_f = 0 ; indexCopy_f < dataSharedMemorySize_f ; indexCopy_f+=blockDim.x){

			const unsigned long indexInSharedMemory_f = indexCopy_f + threadIdx.x;

			const unsigned long indexInData_f = i_p*point_dim + indexCopy_f +threadIdx.x;

			if(indexInData_f < no_data_f and indexInSharedMemory_f < dataSharedMemorySize_f){

				dataSharedMemory[indexInSharedMemory_f] = data[indexInData_f];

			}
		}
		__syncthreads();


		const unsigned long warpId_p = 0;//threadIdx.x/(32*point_dim);
		const unsigned long warpIdToDim_p =threadIdx.x/32;
		const unsigned long dataPointsLeft = no_data_p-i_p;
		const unsigned long limit = min(dataPointsLeft,dataSharedMemorySize_p);
		for(unsigned long indexDataSM_p = 0 ; indexDataSM_p < limit ; indexDataSM_p++)
		{

			//const unsigned long indexPointDataSM_p = indexDataSM_p;
			//const unsigned long indexOutput_p = offEntry_p * no_data_p + i_p + indexDataSM_p;
			const unsigned long indexDimsNoDimension_f = offEntry_p * point_dim;
			const unsigned long indexPointDataSM_p = ( indexDataSM_p + warpId_p ) % limit;
			const unsigned long indexPointDataSMNotOffset_p = indexDataSM_p;
			const long offset_p = indexPointDataSM_p - indexPointDataSMNotOffset_p;
			//const unsigned long indexDimsNoDimension_f = ( offEntry_p + offset_p ) * point_dim;
			const unsigned long indexOutput_p = offEntry_p * no_data_p + i_p + indexDataSM_p + offset_p;

			if( howLongOnM_p < ammountOfSamplesThatUseSameCentroid and offEntry_p < no_dims)
			{

				bool d = true;

				for(unsigned long dimensionIndex_f = 0 ; dimensionIndex_f < point_dim ; dimensionIndex_f++){
					/*
					const unsigned long indexDataSM_f = indexPointDataSM_p*point_dim + dimensionIndex_f;
					const unsigned long indexDims_f = indexDimsNoDimension_f + dimensionIndex_f;
					//assert(indexDims_f < no_dims*point_dim);
					const bool dim = dims[indexDims_f];
					const float cen = centroidSharedMemory[dimensionIndex_f];
					//assert(indexDataSM_f < dataSharedMemorySize_f);
					const float dat = dataSharedMemory[indexDataSM_f];
					d &= (not (dim)) || (abs(cen - dat) < width);
					*/

					const unsigned long offDimensionIndex = (dimensionIndex_f+warpIdToDim_p)%point_dim;
					const unsigned long indexDataSM_f = indexPointDataSM_p*point_dim + offDimensionIndex;
					const unsigned long indexDims_f = indexDimsNoDimension_f + offDimensionIndex;
					assert(indexDims_f < no_dims*point_dim);
					const bool dim = dims[indexDims_f];
					const float cen = centroidSharedMemory[offDimensionIndex];
					assert(indexDataSM_f < dataSharedMemorySize_f);
					const float dat = dataSharedMemory[indexDataSM_f];
					d &= (not (dim)) || (abs(cen - dat) < width);

				}

				//assert(indexOutput_p < ammountOfSamplesThatUseSameCentroid*numberOfCentroids*no_data_p);
				output[indexOutput_p] = d;

				Csum += d;
			}

		}
		__syncthreads();

	}
	if(offEntry_p < no_dims and howLongOnM_p < ammountOfSamplesThatUseSameCentroid){
		Csum_out[offEntry_p] = Csum;
	}

}




__global__ void pointsContainedDeviceSharedMemoryFewerBank(float* __restrict__ data,
		unsigned int* __restrict__ centroids,
		bool* __restrict__ dims,
		bool* __restrict__ output,
		unsigned int* __restrict__ Csum_out,
		float width,
		const unsigned long point_dim,
		const unsigned long no_data_p,
		const unsigned long no_dims,
		const unsigned long ammountOfSamplesThatUseSameCentroid,
		const unsigned long numberOfCentroids,
		const unsigned long centroidSharedMemorySize_f,
		const unsigned long dataSharedMemorySize_f,
		const unsigned long blocksWithSameCentroid){
	extern __shared__ float sharedMemory[];

	//shared memory spit
	float* centroidSharedMemory = sharedMemory;
	float* dataSharedMemory = (float*)&sharedMemory[centroidSharedMemorySize_f];

	//we have that every blocksWithSameCentroid the centroid chances to the next one.
	const unsigned int indexOfCentroidToCentroids = blockIdx.x/blocksWithSameCentroid;
	const unsigned int indexOfCentroidInDataNoDims_f = centroids[indexOfCentroidToCentroids]*point_dim;
	//we want to move the centroid to shared memory.
	if(threadIdx.x < point_dim){
		const long offsetByDimension_f = threadIdx.x;
		const unsigned long indexOfCentroidInData_f = indexOfCentroidInDataNoDims_f + offsetByDimension_f;
		const float partOfACentroid_f = data[indexOfCentroidInData_f];
		centroidSharedMemory[offsetByDimension_f] = partOfACentroid_f;
	}
	//now i have the centroid in shared memory. but its going to cause bank conflicts :( , TODO fix this.

	const unsigned long howLongOnM_p = ( blockIdx.x % blocksWithSameCentroid ) * blockDim.x + threadIdx.x;
	const unsigned long offEntry_p = howLongOnM_p + ( blockIdx.x / blocksWithSameCentroid ) * ammountOfSamplesThatUseSameCentroid;
	unsigned int Csum = 0;
	//_f stand for float , and _p stand for point.

	//now we need to "work" on the data
	// times the threads will need to copy data from global to shared memory.
	const unsigned long no_data_f = no_data_p*point_dim;
	const unsigned long dataSharedMemorySize_p = dataSharedMemorySize_f/point_dim;
	//assert(dataSharedMemorySize_f%point_dim == 0);



	for(unsigned long i_p = 0 ; i_p < no_data_p ; i_p += dataSharedMemorySize_p){
		//copy the data from global to shared memory
		for(unsigned long indexCopy_f = 0 ; indexCopy_f < dataSharedMemorySize_f ; indexCopy_f+=blockDim.x){

			const unsigned long indexInSharedMemory_f = indexCopy_f + threadIdx.x;

			const unsigned long indexInData_f = i_p*point_dim + indexCopy_f +threadIdx.x;

			if(indexInData_f < no_data_f and indexInSharedMemory_f < dataSharedMemorySize_f){

				dataSharedMemory[indexInSharedMemory_f] = data[indexInData_f];

			}
		}
		__syncthreads();


		const unsigned long warpId_p = threadIdx.x/(32*point_dim);
		const unsigned long warpIdToDim_p =threadIdx.x/32;
		const unsigned long dataPointsLeft = no_data_p-i_p;
		const unsigned long limit = min(dataPointsLeft,dataSharedMemorySize_p);
		for(unsigned long indexDataSM_p = 0 ; indexDataSM_p < limit ; indexDataSM_p++)
		{

			//const unsigned long indexPointDataSM_p = indexDataSM_p;
			//const unsigned long indexOutput_p = offEntry_p * no_data_p + i_p + indexDataSM_p;
			const unsigned long indexDimsNoDimension_f = offEntry_p * point_dim;
			const unsigned long indexPointDataSM_p = ( indexDataSM_p + warpId_p ) % limit;
			const unsigned long indexPointDataSMNotOffset_p = indexDataSM_p;
			const long offset_p = indexPointDataSM_p - indexPointDataSMNotOffset_p;
			//const unsigned long indexDimsNoDimension_f = ( offEntry_p + offset_p ) * point_dim;
			const unsigned long indexOutput_p = offEntry_p * no_data_p + i_p + indexDataSM_p + offset_p;

			if( howLongOnM_p < ammountOfSamplesThatUseSameCentroid and offEntry_p < no_dims)
			{

				bool d = true;

				for(unsigned long dimensionIndex_f = 0 ; dimensionIndex_f < point_dim ; dimensionIndex_f++){
					/*
					const unsigned long indexDataSM_f = indexPointDataSM_p*point_dim + dimensionIndex_f;
					const unsigned long indexDims_f = indexDimsNoDimension_f + dimensionIndex_f;
					//assert(indexDims_f < no_dims*point_dim);
					const bool dim = dims[indexDims_f];
					const float cen = centroidSharedMemory[dimensionIndex_f];
					//assert(indexDataSM_f < dataSharedMemorySize_f);
					const float dat = dataSharedMemory[indexDataSM_f];
					d &= (not (dim)) || (abs(cen - dat) < width);
					*/

					const unsigned long offDimensionIndex = (dimensionIndex_f+warpIdToDim_p)%point_dim;
					const unsigned long indexDataSM_f = indexPointDataSM_p*point_dim + offDimensionIndex;
					const unsigned long indexDims_f = indexDimsNoDimension_f + offDimensionIndex;
					//assert(indexDims_f < no_dims*point_dim);
					const bool dim = dims[indexDims_f];
					const float cen = centroidSharedMemory[offDimensionIndex];
					//assert(indexDataSM_f < dataSharedMemorySize_f);
					const float dat = dataSharedMemory[indexDataSM_f];
					d &= (not (dim)) || (abs(cen - dat) < width);

				}

				//assert(indexDataSM_p < dataSharedMemorySize_p);

				//assert(indexOutput_p < ammountOfSamplesThatUseSameCentroid*numberOfCentroids*no_data_p);
				output[indexOutput_p] = d;

				Csum += d;
			}

		}
		__syncthreads();

	}
	if(offEntry_p < no_dims and howLongOnM_p < ammountOfSamplesThatUseSameCentroid){
		Csum_out[offEntry_p] = Csum;

	}

}


__global__ void gpuNotBoolArray(bool* inputAndOutPut,
							 std::size_t lenght){
	const size_t idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx < lenght){
		inputAndOutPut[idx] = not inputAndOutPut[idx];
	}
}


/*
 * This fuction returns if the points are in the hypercube made by the centroid by using a subset of the dimensions
 * It does not use anything fancy.
 */
__global__ void pointsContainedDeviceNaiveFewPoints(float* data,
													unsigned int* centroids,
													bool* output,
													unsigned int* Csum_out,
													const float width,
													const unsigned int point_dim,
													const unsigned int no_data,
													const unsigned int m,
													const unsigned int numberOfCentroids){

	const unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
	const unsigned int currentCentroid = idx/m;

	//check that we are not going out of bound on the centroids array
	if(idx < 1){
		const unsigned int centroidIndex = centroids[currentCentroid];
		const float* centroidArray = &data[centroidIndex*point_dim];
		unsigned int Csum = 0;

		for(unsigned int dataIndex = 0; dataIndex < no_data; ++dataIndex){

			const float* pointArray = &data[dataIndex*point_dim];
			bool d = true;
			for(unsigned int dimensionIndex = 0; dimensionIndex < point_dim; ++dimensionIndex){
				const float centroid_f = centroidArray[dimensionIndex];
				const float data_f = pointArray[dimensionIndex];
				const float diference = abs(centroid_f - data_f);
				d &= (diference < width);
			}
			output[(size_t)idx*(size_t)no_data+(size_t)dataIndex] = d;
			Csum += d;
			//printf(" no_data: %u \n dataIndex: %u \n",no_data,dataIndex);
		}

		Csum_out[idx] = Csum;
	}
	/*
	// one kernel for each hypercube
	unsigned int entry = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int currentCentroid = entry/m;
	if(entry < no_dims){
		//assert(currentCentroid < no_of_ps);
		// for each data point
		unsigned int Csum = 0;
		for(unsigned int j = 0; j < no_data; j++){
			// for all dimmensions in each hypercube / point
			bool d = true;
			for(unsigned int i = 0; i < point_dim; i++){
				//(not (dims[entry*point_dim+i])) ||
				unsigned int centroid_index = centroids[currentCentroid];
				//if(!(centroid_index < no_data)){
				//    printf("num_data: %u, centroid_index: %u, currentCentroid: %u \n", no_data, centroid_index, currentCentroid);
				//}
				assert(centroid_index < no_data);
				assert(entry*point_dim+i < no_dims*point_dim);
				assert(centroid_index*point_dim+i < no_data*point_dim);
				assert(j*point_dim+i < no_data*point_dim);
				const unsigned long entryDims = entry*point_dim+i;
				const float centro = data[centroid_index*point_dim+i];
				const float punto = data[j*point_dim+i];
				const float abss = abs(centro - punto);
				d &= (not (dims[entryDims])) || (abss < width);
			}
			assert(entry < no_dims);
			assert((size_t)entry*(size_t)no_data+(size_t)j < (size_t)no_dims*(size_t)no_data+(size_t)j);
			output[(size_t)entry*(size_t)no_data+(size_t)j] = d;
			Csum += d;
		}
		Csum_out[entry] = Csum;
	}
	*/
}



void pointsContainedKernelNaive(unsigned int dimGrid,
							    unsigned int dimBlock,
							    cudaStream_t stream,
							    float* data,
							    unsigned int* centroids,
							    bool* dims, bool* output,
							    unsigned int* Csum_out,
							    float width,
							    unsigned int point_dim,
							    unsigned int no_data,
							    unsigned int no_dims,
							    unsigned int m){

	pointsContainedDeviceNaive<<<dimGrid, dimBlock, 0, stream>>>(data, centroids, dims,
												 output, Csum_out,
												 width, point_dim, no_data, no_dims, m);


};

void pointsContainedKernelNaiveBreak(unsigned int dimGrid,
									 unsigned int dimBlock,
									 cudaStream_t stream,
									 float* data,
									 unsigned int* centroids,
									 bool* dims,
									 bool* output,
									 unsigned int* Csum_out,
									 float width,
									 unsigned int point_dim,
									 unsigned int no_data,
									 unsigned int no_dims,
									 unsigned int m,
									 unsigned int breakingIntervall){

	pointsContainedDeviceNaiveBreak<<<dimGrid, dimBlock, 0, stream>>>(data,
																	  centroids,
																	  dims,
																	  output,
																	  Csum_out,
																	  width,
																	  point_dim,
																	  no_data,
																	  no_dims,
																	  m,
																	  breakingIntervall);
}


void pointsContainedKernelSharedMemory(unsigned int dimGrid,
									   unsigned int dimBlock,
									   cudaStream_t stream,
									   float* data,
									   unsigned int* centroids,
									   bool* dims,
									   bool* output,
									   unsigned int* Csum_out,
									   float width,
									   unsigned int point_dim,
									   unsigned int no_data,
									   unsigned int number_of_samples,
									   unsigned int m,
									   unsigned int numberOfCentroids,
									   unsigned int maxSharedmemory){
	dimBlock = 64;
	//we block takes care of only on centroid. a centroid is made of point_dim floats
	unsigned long centroidSharedMemorySize_f = point_dim;
	//how many blocks needed to cover m.
	unsigned long blocksWithSameCentroid = ceil((float)m/dimBlock);
	//we need blocksWithSameCentroid per centroid to cover all the sampels
	unsigned long dimGridv2 = blocksWithSameCentroid*numberOfCentroids;

	//need to know how much shared memory we are going to use.
	unsigned long dataSharedMemorySize_f = maxSharedmemory/sizeof(float)-centroidSharedMemorySize_f;
	dataSharedMemorySize_f = (dataSharedMemorySize_f/point_dim);
	dataSharedMemorySize_f = dataSharedMemorySize_f*point_dim;

	unsigned long sharedMemorySize_f = dataSharedMemorySize_f+centroidSharedMemorySize_f;

	pointsContainedDeviceSharedMemory<<<dimGridv2, dimBlock, sharedMemorySize_f*sizeof(float), stream>>>(data,
																								 centroids,
																								 dims,
																								 output,
																								 Csum_out,
																								 width,
																								 point_dim,
																								 no_data,
																								 number_of_samples,
																								 m,
																								 numberOfCentroids,
																								 centroidSharedMemorySize_f,
																								 dataSharedMemorySize_f,
																								 blocksWithSameCentroid);

};


void pointsContainedKernelSharedMemoryBreak(unsigned int dimGrid,
									   	    unsigned int dimBlock,
									   	    cudaStream_t stream,
									   	    float* data,
									   	    unsigned int* centroids,
									   	    bool* dims,
									   	    bool* output,
									   	    unsigned int* Csum_out,
									   	    float width,
									   	    unsigned int point_dim,
									   	    unsigned int no_data,
									   	    unsigned int number_of_samples,
									   	    unsigned int m,
									   	    unsigned int numberOfCentroids,
									   	    unsigned int breakingIntervall){
	dimBlock = 64;
	unsigned long maxSharedmemory = 48000; //48kb can probably go up to more but...
	//we block takes care of only on centroid. a centroid is made of point_dim floats
	unsigned long centroidSharedMemorySize_f = point_dim;
	//how many blocks needed to cover m.
	unsigned long blocksWithSameCentroid = ceil((float)m/dimBlock);
	//we need blocksWithSameCentroid per centroid to cover all the sampels
	unsigned long dimGridv2 = blocksWithSameCentroid*numberOfCentroids;

	//need to know how much shared memory we are going to use.
	unsigned long dataSharedMemorySize_f = maxSharedmemory/sizeof(float)-centroidSharedMemorySize_f;
	dataSharedMemorySize_f = (dataSharedMemorySize_f/point_dim);
	dataSharedMemorySize_f = dataSharedMemorySize_f*point_dim;

	unsigned long sharedMemorySize_f = dataSharedMemorySize_f+centroidSharedMemorySize_f;


	pointsContainedDeviceSharedMemoryBreak<<<dimGridv2, dimBlock, sharedMemorySize_f*sizeof(float), stream>>>(data,
																								              centroids,
																								              dims,
																								              output,
																								              Csum_out,
																								              width,
																								              point_dim,
																								              no_data,
																								              number_of_samples,
																								              m,
																								              numberOfCentroids,
																								              centroidSharedMemorySize_f,
																								              dataSharedMemorySize_f,
																								              blocksWithSameCentroid,
																								              breakingIntervall);

};

/*
 * dimBlock needs to be multiple of 32.
 * dimBlock needs to be >= then point_dim
 */
void pointsContainedKernelSharedMemoryFewBank(unsigned int dimGrid,
											  unsigned int dimBlock,
											  cudaStream_t stream,
											  float* data, unsigned int* centroids,
											  bool* dims,
											  bool* output,
											  unsigned int* Csum_out,
											  float width,
											  unsigned int point_dim,
											  unsigned int no_data,
											  unsigned int number_of_samples,
											  unsigned int m,
											  unsigned int numberOfCentroids){
	dimBlock = 64;
	unsigned long maxSharedmemory = 48000; //48kb can probably go up to more but...
	//we block takes care of only on centroid. a centroid is made of point_dim floats
	unsigned long centroidSharedMemorySize_f = point_dim;
	//how many blocks needed to cover m.
	unsigned long blocksWithSameCentroid = ceil((float)m/dimBlock);
	//we need blocksWithSameCentroid per centroid to cover all the sampels
	unsigned long dimGridv2 = blocksWithSameCentroid*numberOfCentroids;

	//need to know how much shared memory we are going to use.
	unsigned long dataSharedMemorySize_f = maxSharedmemory/sizeof(float)-centroidSharedMemorySize_f;
	dataSharedMemorySize_f = (dataSharedMemorySize_f/point_dim);
	dataSharedMemorySize_f = dataSharedMemorySize_f*point_dim;

	unsigned long sharedMemorySize_f = dataSharedMemorySize_f+centroidSharedMemorySize_f;

	//std::cout << "no_data_p " << no_data << std::endl << "point_dim " << point_dim << std::endl;
	/*//calculate how much we are going to use
	unsigned long long sharedMemorySize = ( centroidSharedMemorySize +dataSharedMemorySize )*sizeof(float);
	*/
	//std::cout << "sharedMemorySize " << dataSharedMemorySize << std::endl;

	//std::cout << "dimGridv2*dimBlock*20: " << dimGridv2*dimBlock*20 << std::endl;
	pointsContainedDeviceSharedMemoryFewBank<<<dimGridv2, dimBlock, sharedMemorySize_f*sizeof(float), stream>>>(data, centroids, dims,
												 output, Csum_out,
												 width, point_dim, no_data, number_of_samples, m, numberOfCentroids,centroidSharedMemorySize_f,dataSharedMemorySize_f,blocksWithSameCentroid);
	//std::cout << "done with kernel call" << std::endl;
};

void pointsContainedKernelSharedMemoryFewerBank(unsigned int dimGrid,
												unsigned int dimBlock,
												cudaStream_t stream,
												float* data,
												unsigned int* centroids,
												bool* dims,
												bool* output,
												unsigned int* Csum_out,
												float width,
												unsigned int point_dim,
												unsigned int no_data,
												unsigned int number_of_samples,
												unsigned int m,
												unsigned int numberOfCentroids){
	dimBlock = 64;

	unsigned long maxSharedmemory = 48000; //48kb can probably go up to more but...
	//we block takes care of only on centroid. a centroid is made of point_dim floats
	unsigned long centroidSharedMemorySize_f = point_dim;
	//how many blocks needed to cover m.
	unsigned long blocksWithSameCentroid = ceil((float)m/dimBlock);
	//we need blocksWithSameCentroid per centroid to cover all the sampels
	unsigned long dimGridv2 = blocksWithSameCentroid*numberOfCentroids;

	//need to know how much shared memory we are going to use.
	unsigned long dataSharedMemorySize_f = maxSharedmemory/sizeof(float)-centroidSharedMemorySize_f;
	dataSharedMemorySize_f = (dataSharedMemorySize_f/point_dim);
	dataSharedMemorySize_f = dataSharedMemorySize_f*point_dim;

	unsigned long sharedMemorySize_f = dataSharedMemorySize_f+centroidSharedMemorySize_f;

	//std::cout << "no_data_p " << no_data << std::endl << "point_dim " << point_dim << std::endl;
	/*//calculate how much we are going to use
	unsigned long long sharedMemorySize = ( centroidSharedMemorySize +dataSharedMemorySize )*sizeof(float);
	*/
	//std::cout << "sharedMemorySize " << dataSharedMemorySize << std::endl;

	//std::cout << "dimGridv2*dimBlock*20: " << dimGridv2*dimBlock*20 << std::endl;
	pointsContainedDeviceSharedMemoryFewerBank<<<dimGridv2, dimBlock, sharedMemorySize_f*sizeof(float)>>>(data, centroids, dims,
												 output, Csum_out,
												 width, point_dim, no_data, number_of_samples, m, numberOfCentroids,centroidSharedMemorySize_f,dataSharedMemorySize_f,blocksWithSameCentroid);
	//std::cout << "done with kernel call" << std::endl;
};



void notBoolArray(unsigned int dimBlock,
				  cudaStream_t stream,
				  bool* imputAndOutput,
				  std::size_t lenght){

	unsigned int dimGrid = lenght/dimBlock;
	if(lenght%dimBlock != 0){
		dimGrid++;
	}

	gpuNotBoolArray<<<dimGrid,dimBlock,0,stream>>>(imputAndOutput,lenght);
}





std::pair<std::vector<std::vector<bool>*>*,std::vector<unsigned int>*> pointsContained(std::vector<std::vector<bool>*>* dims,
																					   std::vector<std::vector<float>*>* data,
																					   std::vector<unsigned int>* centroids,
																					   int m, float width){
	

	// Calculaating sizes
	int point_dim = data->at(0)->size();
	int no_of_points = data->size();
	int no_of_dims = dims->size();
	int no_of_centroids = centroids->size();

	int floats_in_data = point_dim * no_of_points;
	int bools_in_dims = no_of_dims * point_dim;
	int bools_in_output = no_of_points * no_of_dims;
	int ints_in_output_count = no_of_dims;

	int size_of_data = floats_in_data*sizeof(float);
	int size_of_dims = bools_in_dims*sizeof(bool);
	int size_of_centroids = no_of_centroids*sizeof(unsigned int);
	int size_of_output = bools_in_output*sizeof(bool);
	int size_of_output_count = ints_in_output_count*sizeof(unsigned int);

	// allocating on the host
	float* data_h = (float*) malloc(size_of_data);
	bool* dims_h = (bool*) malloc(size_of_dims);
	unsigned int* centroids_h = (unsigned int*) malloc(size_of_centroids);
	bool* output_h = (bool*) malloc(size_of_output);
	unsigned int* output_count_h = (unsigned int*) malloc(size_of_output_count);

	// filling data array
	for(int i= 0; i < no_of_points; i++){
		for(int j = 0; j < point_dim; j++){
			data_h[i*point_dim+j] = data->at(i)->at(j);
		}
	}

	// filling dims array
	for(int i= 0; i < no_of_dims; i++){
		for(int j = 0; j < point_dim; j++){
			dims_h[i*point_dim+j] = dims->at(i)->at(j);
		}
	}

	// filling centroid array
	for(int i= 0; i < no_of_centroids; i++){
		centroids_h[i] = centroids->at(i);
	}

	// allocating on device
	float* data_d;
	bool* dims_d;
	unsigned int* centroids_d;
	bool* output_d;
	unsigned int* output_count_d;

	cudaMalloc((void **) &data_d, size_of_data);
	cudaMalloc((void **) &dims_d, size_of_dims);
	cudaMalloc((void **) &centroids_d, size_of_centroids);
	cudaMalloc((void **) &output_d, size_of_output);
	cudaMalloc((void **) &output_count_d, size_of_output_count);

	//Copy from host to device

	cudaMemcpy(data_d, data_h, size_of_data, cudaMemcpyHostToDevice);
	cudaMemcpy(dims_d, dims_h, size_of_dims, cudaMemcpyHostToDevice);
	cudaMemcpy(centroids_d, centroids_h, size_of_centroids, cudaMemcpyHostToDevice);


	//Call kernel
	pointsContainedDeviceNaive<<<ceil((no_of_dims)/256.0), 256>>>(data_d,
																  centroids_d,
																  dims_d,
																  output_d,
																  output_count_d,
																  width, point_dim, no_of_points, no_of_dims, m);

	// pointsContainedWrapper(ceil((no_of_dims)/256.0), 256, NULL,
	// 					   data_d,
	// 					   centroids_d,
	// 					   dims_d,
	// 					   output_d,
	// 					   output_count_d,
	// 					   width, point_dim, no_of_points, no_of_dims, m,pointContainedSmem
	// 					   );


	// copy from device
	cudaMemcpy(output_h, output_d, size_of_output, cudaMemcpyDeviceToHost);
	cudaMemcpy(output_count_h, output_count_d, size_of_output_count, cudaMemcpyDeviceToHost);

	// construnct output
	auto output =  new std::vector<std::vector<bool>*>;
	auto output_count =  new std::vector<unsigned int>;


	for(int i = 0; i < no_of_dims; i++){
		auto a =  new std::vector<bool>;
		for(int j = 0; j < no_of_points; j++){
			a->push_back(output_h[i*no_of_points+j]);
		}
		output->push_back(a);
	}

	for(int i = 0; i < no_of_dims; i++){
		output_count->push_back(output_count_h[i]);
	}


	cudaFree(data_d);
	cudaFree(dims_d);
	cudaFree(centroids_d);
	cudaFree(output_d);
	cudaFree(output_count_d);
	free(data_h);
	free(dims_h);
	free(centroids_h);
	free(output_h);
	free(output_count_h);


	return std::make_pair(output,output_count);
};



void pointsContainedWrapper(unsigned int dimGrid,
							unsigned int dimBlock,
							cudaStream_t stream,
							float* data,
							unsigned int* centroids,
							bool* dims,
							bool* output,
							unsigned int* Csum_out,
							float width,
							unsigned int point_dim,
							unsigned int no_data,
							unsigned int no_dims,
							unsigned int m,
							pointContainedType type
							){

	if(type == pointContainedNaive){
		pointsContainedKernelNaive(dimGrid,
								   dimBlock,
								   stream,
								   data,
								   centroids,
								   dims,
								   output,
								   Csum_out,
								   width,
								   point_dim,
								   no_data,
								   no_dims,
								   m);
	}else if(type == pointContainedSmem){
		pointsContainedKernelSharedMemory(dimGrid,
										  dimBlock,
										  stream,
										  data,
										  centroids,
										  dims,
										  output,
										  Csum_out,
										  width,
										  point_dim,
										  no_data,
										  no_dims,
										  m,
										  ceilf((float)no_dims/m));
	}
};





