/*
 * pointsContainedDevice.h
 *
 *  Created on: Apr 13, 2020
 *      Author: mikkel
 */

#ifndef POINTSCONTAINEDDEVICE_H_
#define POINTSCONTAINEDDEVICE_H_


/*
 * this does not have data in shared memory
 */
void pointsContainedKernelNaive(unsigned int dimGrid,
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
								unsigned int m);

/*
 * this fuction has the data and the centroids in Shared Memory
 * the centroid and at least one data points need to be able to fit into shared memory to being able to work
 */
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
		unsigned int no_dims,
		unsigned int m,
		unsigned int numberOfCentroids);
/*
 * This fuction has the data and the centroids in Shared Memory and no bank conflicts
 * the centroid and at least dimBlock/32 data points need to be able to fit into shared memory to being able to work
 */
void pointsContainedKernelSharedMemoryFewBank(unsigned int dimGrid,
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
		unsigned int numberOfCentroids);

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
												unsigned int no_dims,
												unsigned int m,
												unsigned int numberOfCentroids);



void notBoolArray(unsigned int dimBlock,
				  cudaStream_t stream,
				  bool* imputAndOutput,
				  std::size_t lenght);


#endif /* POINTSCONTAINEDDEVICE_H_ */
