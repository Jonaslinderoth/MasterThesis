#ifndef DOCGPU_KERNELS_H
#define DOCGPU_KERNELS_H
#include <vector>
#include <curand.h>
#include <curand_kernel.h>


std::pair<std::vector<std::vector<bool>*>*,std::vector<unsigned int>*>
	findDimmensions(std::vector<std::vector<float>*>* data,
					std::vector<unsigned int>* centroids,
					std::vector<std::vector<unsigned int>*>* samples,
					int m, float width = 10.0);



std::pair<std::vector<std::vector<bool>*>*,std::vector<unsigned int>*>
	findDimmensionsChunk(std::vector<std::vector<float>*>* data,
					std::vector<unsigned int>* centroids,
					std::vector<std::vector<unsigned int>*>* samples,
					int m, float width = 10.0);
/*
std::pair<std::vector<std::vector<bool>*>*,std::vector<unsigned int>*>
	pointsContainedSharedMemoryFewBank(std::vector<std::vector<bool>*>* dims,
					    std::vector<std::vector<float>*>* data,
					    std::vector<unsigned int>* centroids,
					    int m,  float width = 10.0);
*/

int argMax(std::vector<float>* scores);


//std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*>
//	findClusterUsingGPU(std::vector<std::vector<float>*>* data,
//				float alpha = 0.1, float beta = 0.25, float width = 15);


float* scoreHost(unsigned int* Cluster_size,
				 unsigned int* Dim_count,
				 float* score_output,
				 int len,
				 float alpha,
				 float beta,
				 unsigned int number_of_points);


void findDimmensionsKernel(unsigned int dimGrid, unsigned int dimBlock,
                           cudaStream_t stream,
        unsigned int* Xs_d, unsigned int* ps_d, float* data_d, bool* res_d,
						   unsigned int* Dsum_out, unsigned int point_dim, unsigned int no_of_samples, unsigned int no_in_sample, unsigned int no_of_ps,
						   unsigned int m, float width, unsigned int number_of_points);

void scoreKernel(unsigned int dimGrid,
				 unsigned int dimBlock,
				 cudaStream_t stream,
				 unsigned int* Cluster_size,
				 unsigned int* Dim_count,
				 float* score_output,
				 unsigned int len,
				 float alpha,
				 float beta,
				 unsigned int num_points);



void notDevice(unsigned int dimGrid,
			   unsigned int dimBlock,
			   cudaStream_t stream,
			   bool* array,
			   unsigned int length);
	
/*
 * generates a random unsigned integer array on the gpu.
 * if the randomSeed seen is true it takes a random seed from the device. else it uses seed.
 *
 */
bool generateRandomStatesArray(cudaStream_t stream,
							   curandState* d_randomStates,
							   const size_t size,
							   const bool randomSeed = true,
							   unsigned int seed = 0,
							   unsigned int dimBlock = 1024);

/* This fuction makes an array of random numbers between min and max in the gpu , given the allocation
 * and the states.
 * to get the states generate random states array call generateRandomStatesArray.
 */
bool generateRandomIntArrayDevice(cudaStream_t stream,
								  unsigned int* d_randomIndexes,
								  curandState* d_randomStates,
								  const size_t size_of_randomStates,
								  const size_t size,
								  const unsigned int max = 100,
								  const unsigned int min = 0,
								  unsigned int dimBlock = 1024);







#endif
