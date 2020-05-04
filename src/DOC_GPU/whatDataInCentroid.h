/*
 * whatDataInCentroid.h
 *
 *  Created on: Apr 19, 2020
 *      Author: mikkel
 */

#ifndef WHATDATAINCENTROID_H_
#define WHATDATAINCENTROID_H_
#include <vector>

enum containedType {NaiveContained, ChunksContained};

bool whatDataIsInCentroid(size_t dimGrid,
						  size_t dimBlock,
						  cudaStream_t stream,
						  bool* output,
						  float* data,
						  unsigned int* centroids,
						  bool* dimensions,
						  const float width,
						  const unsigned int point_dim,
						  const unsigned int no_data_p);

bool whatDataIsInCentroidFewPoints(size_t dimGrid,
								   size_t dimBlock,
								   cudaStream_t stream,
						  	   	   bool* output,
						  	   	   float* data,
								   bool* dimensions,
						  	   	   unsigned int* centroids,
						  	   	   const float width,
						  	   	   const unsigned int point_dim,
						  	   	   const unsigned int no_data_p);

bool whatDataIsInCentroidChunks(size_t dimGrid,
								size_t dimBlock,
								cudaStream_t stream,
								bool* output,
								float* data,
								unsigned int* centroids,
								bool* dimensions,
								const float width,
								const unsigned int point_dim,
								const unsigned int no_data_p);




void whatDataIsInCentroidKernelFewPointsKernel(
											   unsigned int dimGrid,
											   unsigned int dimBlock,
											   cudaStream_t stream,
											   bool* output,
											   float* data,
											   unsigned int* centroids,
											   bool* dims,
											   const float width,
											   const unsigned int point_dim,
											   const unsigned int no_data);

std::vector<bool>*
whatDataIsInCentroidTester(std::vector<bool>* dims,
						   std::vector<std::vector<float>*>* data,
						   unsigned int centroid,
						   float width, containedType type = NaiveContained);

#endif /* WHATDATAINCENTROID_H_ */
