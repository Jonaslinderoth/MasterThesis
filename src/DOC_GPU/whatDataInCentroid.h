/*
 * whatDataInCentroid.h
 *
 *  Created on: Apr 19, 2020
 *      Author: mikkel
 */

#ifndef WHATDATAINCENTROID_H_
#define WHATDATAINCENTROID_H_
#include <vector>

bool whatDataIsInCentroid(cudaStream_t stream,
						  unsigned int dimBlock,
						  bool* output,
						  float* data,
						  unsigned int* centroids,
						  bool* dimensions,
						  const float width,
						  const unsigned int point_dim,
						  const unsigned int no_data_p);

bool whatDataIsInCentroidFewPoints(cudaStream_t stream,
						  	   	   unsigned int dimBlock,
						  	   	   bool* output,
						  	   	   float* data,
						  	   	   unsigned int* centroids,
						  	   	   const float width,
						  	   	   const unsigned int point_dim,
						  	   	   const unsigned int no_data_p);

bool whatDataIsInCentroidChunks(cudaStream_t stream,
								unsigned int dimBlock,
								bool* output,
								float* data,
								unsigned int* centroids,
								const float width,
								const unsigned int point_dim,
								const unsigned int no_data_p);


std::vector<bool>*
whatDataIsInCentroidTester(std::vector<bool>* dims,
						   std::vector<std::vector<float>*>* data,
						   unsigned int centroid,
						   float width);

#endif /* WHATDATAINCENTROID_H_ */
