/*
 * whatDataInCentroid.h
 *
 *  Created on: Apr 19, 2020
 *      Author: mikkel
 */

#ifndef WHATDATAINCENTROID_H_
#define WHATDATAINCENTROID_H_


bool whatDataIsInCentroid(cudaStream_t stream,
						  unsigned int dimBlock,
						  bool* output,
						  unsigned int* count,
						  float* data,
						  unsigned int* centroids,
						  bool* dimensions,
						  const float width,
						  const unsigned int point_dim,
						  const unsigned int no_data_p);

bool whatDataIsInCentroidFewPoints(cudaStream_t stream,
						  	   	   unsigned int dimBlock,
						  	   	   bool* output,
						  	   	   unsigned int* count,
						  	   	   float* data,
						  	   	   unsigned int* centroids,
						  	   	   const float width,
						  	   	   const unsigned int point_dim,
						  	   	   const unsigned int no_data_p);


#endif /* WHATDATAINCENTROID_H_ */
