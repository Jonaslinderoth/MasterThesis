/*
 * DOCGPU.h
 *
 *  Created on: Feb 14, 2020
 *      Author: mikkel
 */

#ifndef DOCGPU_H_
#define DOCGPU_H_

#include "src/dataReader/DataReader.h"
# include <stdio.h>
# include <stdlib.h>
# include <cuda.h>
# include <curand.h>
#include "../dataReader/DataReader.h"




class DOCGPU {
public:
	DOCGPU();
	bool generateRandomSubSets(DataReader* dataReader);
	unsigned int* cudaRandomNumberArray(const size_t lenght ,const curandGenerator_t* gen, unsigned int* array = nullptr);
	virtual ~DOCGPU();

};

#endif /* DOCGPU_H_ */
