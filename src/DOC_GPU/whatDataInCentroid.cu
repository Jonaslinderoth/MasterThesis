#include "whatDataInCentroid.h"
#include <assert.h>

__global__
void whatDataIsInCentroidKernel(bool* output,
								unsigned int* count,
								float* data,
								bool* dimensions,
								const unsigned int* centroid,
								const unsigned int no_data_p,
								const unsigned int point_dim,
								const float width){
	extern __shared__ unsigned int countSM[];

	countSM[threadIdx.x] = 0;

	for(unsigned int indexDataChunk_p = 0 ; indexDataChunk_p < no_data_p ; indexDataChunk_p += blockDim.x){
		const size_t indexData_p = indexDataChunk_p+threadIdx.x;
		if(indexData_p < no_data_p){
			const size_t indexDataNoDim_f = indexData_p*point_dim;
			const size_t centroid_f = centroid[0]*point_dim;
			bool d = true;
			for(unsigned int indexDim = 0 ; indexDim < point_dim ; indexDim++){
				const size_t indexData_f = indexDataNoDim_f + indexDim;
				const float dat = data[indexData_f];
				const float cen = data[centroid_f+indexDim];
				const bool dim = dimensions[indexDim];
				d &= (not (dim)) || (abs(cen - dat) < width);


			}
			output[indexData_p] = d;
			countSM[threadIdx.x] += d;

		}
	}

	__syncthreads();
	for(unsigned int s = blockDim.x/2; s > 0; s/=2) {
		if(threadIdx.x < s){
			assert(threadIdx.x+s < blockDim.x);
			countSM[threadIdx.x] += countSM[threadIdx.x+s];
		}
		__syncthreads();
	}

	if(threadIdx.x == 0){
		count[0] = countSM[threadIdx.x];
	};

}



__global__
void whatDataIsInCentroidKernelFewPoints(bool* output,
										 unsigned int* count,
										 float* data,
										 const unsigned int* centroid,
										 const unsigned int no_data_p,
										 const unsigned int point_dim,
										 const float width){
	extern __shared__ unsigned int countSM[];

	countSM[threadIdx.x] = 0;

	for(unsigned int indexDataChunk_p = 0 ; indexDataChunk_p < no_data_p ; indexDataChunk_p += blockDim.x){
		const size_t indexData_p = indexDataChunk_p+threadIdx.x;
		if(indexData_p < no_data_p){
			const size_t indexDataNoDim_f = indexData_p*point_dim;
			const size_t centroid_f = centroid[0]*point_dim;
			bool d = true;
			for(unsigned int indexDim = 0 ; indexDim < point_dim ; indexDim++){
				const size_t indexData_f = indexDataNoDim_f + indexDim;
				const float dat = data[indexData_f];
				const float cen = data[centroid_f+indexDim];
				d &= abs(cen - dat) < width;


			}
			output[indexData_p] = d;
			countSM[threadIdx.x] += d;

		}
	}

	__syncthreads();
	for(unsigned int s = blockDim.x/2; s > 0; s/=2) {
		if(threadIdx.x < s){
			assert(threadIdx.x+s < blockDim.x);
			countSM[threadIdx.x] += countSM[threadIdx.x+s];
		}
		__syncthreads();
	}

	if(threadIdx.x == 0){
		count[0] = countSM[threadIdx.x];
	};

}


bool whatDataIsInCentroid(cudaStream_t stream,
						  unsigned int dimBlock,
						  bool* output,
						  unsigned int* count,
						  float* data,
						  unsigned int* centroids,
						  bool* dimensions,
						  const float width,
						  const unsigned int point_dim,
						  const unsigned int no_data_p){



	whatDataIsInCentroidKernel<<<1,dimBlock,dimBlock*sizeof(unsigned int),stream>>>(output,
																					 count,
																					 data,
																					 dimensions,
																					 centroids,
																					 no_data_p,
																					 point_dim,
																					 width);
	return true;
}


bool whatDataIsInCentroidFewPoints(cudaStream_t stream,
						  	  	   unsigned int dimBlock,
						  	  	   bool* output,
						  	  	   unsigned int* count,
						  	  	   float* data,
						  	  	   unsigned int* centroids,
						  	  	   const float width,
						  	  	   const unsigned int point_dim,
						  	  	   const unsigned int no_data_p){



	whatDataIsInCentroidKernelFewPoints<<<1,dimBlock,dimBlock*sizeof(unsigned int),stream>>>(output,
																					 	 	 count,
																					 	 	 data,
																					 	 	 centroids,
																					 	 	 no_data_p,
																					 	 	 point_dim,
																					 	 	 width);
	return true;
}


