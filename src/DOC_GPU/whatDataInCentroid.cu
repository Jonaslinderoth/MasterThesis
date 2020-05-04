#include "whatDataInCentroid.h"
#include "../randomCudaScripts/Utils.h"

#include <assert.h>

__global__ void whatDataIsInCentroidKernel(bool* output,
								float* data,
								bool* dimensions,
								const unsigned int* centroid,
								const unsigned int no_data_p,
								const unsigned int point_dim,
								const float width){
	unsigned int indexData_p = threadIdx.x + blockIdx.x*blockDim.x;
	if(indexData_p < no_data_p){
		const size_t indexDataNoDim_f = indexData_p*point_dim;
		const size_t centroid_f = centroid[0]*point_dim;
		bool d = true;
		for(unsigned int indexDim = 0 ; indexDim < point_dim ; indexDim++){
			const size_t indexData_f = indexDataNoDim_f + indexDim;
			const float dat = data[indexData_f];
			const float cen = data[centroid_f+indexDim];
			//printf("%u : c: %f p: %f \n", indexData_p, cen, dat);
			const bool dim = dimensions[indexDim];
			d &= (not (dim)) || (abs(cen - dat) < width);
		}
		output[indexData_p] = d;
	}
}

// Hack
struct floatArray4{
	float f0;
	float f1;
	float f2;
	float f3;
};
struct floatArray8{
	float f0;
	float f1;
	float f2;
	float f3;
	float f4;
	float f5;
	float f6;
	float f7;
};

// Hack
struct boolArray8{
	bool b0;
	bool b1;
	bool b2;
	bool b3;
	bool b4;
	bool b5;
	bool b6;
	bool b7;
};

// Hack
struct boolArray4{
	bool b0;
	bool b1;
	bool b2;
	bool b3;
};

__global__ void whatDataIsInCentroidChunks(bool* output,
								float* data,
								bool* dimensions,
								const unsigned int* centroid,
								const unsigned int no_data_p,
								const unsigned int point_dim,
								const float width){
	floatArray8 pointBuffer;
	floatArray8 centroidBuffer;
	boolArray8 dimBuffer;
	unsigned int indexData_p = threadIdx.x + blockIdx.x*blockDim.x;
	if(indexData_p < no_data_p){
		const size_t indexDataNoDim_f = indexData_p*point_dim;
		const size_t centroid_f = centroid[0]*point_dim;
		bool d = true;
		unsigned int indexDim = 0;
		// Process the data in chunks of 8
		for( ; indexDim < (point_dim/8)*8 ; indexDim+=8){
			const size_t indexData_f = indexDataNoDim_f + indexDim;

			pointBuffer = *((floatArray8*)(data+indexData_f));
			centroidBuffer = *((floatArray8*)(data+centroid_f+indexDim));
			dimBuffer = *((boolArray8*)(dimensions+indexDim));
			
			d &= (not (dimBuffer.b0)) || (abs(centroidBuffer.f0 - pointBuffer.f0) < width);
			d &= (not (dimBuffer.b1)) || (abs(centroidBuffer.f1 - pointBuffer.f1) < width);
			d &= (not (dimBuffer.b2)) || (abs(centroidBuffer.f2 - pointBuffer.f2) < width);
			d &= (not (dimBuffer.b3)) || (abs(centroidBuffer.f3 - pointBuffer.f3) < width);
			d &= (not (dimBuffer.b4)) || (abs(centroidBuffer.f4 - pointBuffer.f4) < width);
			d &= (not (dimBuffer.b5)) || (abs(centroidBuffer.f5 - pointBuffer.f5) < width);
			d &= (not (dimBuffer.b6)) || (abs(centroidBuffer.f6 - pointBuffer.f6) < width);
			d &= (not (dimBuffer.b7)) || (abs(centroidBuffer.f7 - pointBuffer.f7) < width);
		}
		// process remaining in chunks of 4
		for(; indexDim < ((point_dim%8)/4)*4 ; indexDim+=4){
			const size_t indexData_f = indexDataNoDim_f + indexDim;
			pointBuffer = *(floatArray8*)((floatArray4*)(data+indexData_f));
			centroidBuffer = *(floatArray8*)((floatArray4*)(data+centroid_f+indexDim));
			dimBuffer = *(boolArray8*)((boolArray4*)(dimensions+indexDim));
			
			d &= (not (dimBuffer.b0)) || (abs(centroidBuffer.f0 - pointBuffer.f0) < width);
			d &= (not (dimBuffer.b1)) || (abs(centroidBuffer.f1 - pointBuffer.f1) < width);
			d &= (not (dimBuffer.b2)) || (abs(centroidBuffer.f2 - pointBuffer.f2) < width);
			d &= (not (dimBuffer.b3)) || (abs(centroidBuffer.f3 - pointBuffer.f3) < width);
		}
		// process the remaining up to 3 points
		for(; indexDim < (point_dim%8)%4 ; indexDim++){
			const size_t indexData_f = indexDataNoDim_f + indexDim;
			pointBuffer.f0 = data[indexData_f];
			centroidBuffer.f0 = data[centroid_f+indexDim];
			dimBuffer.b0 = dimensions[indexDim];
			
			d &= (not (dimBuffer.b0)) || (abs(centroidBuffer.f0 - pointBuffer.f0) < width);
		}

		
		output[indexData_p] = d;
	}
}

__global__ void whatDataIsInCentroidKernelFewPoints(bool* output,
										 float* data,
										 const unsigned int* centroid,
										 const unsigned int no_data_p,
										 const unsigned int point_dim,
										 const float width){
	unsigned int indexData_p = threadIdx.x + blockIdx.x*blockDim.x;
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
	}
}


bool whatDataIsInCentroid(cudaStream_t stream,
						  unsigned int dimBlock,
						  bool* output,
						  float* data,
						  unsigned int* centroids,
						  bool* dimensions,
						  const float width,
						  const unsigned int point_dim,
						  const unsigned int no_data_p){



	whatDataIsInCentroidKernel<<<1,dimBlock,dimBlock*sizeof(unsigned int),stream>>>(output,
																					 data,
																					 dimensions,
																					 centroids,
																					 no_data_p,
																					 point_dim,
																					 width);
	return true;
}


bool whatDataIsInCentroidChunks(cudaStream_t stream,
						  unsigned int dimBlock,
						  bool* output,
						  float* data,
						  unsigned int* centroids,
						  bool* dimensions,
						  const float width,
						  const unsigned int point_dim,
						  const unsigned int no_data_p){



	whatDataIsInCentroidChunks<<<1,dimBlock,dimBlock*sizeof(unsigned int),stream>>>(output,
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
						  	  	   float* data,
						  	  	   unsigned int* centroids,
						  	  	   const float width,
						  	  	   const unsigned int point_dim,
						  	  	   const unsigned int no_data_p){



	whatDataIsInCentroidKernelFewPoints<<<1,dimBlock,dimBlock*sizeof(unsigned int),stream>>>(output,
																					 	 	 data,
																					 	 	 centroids,
																					 	 	 no_data_p,
																					 	 	 point_dim,
																					 	 	 width);
	return true;
}





std::vector<bool>* whatDataIsInCentroidTester(std::vector<bool>* dims,
															std::vector<std::vector<float>*>* data,
															unsigned int centroid,
															float width){

	// Calculaating sizes
	unsigned int point_dim = dims->size();
	unsigned int no_of_points = data->size();
	unsigned int no_of_centroids = 1;

	unsigned int floats_in_data = point_dim * no_of_points;
	unsigned int bools_in_dims = point_dim;
	unsigned int bools_in_output = no_of_points;

	unsigned int size_of_data = floats_in_data*sizeof(float);
	unsigned int size_of_dims = bools_in_dims*sizeof(bool);
	unsigned int size_of_centroids = no_of_centroids*sizeof(unsigned int);
	unsigned int size_of_output = bools_in_output*sizeof(bool);


	// allocating on the host
	float* data_h = (float*) malloc(size_of_data);
	bool* dims_h = (bool*) malloc(size_of_dims);
	unsigned int* centroids_h = (unsigned int*) malloc(size_of_centroids);
	bool* output_h = (bool*) malloc(size_of_output);

	// filling data array
	for(int i= 0; i < no_of_points; i++){
		for(int j = 0; j < point_dim; j++){
			data_h[i*point_dim+j] = data->at(i)->at(j);
		}
	}

	// filling dims array
	for(int j = 0; j < point_dim; j++){
		dims_h[j] = dims->at(j);
	}


	// filling centroid array
	centroids_h[0] = centroid;


	// allocating on device
	float* data_d;
	bool* dims_d;
	unsigned int* centroids_d;
	bool* output_d;

	cudaMalloc((void **) &data_d, size_of_data);
	cudaMalloc((void **) &dims_d, size_of_dims);
	cudaMalloc((void **) &centroids_d, size_of_centroids);
	cudaMalloc((void **) &output_d, size_of_output);

	//Copy from host to device

	cudaMemcpy(data_d, data_h, size_of_data, cudaMemcpyHostToDevice);
	cudaMemcpy(dims_d, dims_h, size_of_dims, cudaMemcpyHostToDevice);
	cudaMemcpy(centroids_d, centroids_h, size_of_centroids, cudaMemcpyHostToDevice);


	// Call kernel
	whatDataIsInCentroidKernel<<<ceilf((float)no_of_points/1024), 1024>>>(output_d, data_d, dims_d, centroids_d,
																  no_of_points, point_dim, width);


	// copy from device
	cudaMemcpy(output_h, output_d, size_of_output, cudaMemcpyDeviceToHost);

	// construnct output
	auto output =  new std::vector<bool>;



	for(int j = 0; j < no_of_points; j++){
		output->push_back(output_h[j]);
	}



	cudaFree(data_d);
	cudaFree(dims_d);
	cudaFree(centroids_d);
	cudaFree(output_d);
	free(data_h);
	free(dims_h);
	free(centroids_h);
	free(output_h);

	
	return output;
};