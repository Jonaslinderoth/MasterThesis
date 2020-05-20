#include "whatDataInCentroid.h"
#include "../randomCudaScripts/DeleteFromArray.h"
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
		for(; indexDim < (point_dim/8)*8 +((point_dim%8)/4)*4 ; indexDim+=4){
			const size_t indexData_f = indexDataNoDim_f + indexDim;
			{
				floatArray4 tmp = *((floatArray4*)(data+indexData_f));
				pointBuffer.f0 = tmp.f0;
				pointBuffer.f1 = tmp.f1;
				pointBuffer.f2 = tmp.f2;
				pointBuffer.f3 = tmp.f3;
				tmp = *((floatArray4*)(data+centroid_f+indexDim));
				centroidBuffer.f0 = tmp.f0;
				centroidBuffer.f1 = tmp.f1;
				centroidBuffer.f2 = tmp.f2;
				centroidBuffer.f3 = tmp.f3;
			}
			{
				boolArray4 tmp = *((boolArray4*)(dimensions+indexDim));
				dimBuffer.b0 = tmp.b0;
				dimBuffer.b1 = tmp.b1;
				dimBuffer.b2 = tmp.b2;
				dimBuffer.b3 = tmp.b3;
			}
			
			d &= (not (dimBuffer.b0)) || (abs(centroidBuffer.f0 - pointBuffer.f0) < width);
			d &= (not (dimBuffer.b1)) || (abs(centroidBuffer.f1 - pointBuffer.f1) < width);
			d &= (not (dimBuffer.b2)) || (abs(centroidBuffer.f2 - pointBuffer.f2) < width);
			d &= (not (dimBuffer.b3)) || (abs(centroidBuffer.f3 - pointBuffer.f3) < width);
		}
		// process the remaining up to 3 points
		for(; indexDim < (point_dim/8)*8 +((point_dim%8)/4)*4+(point_dim%8)%4 ; indexDim++){
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


bool whatDataIsInCentroid(size_t dimGrid,
						  size_t dimBlock,
						  cudaStream_t stream,
						  bool* output,
						  float* data,
						  unsigned int* centroids,
						  bool* dimensions,
						  const float width,
						  const unsigned int point_dim,
						  const unsigned int no_data_p){



	whatDataIsInCentroidKernel<<<dimGrid,dimBlock,dimBlock*sizeof(unsigned int),stream>>>(output,
																					data,
																					dimensions,
																					centroids,
																					no_data_p,
																					point_dim,
																					width);
	return true;
}


bool whatDataIsInCentroidChunks(size_t dimGrid,
								size_t dimBlock,
								cudaStream_t stream,
								bool* output,
								float* data,
								unsigned int* centroids,
								bool* dimensions,
								const float width,
								const unsigned int point_dim,
								const unsigned int no_data_p){



	whatDataIsInCentroidChunks<<<dimGrid,dimBlock,dimBlock*sizeof(unsigned int),stream>>>(output,
																					data,
																					dimensions,
																					centroids,
																					no_data_p,
																					point_dim,
																					width);
	return true;
}


bool whatDataIsInCentroidFewPoints(size_t dimGrid,
								   size_t dimBlock,
								   cudaStream_t stream,
						  	  	   bool* output,
						  	  	   float* data,
						  	  	   bool* dimensions,
						  	  	   unsigned int* centroids,
						  	  	   const float width,
						  	  	   const unsigned int point_dim,
						  	  	   const unsigned int no_data_p){



	whatDataIsInCentroidKernelFewPoints<<<dimGrid,dimBlock,dimBlock*sizeof(unsigned int),stream>>>(output,
																					 	 	 data,
																					 	 	 centroids,
																					 	 	 no_data_p,
																					 	 	 point_dim,
																					 	 	 width);
	return true;
}



__global__ void gpuWhereThingsGo(unsigned int* d_outData,
								const unsigned int* d_data,
								const unsigned int size){
	const size_t idx = blockIdx.x*blockDim.x+threadIdx.x;

	if(idx <= size){
		d_outData[idx] = size+2;
		const unsigned int offset = d_data[idx];
		unsigned int nextOffset = offset+1;
		if(idx != size){
			nextOffset = d_data[idx+1];
		}
		if(offset == nextOffset){
			d_outData[idx] = idx-offset;
		}
	}
}





__global__ void gpuDimensionChanger(float* d_outData,
								    const unsigned int* d_wereThingsGoArray,
								    const float* d_data,
								    const unsigned int numElements,
								    const unsigned int dimensions,
								    const unsigned int dimensionRemaning){
	const size_t idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx < numElements*dimensions){
		const size_t pointIdex = idx/dimensions;
		const size_t dimIndex = idx%dimensions;
		const size_t newPointIdex = pointIdex*dimensionRemaning;
		const size_t go = d_wereThingsGoArray[dimIndex];
		// printf("try writing to %u\n",go);
		if(go<dimensions){
			const size_t newIndex = newPointIdex+go;
			const float theData = d_data[idx];
			d_outData[newIndex] = theData;
		}
	}
}

	
void whatDataIsInCentroidKernelFewPointsKernel(unsigned int dimGrid,
											   unsigned int dimBlock,
											   cudaStream_t stream,
											   bool* output,
											   float* data,
											   unsigned int* centroids,
											   bool* dims,
											   const float width,
											   const unsigned int point_dim,
											   const unsigned int no_data){

	//i want a prefix sum of what dimensions are used.
	int size_of_out_blelloch = sizeof(unsigned int)*(point_dim+1);
	unsigned int* d_out_blelloch;
	checkCudaErrors(cudaMalloc(&d_out_blelloch, size_of_out_blelloch));
	sum_scan_blelloch(stream, d_out_blelloch,dims,point_dim+1, true);
	unsigned int* h_out_blelloch;
	cudaMallocHost(&h_out_blelloch,4*sizeof(unsigned int));
	cudaMemcpyAsync(h_out_blelloch, d_out_blelloch+point_dim, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);

	cudaStreamSynchronize(stream);
	const unsigned int dimensionsLeft = point_dim-(h_out_blelloch[0]);
   
	unsigned int* d_out_whereThingsGo;
	checkCudaErrors(cudaMalloc(&d_out_whereThingsGo, size_of_out_blelloch));

	const unsigned int dimBlockWhereThingsGo = dimBlock;
	unsigned int dimGridWhereThingsGo = point_dim/dimBlock;
	if(point_dim%dimBlock != 0){
		dimGridWhereThingsGo++;
	}

	gpuWhereThingsGo<<<dimGridWhereThingsGo,dimBlockWhereThingsGo,0,stream>>>(d_out_whereThingsGo,d_out_blelloch,point_dim);


	unsigned int size_of_reducedData = sizeof(float)*dimensionsLeft*no_data;

	float* d_reducedData;
	checkCudaErrors(cudaMalloc(&d_reducedData, size_of_reducedData));

	const unsigned int dimBlockgpuDimensionChanger = dimBlock;
	unsigned int dimGridgpuDimensionChanger = (no_data*point_dim)/dimBlock;
	if((no_data*point_dim)%dimBlock != 0){
		dimGridgpuDimensionChanger++;
	}
	gpuDimensionChanger<<<dimGridgpuDimensionChanger,dimBlockgpuDimensionChanger,0,stream>>>(d_reducedData,d_out_whereThingsGo,data,no_data,point_dim,dimensionsLeft);


	whatDataIsInCentroidFewPoints(dimGrid,
								  dimBlock,
								  stream,
								  output,
								  d_reducedData,
								  dims,
								  centroids,
								  width,
								  dimensionsLeft,
								  no_data);


};



__global__ void whatDataIsInCentroidLessReading(bool* output,
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
			const bool dim = dimensions[indexDim];
			if(dim){
				const float dat = data[indexData_f];
				const float cen = data[centroid_f+indexDim];
				//printf("%u : c: %f p: %f \n", indexData_p, cen, dat);
				d &= (abs(cen - dat) < width);
			}
		}
		output[indexData_p] = d;
	}
}



__global__ void whatDataIsInCentroidLessReadingAndBreaking(bool* output,
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
			const bool dim = dimensions[indexDim];
			if(dim){
				const float dat = data[indexData_f];
				const float cen = data[centroid_f+indexDim];
				//printf("%u : c: %f p: %f \n", indexData_p, cen, dat);
				d &= (abs(cen - dat) < width);
				if(!d){
					break;
				}
			}
		}
		output[indexData_p] = d;
	}
}

bool whatDataIsInCentroidLessReadingWrapper(size_t dimGrid,
						  size_t dimBlock,
						  cudaStream_t stream,
						  bool* output,
						  float* data,
						  unsigned int* centroids,
						  bool* dimensions,
						  const float width,
						  const unsigned int point_dim,
						  const unsigned int no_data_p){



	whatDataIsInCentroidLessReading<<<dimGrid,dimBlock,dimBlock*sizeof(unsigned int),stream>>>(output,
																					data,
																					dimensions,
																					centroids,
																					no_data_p,
																					point_dim,
																					width);
	return true;
}

bool whatDataIsInCentroidLessReadingAndBreakingWrapper(size_t dimGrid,
						  size_t dimBlock,
						  cudaStream_t stream,
						  bool* output,
						  float* data,
						  unsigned int* centroids,
						  bool* dimensions,
						  const float width,
						  const unsigned int point_dim,
						  const unsigned int no_data_p){



	whatDataIsInCentroidLessReadingAndBreaking<<<dimGrid,dimBlock,dimBlock*sizeof(unsigned int),stream>>>(output,
																					data,
																					dimensions,
																					centroids,
																					no_data_p,
																					point_dim,
																					width);
	return true;
}


 

std::vector<bool>* whatDataIsInCentroidTester(std::vector<bool>* dims,
											  std::vector<std::vector<float>*>* data,
											  unsigned int centroid,
											  float width,
											  containedType type){

	// Calculaating sizes
	unsigned int point_dim = dims->size();
	unsigned int no_of_points = data->size();
	unsigned int no_of_centroids = 1;

	unsigned int floats_in_data = point_dim * no_of_points;
	unsigned int bools_in_dims = point_dim;
	unsigned int bools_in_output = no_of_points;

	unsigned int size_of_data = floats_in_data*sizeof(float);
	unsigned int size_of_dims = (bools_in_dims+1)*sizeof(bool);
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
	if(type == NaiveContained){
		whatDataIsInCentroidKernel<<<ceilf((float)no_of_points/1024), 1024>>>(output_d, data_d, dims_d, centroids_d,
																			  no_of_points, point_dim, width);
		
	}else if(type == ChunksContained){
		whatDataIsInCentroidChunks<<<ceilf((float)no_of_points/1024), 1024>>>(output_d, data_d, dims_d, centroids_d,
																			  no_of_points, point_dim, width);
	}else if(type == FewDimsContained){
		cudaStream_t stream;
		cudaStreamCreate ( &stream);
		whatDataIsInCentroidKernelFewPointsKernel(ceilf((float)no_of_points/1024),
												  1024,
												  stream,
												  output_d,
												  data_d,
												  centroids_d,
												  dims_d,
												  width,
												  point_dim,
												  no_of_points
												  );
	cudaStreamSynchronize(stream);
		cudaStreamDestroy(stream);
	}else if(type == LessReadingContained){
		whatDataIsInCentroidLessReading<<<ceilf((float)no_of_points/1024), 1024>>>(output_d, data_d, dims_d, centroids_d,
																			  no_of_points, point_dim, width);
	}else if(type == LessReadingBreakContained){
		whatDataIsInCentroidLessReadingAndBreaking<<<ceilf((float)no_of_points/1024), 1024>>>(output_d, data_d, dims_d, centroids_d,
																			  no_of_points, point_dim, width);		
	}


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
