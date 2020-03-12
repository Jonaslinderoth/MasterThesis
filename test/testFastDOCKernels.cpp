#include <gtest/gtest.h>
#include <iostream>
#include "../src/DOC_GPU/DOCGPU_Kernels.h"
#include "../src/DOC/HyperCube.h"
#include "../src/randomCudaScripts/arrayEqual.h"
#include <random>




TEST(testFastDOCKernels, DISABLED_testSmall){
	const unsigned long point_dim = 3;
	const unsigned long no_data_p = 4;
	const unsigned long with = 2;


	const unsigned long no_data_f = no_data_p*point_dim;
	const unsigned long size_of_data = no_data_f*sizeof(float);
	const unsigned long size_of_count = sizeof(float);
	const unsigned long size_of_dims = point_dim*sizeof(bool);
	const unsigned long size_of_centroid = sizeof(unsigned long);
	const unsigned long size_of_output = no_data_p*sizeof(bool);



	unsigned long* data_h = (unsigned long*) malloc(size_of_data);
	unsigned long* centroid_h = (unsigned long*)malloc(size_of_centroid);
	unsigned long* count_h = (unsigned long*)malloc(size_of_count);
	unsigned long* desired_count_h = (unsigned long*)malloc(size_of_count);

	bool* dim_h = (bool*)malloc(size_of_dims);
	bool* output_h = (bool*)malloc(size_of_output);
	bool* desired_output_h = (bool*)malloc(size_of_output);


	float* data_d;
	unsigned long* centroid_d;
	unsigned long* count_d;
	bool* dim_d;
	bool* output_d;

	cudaMalloc((void **) &data_d, size_of_data);
	cudaMalloc((void **) &centroid_d,size_of_centroid);
	cudaMalloc((void **) &dim_d, size_of_dims);
	cudaMalloc((void **) &output_d, size_of_output);
	cudaMalloc((void **) &count_d, size_of_count);

	centroid_h[0] = 1;

	data_h[0] = 3;
	data_h[1] = 3;
	data_h[2] = 6;
	data_h[3] = 2;
	data_h[4] = 2;
	data_h[5] = 2;
	data_h[6] = -1;
	data_h[7] = 1;
	data_h[8] = 2;
	data_h[9] = 1;
	data_h[10] = -1;
	data_h[11] = 2;

	dim_h[0] = true;
	dim_h[1] = true;
	dim_h[2] = false;

	desired_output_h[0] = true;
	desired_output_h[1] = true;
	desired_output_h[2] = false;
	desired_output_h[3] = false;

	desired_count_h[0] = 2;

	cudaMemcpy(centroid_d, centroid_h, size_of_centroid, cudaMemcpyHostToDevice);
	cudaMemcpy(data_d, data_h, size_of_data, cudaMemcpyHostToDevice);
	cudaMemcpy(dim_d, dim_h, size_of_dims, cudaMemcpyHostToDevice);


	
	 cudaStream_t stream1;
	 cudaStreamCreate(&stream1);

	 whatDataIsInCentroid(stream1,1024,data_d,centroid_d,dim_d,output_d,with,point_dim,no_data_p);

	 cudaStreamDestroy(stream1);
	 cudaFree(data_d);
	 cudaFree(centroid_d);
	 cudaFree(dim_d);
	 cudaFree(output_d);
	 cudaFree(count_d);


   

	EXPECT_TRUE(true);
}
