#include <gtest/gtest.h>
#include <iostream>
#include "../src/DOC_GPU/DOCGPU_Kernels.h"
#include "../src/DOC/HyperCube.h"
#include "../src/randomCudaScripts/arrayEqual.h"
#include <random>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}



TEST(testFastDOCKernels, testSmall){
	const unsigned long point_dim = 3;
	const unsigned long no_data_p = 4;
	const unsigned long with = 2;


	const unsigned long no_data_f = no_data_p*point_dim;
	const unsigned long size_of_data = no_data_f*sizeof(float);
	const unsigned long size_of_count = sizeof(unsigned long);
	const unsigned long size_of_dims = point_dim*sizeof(bool);
	const unsigned long size_of_centroid = sizeof(unsigned long);
	const unsigned long size_of_output = no_data_p*sizeof(bool);



	float* data_h = (float*)malloc(size_of_data);
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

	gpuErrchk(cudaMalloc((void **) &data_d, size_of_data));
	gpuErrchk(cudaMalloc((void **) &centroid_d,size_of_centroid));
	gpuErrchk(cudaMalloc((void **) &dim_d, size_of_dims));
	gpuErrchk(cudaMalloc((void **) &output_d, size_of_output));
	gpuErrchk(cudaMalloc((void **) &count_d, size_of_count));

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

	output_h[0] = true;
	output_h[1] = true;
	output_h[2] = true;
	output_h[3] = true;

	gpuErrchk(cudaMemcpy(output_d, output_h, size_of_output, cudaMemcpyHostToDevice));

	gpuErrchk(cudaMemcpy(centroid_d, centroid_h, size_of_centroid, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(data_d, data_h, size_of_data, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dim_d, dim_h, size_of_dims, cudaMemcpyHostToDevice));

	cudaStream_t stream1;
	cudaStreamCreate(&stream1);

	whatDataIsInCentroid(stream1,
						 1024,
						 output_d,
						 count_d,
						 data_d,
						 centroid_d,
						 dim_d,
						 with,
						 point_dim,
						 no_data_p);

	gpuErrchk(cudaMemcpy(output_h, output_d, size_of_output, cudaMemcpyDeviceToHost));

	for(int i = 0  ; i < no_data_p ; i++){

		EXPECT_EQ(output_h[i], desired_output_h[i]) << "at index: " << i;
	}
	//count_h[0] = 0;
	gpuErrchk(cudaMemcpy(count_h, count_d, size_of_count, cudaMemcpyDeviceToHost));

	//if(count_h[0] != desired_count_h[0]){
	//	std::cout << "count wrong is: " << count_h[0] << " shoud be " << desired_count_h[0] << std::endl;
	//}


	EXPECT_EQ(count_h[0] , desired_count_h[0]);

}



TEST(testFastDOCKernels, SUPER_SLOW_testMedium){


	std::mt19937 gen{0};
	gen.seed(1);
	static std::random_device rand;
	std::uniform_int_distribution<int> distSmall(6, 20);
	std::uniform_int_distribution<int> distBig(20, 30);
	const unsigned smallValue = distBig(rand);



	const float with = 10.0;

	for(unsigned long point_dim = 10 ; point_dim < 350-smallValue; point_dim +=smallValue){
		for(unsigned long no_data = 10 ; no_data < 5000-smallValue; no_data +=smallValue){
			unsigned long count = 0;
			std::uniform_real_distribution<float> distPoint(0, 100);

			std::vector<std::vector<float>> data;
			for(unsigned long indexData = 0; indexData < no_data ; indexData++){
				std::vector<float> point;
				for(unsigned long indexDim = 0 ; indexDim < point_dim ; indexDim++){
					point.push_back(distPoint(rand));
				}
				data.push_back(point);
			}

			std::vector<float> centroid;
			std::uniform_int_distribution<unsigned long> distCentroid(0, data.size()-1);
			unsigned long centroidIndex = distCentroid(rand);
			centroid = data.at(centroidIndex);

			std::bernoulli_distribution distBool(0.5);
			std::vector<bool> dimensions;

			for(unsigned long indexDims = 0 ; indexDims < point_dim ; indexDims++){
				dimensions.push_back(distBool(rand));
			}


			HyperCube HyperCube(&centroid,with,&dimensions);

			for(std::vector<std::vector<float>>::iterator iter = data.begin() ; iter != data.end() ; ++iter){
				count += (unsigned long)HyperCube.pointContained(&(*iter));
			}

			const unsigned long no_data_f = no_data*point_dim;
			const unsigned long size_of_data = no_data_f*sizeof(float);
			const unsigned long size_of_count = sizeof(unsigned long);
			const unsigned long size_of_dims = point_dim*sizeof(bool);
			const unsigned long size_of_centroid = sizeof(unsigned long);
			const unsigned long size_of_output = no_data*sizeof(bool);



			float* data_h = (float*)malloc(size_of_data);
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

			gpuErrchk(cudaMalloc((void **) &data_d, size_of_data));
			gpuErrchk(cudaMalloc((void **) &centroid_d,size_of_centroid));
			gpuErrchk(cudaMalloc((void **) &dim_d, size_of_dims));
			gpuErrchk(cudaMalloc((void **) &output_d, size_of_output));
			gpuErrchk(cudaMalloc((void **) &count_d, size_of_count));

			centroid_h[0] = centroidIndex;

			unsigned long indexData = 0;
			for(std::vector<std::vector<float>>::iterator iterData = data.begin() ; iterData != data.end() ; ++iterData){
				for(std::vector<float>::iterator iterPoint = iterData->begin() ; iterPoint != iterData->end() ; ++iterPoint){
					data_h[indexData] = *iterPoint;
					indexData++;
				}
			}

			unsigned long indexDims = 0;
			for(std::vector<bool>::iterator iter = dimensions.begin() ; iter != dimensions.end() ; ++iter){
				dim_h[indexDims] = *iter;
				indexDims++;
			}



			gpuErrchk(cudaMemcpy(output_d, output_h, size_of_output, cudaMemcpyHostToDevice));

			gpuErrchk(cudaMemcpy(centroid_d, centroid_h, size_of_centroid, cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(data_d, data_h, size_of_data, cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(dim_d, dim_h, size_of_dims, cudaMemcpyHostToDevice));

			cudaStream_t stream1;
			cudaStreamCreate(&stream1);

			whatDataIsInCentroid(stream1,
								 1024,
								 output_d,
								 count_d,
								 data_d,
								 centroid_d,
								 dim_d,
								 with,
								 point_dim,
								 no_data);

			gpuErrchk((cudaMemcpy(output_h, output_d, size_of_output, cudaMemcpyDeviceToHost)));

			indexData = 0;
			for(std::vector<std::vector<float>>::iterator iter = data.begin() ; iter != data.end() ; ++iter){

				//if(output_h[indexData] != HyperCube.pointContained(&(*iter)) ){
				//	std::cout << "fail " << output_h[indexData] << " != " << HyperCube.pointContained(&(*iter)) << " at: " << indexData << std::endl;
				//}

				EXPECT_EQ(output_h[indexData] , HyperCube.pointContained(&(*iter)));
				indexData++;
			}

			gpuErrchk((cudaMemcpy(count_h, count_d, size_of_count, cudaMemcpyDeviceToHost)));



			EXPECT_EQ(count_h[0] , count);

			//if(count_h[0] != count){

				//std::cout << "yee ha" << std::endl;
			//}

			delete data_h;
			delete centroid_h;
			delete count_h;
			delete desired_count_h;

			cudaFree(data_d);
			cudaFree(centroid_d);
			cudaFree(count_d);
			cudaFree(dim_d);
			cudaFree(output_d);
			//std::cout << "foo: " << point_dim << " bar: " << no_data << std::endl;
		}
	}
}

