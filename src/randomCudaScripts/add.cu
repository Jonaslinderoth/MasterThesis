#include <stdio.h>
#include <iostream>


__global__ void add_gpu(float *a, float *b, float *out, int n){
	int id = (blockIdx.x*blockDim.x)+threadIdx.x;

	if (id < n){
		out[id] = a[id] + b[id];
	}

}


float* add(float* a, float* b, int n){
	// host arrays
	float* h_out;

	h_out = (float*)malloc(sizeof(float)*n);

	// device arrays
	float* d_a;
	float* d_b;
	float* d_out;


	cudaMalloc((void **) &d_a, sizeof(float)*n);

	cudaMalloc((void **) &d_b, sizeof(float)*n);
	cudaMalloc((void **) &d_out, sizeof(float)*n);

	// copy to device
	cudaMemcpy( d_a, a, sizeof(float)*n, cudaMemcpyHostToDevice);
    cudaMemcpy( d_b, b, sizeof(float)*n, cudaMemcpyHostToDevice);
    int blockSize, gridSize;

    // Number of threads in each thread block
    blockSize = 256;

    // Number of thread blocks in grid
    gridSize = (int)(ceil((float)n/((float)blockSize)));


    add_gpu<<<gridSize, blockSize>>>(d_a, d_b, d_out, n);
    cudaMemcpy(h_out, d_out, sizeof(float)*n, cudaMemcpyDeviceToHost);

    // Release device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    return h_out;

}

float sum(float* v, int n){
	double res = 0;
	int i;
	for (i = 0; i < n; i++){
		res += v[i];
	}
	return res;
}



