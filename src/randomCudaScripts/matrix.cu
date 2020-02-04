#include <iostream>
#define TILE_WIDTH 16

__global__ void matMulSimpleDevice(float* d_M, float* d_N, float* d_p, int width){
	int row = blockIdx.y*blockDim.y+threadIdx.y;
	int col = blockIdx.x*blockDim.x+threadIdx.x;

	if ((row < width) && (col < width)){
		float pvalue = 0;
		for(int k = 0; k < width; ++k){
			pvalue += d_M[row*width+k]*d_N[k*width+col];
		}
		d_p[row*width+col] = pvalue;
	}
}

__global__ void matMulAdvancedDevice(float* d_M, float* d_N, float* d_p, int width){
	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];


	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;

	int row = by * TILE_WIDTH + ty;
	int col = bx * TILE_WIDTH + tx;

	float pvalue = 0;
	int a = width/TILE_WIDTH;
	if (width%TILE_WIDTH) a++;

	for (int m = 0; m < a; m++){
		if((row < width) && (m * TILE_WIDTH+tx < width)){
			Mds[ty][tx] = d_M[row*width + m * TILE_WIDTH+tx];
		}else{
			Mds[ty][tx] = 0.0;
		}
		if((col < width) && (m * TILE_WIDTH+ty < width)){
			Nds[ty][tx] = d_N[(m*TILE_WIDTH+ty)*width + col];
		}else{
			Nds[ty][tx] = 0.0;
		}
		__syncthreads();

		for (int k = 0; k < TILE_WIDTH; k++){
			//if((m * TILE_WIDTH + tx < width) && (m * TILE_WIDTH + ty < width)){
				pvalue += Mds[ty][k] * Nds[k][tx];
			//}

		}
		__syncthreads();
	}

	if ((row < width) && (col < width)){
		d_p[row * width + col] = pvalue;
	}

}


float* matMulSimple(float* a, float* b, int width){
	// host arrays
	float* h_out;
	int matSize = sizeof(float)*width*width;
	h_out = (float*)malloc(matSize); // matrix of width time width

	// device arrays
	float* d_M;
	float* d_N;
	float* d_p;


	cudaMalloc((void **) &d_M, matSize);
	cudaMalloc((void **) &d_N, matSize);
	cudaMalloc((void **) &d_p, matSize);

	// copy to device
	cudaMemcpy( d_M, a, matSize, cudaMemcpyHostToDevice);
    cudaMemcpy( d_N, b, matSize, cudaMemcpyHostToDevice);


    int blockWidth = 16;
    int numBlocks = width/blockWidth;
    if (width%blockWidth) numBlocks++;
    dim3 dimGrid(numBlocks, numBlocks);
    dim3 dimBlock(blockWidth, blockWidth);


    matMulSimpleDevice<<<dimGrid, dimBlock>>>(d_M, d_N, d_p, width);
    cudaMemcpy(h_out, d_p, matSize, cudaMemcpyDeviceToHost);

    // Release device memory
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_p);
    return h_out;
}


float* matMulAdvanced(float* a, float* b, int width){
	// host arrays
	float* h_out;
	int matSize = sizeof(float)*width*width;
	h_out = (float*)malloc(matSize); // matrix of width time width

	// device arrays
	float* d_M;
	float* d_N;
	float* d_p;


	cudaMalloc((void **) &d_M, matSize);
	cudaMalloc((void **) &d_N, matSize);
	cudaMalloc((void **) &d_p, matSize);

	// copy to device
	cudaMemcpy( d_M, a, matSize, cudaMemcpyHostToDevice);
    cudaMemcpy( d_N, b, matSize, cudaMemcpyHostToDevice);


    int numBlocks = width/TILE_WIDTH;
    if (width%TILE_WIDTH) numBlocks++;
    dim3 dimGrid(numBlocks, numBlocks);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);


    matMulAdvancedDevice<<<dimGrid, dimBlock>>>(d_M, d_N, d_p, width);
    cudaMemcpy(h_out, d_p, matSize, cudaMemcpyDeviceToHost);

    // Release device memory
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_p);
    return h_out;
}

float matSum(float* v, int width){
	double res = 0;
	int i;
	int j;
	for (i = 0; i < width; i++){
		for (j = 0; j < width; j++){
		res += v[i*width+j];
		}
	}
	return res;
}

bool matDiagEQ(float* v, float* w, int width){
	bool res = true;
	for (int i = 0; i < width; i++){
		res &= (v[i*width+i] == w[i*width+i]);
	}
	return res;
}


void printMat(float* v, int width){
	for(int i = 0; i< width; i++){
		for(int j = 0; j < width; j ++){
			std::cout << v[i*width+j] << ", ";
		}
		std::cout << std::endl;
	}
}
