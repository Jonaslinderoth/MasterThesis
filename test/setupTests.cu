#include <gtest/gtest.h>
#include <iostream>
#include "../src/randomCudaScripts/add.h"
#include "../src/randomCudaScripts/matrix.h"

TEST(setupTest, testingWorks){
  EXPECT_EQ(2 + 2, 4);
  EXPECT_EQ(2 + 3, 5);
}

TEST(setupTest, cudaWorksAdd){
	int n = 100000000;
	float* a;
	float* b;
	float* out;

	a = (float*) malloc(sizeof(float)*n);
	b = (float*) malloc(sizeof(float)*n);
	out = (float*) malloc(sizeof(float)*n);

	for(int i = 0; i < n; i++){
		a[i] = i+1;
		b[i] = -i;
	}

	out = add(a, b, n);


  EXPECT_EQ(sum(out,n), n);
}


TEST(setupTest, cudaWorksMatMulSimple){
	int n = 2;
	float* a;
	float* b;
	float* out;

	a = (float*) malloc(sizeof(float)*n*n);
	b = (float*) malloc(sizeof(float)*n*n);
	out = (float*) malloc(sizeof(float)*n*n);

	a[0] = b[0] = 1;
	a[1] = b[1] = 2;
	a[2] = b[2] = 3;
	a[3] = b[3] = 4;

	out = matMulSimple(a, b, n);
	EXPECT_EQ(matSum(out,n), 1*1+2*3+2*1+2*4+1*3+3*4+2*3+4*4);

}



TEST(setupTest, cudaWorksMatMulSimple2){
	int n = 333;
	float* a;
	float* b;
	float* out;

	a = (float*) malloc(sizeof(float)*n*n);
	b = (float*) malloc(sizeof(float)*n*n);
	out = (float*) malloc(sizeof(float)*n*n);

	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			a[i*n+j] = 1;
			b[i*n+j] = 1;
		}

	}

	out = matMulSimple(a, b, n);
	float r = matSum(out,n);


  EXPECT_EQ(r, n*n*n);
}


TEST(setupTest, cudaWorksMatMulAdvanced){
	int n = 2;
	float* a;
	float* b;
	float* out;

	a = (float*) malloc(sizeof(float)*n*n);
	b = (float*) malloc(sizeof(float)*n*n);
	out = (float*) malloc(sizeof(float)*n*n);

	a[0] = b[0] = 1;
	a[1] = b[1] = 2;
	a[2] = b[2] = 3;
	a[3] = b[3] = 4;

	out = matMulAdvanced(a, b, n);
	EXPECT_EQ(matSum(out,n), 1*1+2*3+2*1+2*4+1*3+3*4+2*3+4*4);

}



TEST(setupTest, cudaWorksMatMulAdvanced2){
	int n = 333;
	float* a;
	float* b;
	float* out;

	a = (float*) malloc(sizeof(float)*n*n);
	b = (float*) malloc(sizeof(float)*n*n);
	out = (float*) malloc(sizeof(float)*n*n);

	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			a[i*n+j] = 1;
			b[i*n+j] = 1;
		}
	}

	out = matMulAdvanced(a, b, n);

	float r = matSum(out,n);


  EXPECT_EQ(r, n*n*n);
}



TEST(setupTest, cudaWorksMatCompareRand){
	int n = 1000;
	float* a;
	float* b;
	float* out1;
	float* out2;

	a = (float*) malloc(sizeof(float)*n*n);
	b = (float*) malloc(sizeof(float)*n*n);
	out1 = (float*) malloc(sizeof(float)*n*n);
	out2 = (float*) malloc(sizeof(float)*n*n);

	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			a[i*n+j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
			b[i*n+j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		}
	}


	out1 = matMulSimple(a, b, n);
	out2 = matMulAdvanced(a, b, n);

  EXPECT_EQ(matDiagEQ(out1, out2, n), true);
}
