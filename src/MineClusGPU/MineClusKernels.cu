#include "MineClusKernels.h"
#include <assert.h>
#include <iostream>
#include <tuple>
#include "../randomCudaScripts/Utils.h"


/**
   Creates the initial candidtes given the dimensions. 
*/
__global__ void createInitialCandidates(unsigned int dim, unsigned int* output){
	unsigned int candidate = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int numberOfBlocksPrPoint = ceilf((float)dim/32);
	unsigned int myBlock = candidate/32;
	
	if(candidate < dim){
	
		// make sure all are 0;
		for(int i = 0; i < numberOfBlocksPrPoint; i++){
			assert(candidate+dim * i < dim*(ceilf((float)dim/32)));
			output[candidate+dim * i] = 0;
		}
		// set the correct candidate
		unsigned int output_block = (1 << (candidate%32));
		//printf("candidat %u have value %u, writing output at position %u\n", candidate, output_block, candidate+dim*myBlock);
		output[candidate+dim*myBlock] = output_block;
	}
}


/**
   Thin wrapper for CreateInitialCandidates
*/
void createInitialCandidatesWrapper(unsigned int dimGrid,
									unsigned int dimBlock,
									cudaStream_t stream,
									unsigned int dim,
									unsigned int* output
									){
	createInitialCandidates<<<dimGrid, dimBlock, 0, stream>>>(dim, output);
}

/**
   This function is only for testing
*/
std::vector<unsigned int> createInitialCandidatesTester(unsigned int dim){
	unsigned int dimBlock = 1024;
	unsigned int dimGrid = ceilf((float)dim/dimBlock);

	size_t sizeof_output = dim*ceilf((float)dim/32)*sizeof(unsigned int);

	unsigned int* output_h = (unsigned int*) malloc(sizeof_output);
	unsigned int* output_d;

	cudaMalloc((void**) &output_d, sizeof_output);

	createInitialCandidates<<<dimGrid, dimBlock>>>(dim, output_d);

	cudaMemcpy(output_h, output_d, sizeof_output, cudaMemcpyDeviceToHost);
	
	std::vector<unsigned int> res;
	for(int i = 0; i < ceilf((float)dim/32)*dim;i++){
		res.push_back(output_h[i]);
	}
	return res;
}





__global__ void extractMax(unsigned int* candidates, float* scores, unsigned int centroid, unsigned int numberOfCandidates,
					  unsigned int* bestIndex,
					  unsigned int dim, unsigned int* bestCandidate, float* bestScore, unsigned int* bestCentroid){
	unsigned int block = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int numberOfBlocks = ceilf((float)dim/32);
	if(scores[0] > bestScore[0]){
		if(block < numberOfBlocks){
			bestCandidate[block] = candidates[block*numberOfCandidates+bestIndex[0]];
		}
		if(block == numberOfBlocks){
			bestCentroid[0] = centroid;
		}
		if(block == numberOfBlocks+1){
			bestScore[0] = scores[bestIndex[0]];
		}	
	}
}



void extractMaxWrapper(unsigned int dimGrid, unsigned int dimBlock, cudaStream_t stream,
					   unsigned int* candidates, float* scores, unsigned int centroid, unsigned int numberOfCandidates,
					   unsigned int* bestIndex,
					   unsigned int dim, unsigned int* bestCandidate, float* bestScore, unsigned int* bestCentroid){
	extractMax<<<dimGrid, dimBlock, 0, stream>>>(candidates, scores, centroid, numberOfCandidates,
												 bestIndex,
												 dim, bestCandidate, bestScore, bestCentroid);
}


std::pair<std::vector<unsigned int>, float> extractMaxTester(std::vector<bool> oldCandidate,
															 unsigned int oldScore, unsigned int oldCentroid,
															 std::vector<std::vector<bool>> newCandidates,
															 std::vector<float> newScores, unsigned int newCentroid,
															 unsigned int index){
	unsigned int numberOfBlocks = ceilf((float)oldCandidate.size()/32);
	unsigned int dim = oldCandidate.size();
	unsigned int numberOfCandidates = newCandidates.size();

	unsigned int* oldCandidate_h;
	unsigned int* newCandidates_h;

	unsigned int* oldCentroid_h;
	unsigned int* newCentroid_h;

	float* oldScore_h;
	float* newScores_h;

	cudaMallocHost((void**) &oldCandidate_h, numberOfBlocks*sizeof(unsigned int));
	cudaMallocHost((void**) &newCandidates_h, numberOfCandidates*numberOfBlocks*sizeof(unsigned int));

	cudaMallocHost((void**) &oldCentroid_h, sizeof(unsigned int));
	cudaMallocHost((void**) &newCentroid_h, sizeof(unsigned int));

	cudaMallocHost((void**) &oldScore_h, sizeof(float));
	cudaMallocHost((void**) &newScores_h, numberOfCandidates*sizeof(float));


	unsigned int* oldCandidate_d;
	unsigned int* newCandidates_d;

	unsigned int* oldCentroid_d;
	unsigned int* newCentroid_d;

	float* oldScore_d;
	float* newScores_d;
	
	cudaMalloc((void**) &oldCandidate_d, numberOfBlocks*sizeof(unsigned int));
	cudaMalloc((void**) &newCandidates_d, numberOfCandidates*numberOfBlocks*sizeof(unsigned int));

	cudaMalloc((void**) &oldCentroid_d, sizeof(unsigned int));
	cudaMalloc((void**) &newCentroid_d, sizeof(unsigned int));

	cudaMalloc((void**) &oldScore_d, sizeof(float));
	cudaMalloc((void**) &newScores_d, numberOfCandidates*sizeof(float));

	for(unsigned int i = 0; i < newScores.size(); i++){
		newScores_h[i] = newScores.at(i);
	}
	
	unsigned int value = 0;
	unsigned int blockNr = 0;
	for(unsigned int i = 0; i < oldCandidate.size(); i++){
		if(i %32 == 0 && i !=0){
			oldCandidate_h[blockNr] = value;
			blockNr++;
			value = 0;
		}
		value |= (oldCandidate.at(i) << i);
	}
	oldCandidate_h[blockNr] = value;

	oldScore_h[0] = oldScore;
	oldCentroid_h[0] = oldCentroid;


	for(unsigned int i = 0; i < numberOfCandidates; i++){
		unsigned int block = 0;
		unsigned int blockNr = 0;
		for(int j = 0; j < dim; j++){
			if (j % 32 == 0 && j != 0){
				newCandidates_h[i+blockNr*numberOfCandidates] = block;
				block = 0;
				blockNr++;
			}
			block |= (newCandidates.at(i).at(j) << j);
		}
		newCandidates_h[i+blockNr*numberOfCandidates] = block;
	}
	
	unsigned int* index_h;
	cudaMallocHost((void**) &index_h, sizeof(unsigned int));
	index_h[0] = index;
	
	unsigned int* index_d;
	cudaMalloc((void**) &index_d, sizeof(unsigned int));
	
	cudaMemcpy(index_d, index_h, sizeof(unsigned int), cudaMemcpyDeviceToHost);

	newCentroid_h[0] = newCentroid;

	cudaMemcpy(oldCandidate_d, oldCandidate_h, numberOfBlocks*sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(oldScore_d, oldScore_h, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(oldCentroid_d, oldCentroid_h, sizeof(unsigned int), cudaMemcpyHostToDevice);

	
	cudaMemcpy(newCandidates_d, newCandidates_h, numberOfCandidates*numberOfBlocks*sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(newScores_d, newScores_h, numberOfCandidates*sizeof(float), cudaMemcpyHostToDevice);
	//cudaMemcpy(newCentroid_d, newCentroid_h, sizeof(unsigned int), cudaMemcpyHostToDevice);

	extractMax<<<ceilf((float)(dim+2)/1024), 1024>>>(newCandidates_d, newScores_d, newCentroid, newCandidates.size(), index_d, dim,
												 oldCandidate_d, oldScore_d, oldCentroid_d);

	cudaMemcpy(oldCandidate_h, oldCandidate_d, numberOfBlocks*sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(oldScore_h, oldScore_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(oldCentroid_h, oldCentroid_d, sizeof(float), cudaMemcpyDeviceToHost);

	std::vector<unsigned int> bestCandidate;
	for(int i = 0; i < numberOfBlocks; i++){
		bestCandidate.push_back(oldCandidate_h[i]);
	}

	bestCandidate.push_back(oldCentroid_h[0]);

	std::pair<std::vector<unsigned int>, float> result;
	result = make_pair(bestCandidate, oldScore_h[0]);
	return result;

	
	}


__global__ void findPointsInCluster(unsigned int* candidate, float* data, float* centroid, unsigned int dim, unsigned int numberOfPoints, float width, bool* pointsContained){
	unsigned int point = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int numberOfBlocks = ceilf((float)dim/32);
	if(point < numberOfPoints){
		bool isContained = true;
		for(unsigned int i = 0; i < numberOfBlocks; i++){
			unsigned int block = candidate[i];
			for(unsigned int j = 0; j < 32; j++){
				if(i*32+j < dim){
					bool isDimChosen = (block >> j) & 1;
					float cent = centroid[i*32+j];
					float poin = data[point*dim+i*32+j];
					bool r = (not(isDimChosen)) || ((abs(cent - poin)) < width);
					isContained &= r;
				}else{
					break;
				}
			}
		}
		pointsContained[point] = isContained;
	}
}


void findPointInClusterWrapper(unsigned int dimGrid, unsigned int dimBlock, cudaStream_t stream,
							   unsigned int* candidate, float* data, float* centroid, unsigned int dim,
							   unsigned int numberOfPoints, float width, bool* pointsContained){
	findPointsInCluster<<<dimGrid, dimBlock, 0, stream>>>(candidate, data, centroid, dim, numberOfPoints, width, pointsContained);
}



std::vector<bool> findPointsInClusterTester(std::vector<bool> candidate, std::vector<std::vector<float>*>* data, unsigned int centroid, float width){
	unsigned int numberOfPoints = data->size();
	unsigned int dim = data->at(0)->size();
	unsigned int numberOfBlocks = ceilf((float)dim/32);
	
	size_t sizeOfCandidate = numberOfBlocks*sizeof(unsigned int);
	size_t sizeOfCentroid = dim*sizeof(float);
	size_t sizeOfData = numberOfPoints * dim * sizeof(float);
	size_t sizeOfPointsContained = numberOfPoints * sizeof(bool);
	
	unsigned int* candidate_h;
	float* centroid_h;
	float* data_h;
	bool* pointsContained_h;

	cudaMallocHost((void**) &candidate_h, sizeOfCandidate);
	cudaMallocHost((void**) &centroid_h, sizeOfCentroid);
	cudaMallocHost((void**) &data_h, sizeOfData);
	cudaMallocHost((void**) &pointsContained_h, sizeOfPointsContained);

	unsigned int* candidate_d;
	float* centroid_d;
	float* data_d;
	bool* pointsContained_d;

	cudaMalloc((void**) &candidate_d, sizeOfCandidate);
	cudaMalloc((void**) &centroid_d, sizeOfCentroid);
	cudaMalloc((void**) &data_d, sizeOfData);	
	cudaMalloc((void**) &pointsContained_d, sizeOfPointsContained);
	
	unsigned int value = 0;
	unsigned int blockNr = 0;
	for(unsigned int i = 0; i < candidate.size(); i++){
		if(i %32 == 0 && i !=0){
			candidate_h[blockNr] = value;
			blockNr++;
			value = 0;
		}
		value |= (candidate.at(i) << i);
	}
	candidate_h[blockNr] = value;

	for(unsigned int i = 0; i < numberOfPoints; i++){
		for(unsigned int j = 0; j < dim; j++){
			data_h[i*dim+j] = data->at(i)->at(j);
		}
	}

	for(int i = 0; i < dim; i++){
		centroid_h[i] = data->at(centroid)->at(i);
	}

	
	
	unsigned int dimBlock = 1024;
	unsigned int dimGrid = ceilf((float)numberOfPoints/dimBlock);

	cudaMemcpy(candidate_d, candidate_h, sizeOfCandidate, cudaMemcpyHostToDevice);
	cudaMemcpy(data_d, data_h, sizeOfData, cudaMemcpyHostToDevice);
	cudaMemcpy(centroid_d, centroid_h, sizeOfCentroid, cudaMemcpyHostToDevice);


	
	findPointsInCluster<<<dimGrid, dimBlock>>>(candidate_d, data_d, centroid_d, dim, numberOfPoints, width, pointsContained_d);

	cudaMemcpy(pointsContained_h, pointsContained_d, sizeOfPointsContained, cudaMemcpyDeviceToHost);
	
	auto result = std::vector<bool>();
	for(unsigned int i = 0; i < numberOfPoints; i++){
		result.push_back(pointsContained_h[i]);
	}
	return result;
}


__global__ void orKernel(unsigned int numberOfElements, bool* a, bool* b){
	unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
	if(i < numberOfElements){
		a[i] |= b[i];	
	}
}

void orKernelWrapper(unsigned int dimGrid, unsigned int dimBlock,  cudaStream_t stream,
					  unsigned int numberOfElements, bool* a, bool* b){
	orKernel<<<dimGrid, dimBlock, 0, stream>>>(numberOfElements, a, b);
}



__global__ void disjointClusters(unsigned int* centroids, float* scores, unsigned int* subspaces, float* data, const unsigned int numberOfClusters, unsigned int dim, float width, unsigned int* output){
	extern __shared__ unsigned int out[];
	unsigned int k = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int numberOfComparisons = (numberOfClusters*(numberOfClusters+1))/2 - numberOfClusters;

	
	// setting the output
	if(k < numberOfClusters){
		output[k] = true;
	}
	
	// setting the shared memory
	for(unsigned int i = 0; i < ceilf((float)numberOfClusters/blockDim.x); i++){
		if(threadIdx.x+blockDim.x*i < numberOfClusters){
			out[threadIdx.x+blockDim.x*i] = 1;	
		}
	}
	
	__syncthreads();
	
	if(k < numberOfComparisons){
		unsigned int i = numberOfClusters - 2- floorf(sqrtf(-8*k + 4*numberOfClusters*(numberOfClusters-1)-7)/ 2.0 - 0.5);
		unsigned int j = k + i + 1 - numberOfClusters*(numberOfClusters-1)/2 + (numberOfClusters-i)*((numberOfClusters-i)-1)/2;
		bool isDisjoint = false;
		unsigned int blockNr = 0;
		unsigned int currentBlock = 0;
		unsigned int centroidI = centroids[i];
		unsigned int centroidJ = centroids[j];
		unsigned int numberOfBlocks = ceilf((float)dim/32);

				// if the score i 0, There can be no points in the cluster, thereby it can be deleted
		if(scores[i] == 0 || scores[j] == 0){
			atomicAnd(&out[i], scores[i] != 0);
			atomicAnd(&out[j], scores[j] != 0);
		}else{
			for(unsigned int a = 0; a < dim; a++){
				if(a%32 == 0){
					blockNr = a/32;
					assert(i*numberOfBlocks+blockNr < numberOfBlocks*numberOfClusters);
					assert(j*numberOfBlocks+blockNr < numberOfBlocks*numberOfClusters);
					currentBlock = subspaces[i*numberOfBlocks+blockNr] & subspaces[j*numberOfBlocks+blockNr];
				}
				float tempI = data[centroidI*dim+a];
				float tempJ = data[centroidJ*dim+a];
				isDisjoint |= ((currentBlock >> a%32) & 1) && ((abs(tempI - tempJ) >= 2*width));	
			}


			if(isDisjoint){
				//printf("k,i,j: %u, %u, %u are disjoint\n",k, i,j);
				atomicAnd(&out[i], 1);
				atomicAnd(&out[j], 1);
			}else if(scores[i] < scores[j]){
				//printf("k,i,j: %u,  %u, %u; score %f < %f, keeping %u, deleting %u\n",k, i,j,scores[i], scores[j], j,i);
				atomicAnd(&out[i], 0);
				atomicAnd(&out[j], 1);
			}else if(scores[i] == scores[j]){
				//printf("k,i,j: %u, %u, %u; score %f == %f, keeping %u, deleting %u\n",k, i,j,scores[i], scores[j], min(i,j),max(i,j));
				atomicAnd(&out[min(i,j)], 1);
				atomicAnd(&out[max(i,j)], 0);
			}else{
				//printf("k,i,j: %u, %u, %u; score %f > %f, keeping %u, deleting %u\n",k, i,j,scores[i], scores[j], i,j);
				atomicAnd(&out[i], 1);				
				atomicAnd(&out[j], 0);
			}
		}
	}

	__syncthreads();	
	for(unsigned int i = 0; i < ceilf((float)numberOfClusters/blockDim.x); i++){
		if(threadIdx.x+blockDim.x*i < numberOfClusters){
			atomicAnd(&output[threadIdx.x+blockDim.x*i], out[threadIdx.x+blockDim.x*i]);	
		}
	}
}

void disjointClustersWrapper(unsigned int dimGrid, unsigned int dimBlock, cudaStream_t stream,
							 unsigned int* centroids, float* scores, unsigned int* subspaces,
							 float* data, unsigned int numberOfClusters, unsigned int dim,
							 float width, unsigned int* output){


	unsigned int smem = numberOfClusters*sizeof(unsigned int);
	disjointClusters<<<dimGrid, dimBlock, smem, stream>>>(centroids, scores, subspaces,
														  data, numberOfClusters, dim,
														  width, output);
}

std::vector<bool> disjointClustersTester(std::vector<std::vector<float>*>* data_v, std::vector<unsigned int> centroids_v, std::vector<unsigned int> subspaces_v, std::vector<float> scores_v){
	unsigned int width = 10;
	unsigned int dim = data_v->at(0)->size();
	unsigned int numberOfPoints = data_v->size();
	unsigned int smem = scores_v.size()*sizeof(unsigned int);
	
	float* data;
	float* scores;
	unsigned int* centroids;
	unsigned int* subspaces;
	unsigned int* output;

	unsigned int numberOfComparisons = (scores_v.size()*(scores_v.size()+1))/2-scores_v.size();

	unsigned int dimBlock = 1024;
	unsigned int dimGrid = ceilf((float)numberOfComparisons/dimBlock);
	assert(dimGrid <= 1);

	size_t sizeofData = numberOfPoints*dim*sizeof(float);
	size_t sizeofScores = scores_v.size()*sizeof(float);
	size_t sizeofCentroids = centroids_v.size()*sizeof(unsigned int);
	size_t sizeofSubspaces = subspaces_v.size()*sizeof(unsigned int);
	size_t sizeofOutput = centroids_v.size()*sizeof(unsigned int);


	cudaMallocManaged((void**) &data, sizeofData);
	cudaMallocManaged((void**) &scores, sizeofScores);
	cudaMallocManaged((void**) &centroids, sizeofCentroids);
	cudaMallocManaged((void**) &subspaces, sizeofSubspaces);
	cudaMallocManaged((void**) &output, sizeofOutput);

	for(int i = 0; i < data_v->size(); i++){
		for(int j = 0; j < dim; j++){
			data[i*dim+j] = data_v->at(i)->at(j);	
		}
	}

	for(int i = 0; i < scores_v.size(); i++){
		scores[i] = scores_v.at(i);
	}

	for(int i = 0; i < subspaces_v.size(); i++){
		subspaces[i] = subspaces_v.at(i);
	}

	for(int i = 0; i < centroids_v.size(); i++){
		centroids[i] = centroids_v.at(i);
	}

	disjointClusters<<<dimGrid, dimBlock, smem>>>(centroids, scores, subspaces,
														  data, scores_v.size(), dim,
														  width, output);

	cudaDeviceSynchronize();
	std::vector<bool> output_v;

	for(int i = 0; i < centroids_v.size(); i++){
		output_v.push_back(output[i]);
	}

	cudaFree(data);
	cudaFree(centroids);
	cudaFree(scores);
	cudaFree(subspaces);
	cudaFree(output);
	return output_v;
	
}


__global__ void unsignedIntToBoolArray(unsigned int* input, unsigned int numberOfElements, bool* output){
	unsigned int k = blockIdx.x*blockDim.x+threadIdx.x;
	if(k < numberOfElements){
		output[k] = input[k];
	}
}


void unsignedIntToBoolArrayWrapper(unsigned int dimGrid, unsigned int dimBlock, cudaStream_t stream,
							unsigned int* input, unsigned int numberOfElements, bool* output){
	unsignedIntToBoolArray<<<dimGrid, dimBlock, 0, stream>>>(input, numberOfElements, output);
}


__global__ void copyCentroid(unsigned int* centroids, float* data, unsigned int dim, unsigned int numberOfCentroids, float* centroidsOut){
	unsigned int k = blockIdx.x*blockDim.x+threadIdx.x;
	if(k < dim*numberOfCentroids){
		unsigned int centroidToUse = k/dim;
		unsigned int dimInCentroid = k%dim;
		unsigned int centroidIndex = centroids[centroidToUse];
		centroidsOut[centroidToUse*dim+dimInCentroid] = data[centroidIndex*dim+dimInCentroid];
	}
}

void copyCentroidWrapper(unsigned int dimGrid, unsigned int dimBlock, cudaStream_t stream,
						 unsigned int* centroids, float* data, unsigned int dim,
						 unsigned int numberOfCentroids, float* centroidsOut){
	copyCentroid<<<dimGrid, dimBlock, 0, stream>>>(centroids, data, dim, numberOfCentroids, centroidsOut);
}


__global__ void indexToBoolVector(unsigned int* index, unsigned int numberOfElements, bool* output){
	unsigned int k = blockIdx.x*blockDim.x+threadIdx.x;
	if(k < numberOfElements){
		output[k] = (index[0] == k);
	}
}

void indexToBoolVectorWrapper(unsigned int dimGrid, unsigned int dimBlock, cudaStream_t stream,
							  unsigned int* index, unsigned int numberOfElements, bool* output
							  ){
	indexToBoolVector<<<dimGrid, dimBlock, 0, stream>>>(index, numberOfElements, output);
	
};