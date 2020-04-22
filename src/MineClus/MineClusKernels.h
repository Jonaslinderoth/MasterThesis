#ifndef MINECLUSKERNELS_H
#define MINECLUSKERNELS_H
#include <vector>
#include <tuple>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

enum dublicatesType { Naive, Breaking, MoreBreaking, Hash };
enum transactionsType { Naive_trans, ReducedReads};

void createTransactionsWrapper(unsigned int dimGrid,
							   unsigned int dimBlock,
							   unsigned int smem,
							   cudaStream_t stream,
							   float* data,
							   unsigned int dim,
							   unsigned int numberOfPoints,
							   unsigned int centroidId,
							   float width,
							   unsigned int* output,
							   transactionsType version = transactionsType::Naive_trans);





void createInitialCandidatesWrapper(unsigned int dimGrid,
									unsigned int dimBlock,
									cudaStream_t stream,
									unsigned int dim,
									unsigned int* output
									);


void countSupportWrapper(unsigned int dimGrid,
						 unsigned int dimBlock,
						 cudaStream_t stream,
						 unsigned int* candidates,
						 unsigned int* itemSet,
						 unsigned int dim,
						 unsigned int numberOfItems,
						 unsigned int numberOfCandidates,
						 unsigned int minSupp,
						 float beta,
						 unsigned int* outSupp,
						 float* outScore,
						 bool* outToBeDeleted
						 );



void findDublicatesWrapper(unsigned int dimGrid,
						   unsigned int dimBlock,
						   cudaStream_t stream,
						   unsigned int* candidates,
						   unsigned int numberOfCandidates,
						   unsigned int dim,
						   bool* alreadyDeleted,
						   bool* output,
						   dublicatesType version = dublicatesType::Naive
						   );

void extractMaxWrapper(unsigned int dimGrid, unsigned int dimBlock, cudaStream_t stream,
					   unsigned int* candidates, float* score, unsigned int centroid, unsigned int numberOfCandidates,
					   unsigned int* bestIndex,
					   unsigned int dim, unsigned int* bestCandidate, float* bestScore, unsigned int* bestCentroid);





void findPointInClusterWrapper(unsigned int dimGrid, unsigned int dimBlock, cudaStream_t stream,
							   unsigned int* candidate, float* data, float* centroid, unsigned int dim,
							   unsigned int numberOfPoints, float width, bool* pointsContained);


void orKernelWrapper(unsigned int dimGrid, unsigned int dimBlock,  cudaStream_t stream,
					  unsigned int numberOfElements, bool* a, bool* b);


void disjointClustersWrapper(unsigned int dimGrid, unsigned int dimBlock, cudaStream_t stream,
							 unsigned int* centroids, float* scores, unsigned int* subspaces,
							 float* data, unsigned int numberOfClusters, unsigned int dim,
							 float width, unsigned int* output);

void unsignedIntToBoolArrayWrapper(unsigned int dimGrid, unsigned int dimBlock, cudaStream_t stream,
							unsigned int* input, unsigned int numberOfElements, bool* output);

void copyCentroidWrapper(unsigned int dimGrid, unsigned int dimBlock, cudaStream_t stream,
						 unsigned int* centroids, float* data, unsigned int dim,
						 unsigned int numberOfCentroids, float* centroidsOut);

void indexToBoolVectorWrapper(unsigned int dimGrid, unsigned int dimBlock, cudaStream_t stream,
							  unsigned int* index, unsigned int numberOfElements, bool* output
							  );

// Testing functions
// ONLY FOR TESTING THE KERNELS
std::vector<unsigned int> createTransactionsTester(std::vector<std::vector<float>*>* data, unsigned int centroid, float width);
std::vector<unsigned int> createTransactionsReducedReadsTester(std::vector<std::vector<float>*>* data, unsigned int medoid, float width, size_t smem_size);

std::vector<unsigned int> createInitialCandidatesTester(unsigned int dim);
std::tuple<std::vector<unsigned int>,std::vector<float>, std::vector<bool>> countSupportTester(std::vector<std::vector<bool>> candidates, std::vector<std::vector<bool>> itemSet, unsigned int minSupp, float beta);
std::vector<bool> findDublicatesTester(std::vector<std::vector<bool>> candidates, dublicatesType version = dublicatesType::Naive);
std::pair<std::vector<unsigned int>, float> extractMaxTester(std::vector<bool> oldCandidate,
															 unsigned int oldScore, unsigned int oldCentroid,
															 std::vector<std::vector<bool>> newCandidates,
															 std::vector<float> newScores, unsigned int newCentroid,
															 unsigned int index);
std::vector<bool> findPointsInClusterTester(std::vector<bool> candidate, std::vector<std::vector<float>*>* data, unsigned int centroid, float width);
std::vector<bool> disjointClustersTester(std::vector<std::vector<float>*>* data_v, std::vector<unsigned int> centroids_v, std::vector<unsigned int> subspaces_v, std::vector<float> scores_v);

#endif
