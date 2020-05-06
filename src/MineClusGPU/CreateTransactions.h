#ifndef CREATETRANSACTIONS_H
#define CREATETRANSACTIONS_H
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <assert.h>

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

std::vector<unsigned int> createTransactionsTester(std::vector<std::vector<float>*>* data, unsigned int centroid, float width);
std::vector<unsigned int> createTransactionsReducedReadsTester(std::vector<std::vector<float>*>* data, unsigned int medoid, float width, size_t smem_size);

#endif
