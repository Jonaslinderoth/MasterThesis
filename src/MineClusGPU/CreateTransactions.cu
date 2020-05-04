#include "CreateTransactions.h"


/*
  Naive kernel for creating the itemSet. 
  Takes the data, and a index for a centroid, and creates the itemSet 
  The items are stored with the points as columns, and the dimensions as rows, 
  and then a row major fasion
*/
__global__ void createTransactions(float* data, unsigned int dim, unsigned int numberOfPoints, unsigned int centroidId, float width, unsigned int* output){
	unsigned int point = blockIdx.x*blockDim.x+threadIdx.x;
	if(point < numberOfPoints){
		unsigned int numberOfOutputs = ceilf((float)dim/32);
		// For each of the blocks in the output in this dimension
		for(unsigned int i = 0; i < numberOfOutputs; i++){
			unsigned int output_block = 0;
			// for each bit in a block
			for(unsigned int j = 0; j < 32; j++){
				// break if the last block dont line up with 32 bits
				if(i == numberOfOutputs-1 && j == dim%32 && j != 0){
					break;
				}else{
					assert(dim*centroidId+i*32+j < numberOfPoints*dim);
					assert(point*dim+i*32+j < numberOfPoints*dim);
					// Check if the dimension are within the width, and write to block in register
					output_block |= ((abs(data[dim*centroidId+i*32+j] - data[point*dim+i*32+j]) < width) << j);
				}
			}
			// write block to global memory.
			assert(numberOfPoints*i+point < numberOfPoints*ceilf((float)dim/32));
			output[numberOfPoints*i+point] = output_block;
		}	
	}
}

/*
 This function should return the size of the dynamically allocated shared memory
 https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-dynamic-smem-size
*/
__device__ unsigned dynamic_smem_size ()
{
    unsigned ret; 
    asm volatile ("mov.u32 %0, %dynamic_smem_size;" : "=r"(ret));
    return ret;
}

/*
  Reduced reads kernel for finding the initial candidates
  Takes the data, and a index for a medoid, and creates the transaction set 
  The transactions are stored with the points as columns, and the dimensions as rows, 
  and then a row major fasion
*/
__global__ void createTransactionsReducedReads(float* data, unsigned int dim, unsigned int numberOfPoints, unsigned int medoidId, float width, unsigned int* output){
	extern __shared__ float medoid[];

	unsigned int number_of_dims_in_smem = dynamic_smem_size()/sizeof(float);
	assert(min(number_of_dims_in_smem,dim) <= dim);
	unsigned int number_of_iterations_for_smem = ceilf((float)dim/number_of_dims_in_smem);
	unsigned int dims_processed = 0;
	unsigned int dims_to_process = 0;
	while(dims_processed < dim){
		// compute how many iterations till done
		if((dims_processed + number_of_dims_in_smem) < dim){
			dims_to_process = number_of_dims_in_smem;
			// rounding down to nearest mutple of 32;
			dims_to_process = dims_to_process/32;
			dims_to_process = dims_to_process*32;
		}else{
			dims_to_process = dim-dims_processed;
		}
		
		// load the first chunck (up to 12000 floats) into shared memory
		for(unsigned int i = 0; i < ceilf((float)dims_to_process/blockDim.x); i++){
			size_t id = dims_processed+i*blockDim.x+threadIdx.x;
			if(i*blockDim.x+threadIdx.x < dims_to_process){
				medoid[i*blockDim.x+threadIdx.x] = data[dim*medoidId+id];
			}
		}

		__syncthreads();
		unsigned int point = blockIdx.x*blockDim.x+threadIdx.x;
		if(point < numberOfPoints){
			unsigned int numberOfOutputs = ceilf((float)dims_to_process/32);
			// For each of the blocks in the output in this dimension
			for(unsigned int i = 0; i < numberOfOutputs; i++){
				unsigned int output_block = 0;
				// for each bit in a block
				for(unsigned int j = 0; j < 32; j++){
					// break if the last block dont line up with 32 bits
					if(dims_processed+i*32+j >= dim){
						// a break is okay, since the block is set to 0 at initialisation
						break;
					}else{
						assert(point*dim+dims_processed+i*32+j < numberOfPoints*dim);
						assert(i*32+j < number_of_dims_in_smem);
						// Check if the dimension are within the width, and write to block in register
						output_block |= ((abs(medoid[i*32+j] - data[point*dim+dims_processed+i*32+j]) < width) << j);
					}
				}
				// write block to global memory.
				assert(numberOfPoints*(i+dims_processed/32)+point < numberOfPoints*ceilf((float)dim/32));
				output[numberOfPoints*(i+dims_processed/32)+point] = output_block;
			}	
		}
		dims_processed += dims_to_process;
		__syncthreads();
	}
}



/**
Thin wrapper for createTransactions
*/
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
							   transactionsType version){
	if(version == transactionsType::Naive_trans){
		createTransactions<<<dimGrid, dimBlock, smem, stream>>>(data, dim, numberOfPoints, centroidId, width, output);
	}else{
		createTransactionsReducedReads<<<dimGrid, dimBlock, smem, stream>>>(data, dim, numberOfPoints, centroidId, width, output);		
	}
};



/**
   This function is only for testing that the kernel works correctly
*/
std::vector<unsigned int> createTransactionsTester(std::vector<std::vector<float>*>* data, unsigned int centroid, float width){
	uint size = data->size();
	uint dim = data->at(0)->size();
	uint size_of_data = size*dim*sizeof(float);
	size_t size_of_output = size*ceilf((float)dim/32)*sizeof(unsigned int);
	float* data_h;
	unsigned int* output_h;
	cudaMallocHost((void**) &data_h, size_of_data);
	cudaMallocHost((void**) &output_h, size_of_output);
	for(int i = 0; i < size; i++){
		for(int j = 0; j < dim; j++){
			data_h[i*dim+j] = data->at(i)->at(j);
		}
	}
	float* data_d;
	cudaMalloc((void**) &data_d, size_of_data);
	cudaMemcpy(data_d, data_h, size_of_data, cudaMemcpyHostToDevice);
		
	unsigned int* output_d;
	cudaMalloc((void**) &output_d, size_of_output);

	unsigned int dimBlock = 1024;
	unsigned int dimGrid = ceilf((float)size/dimBlock);
	createTransactions<<<dimGrid, dimBlock>>>(data_d, dim, size, centroid, width, output_d);

	
	
	cudaMemcpy(output_h, output_d, size_of_output, cudaMemcpyDeviceToHost);
	std::vector<unsigned int> res;
	for(int i = 0; i < ceilf((float)dim/32)*size;i++){
		res.push_back(output_h[i]);
	}
	return res;
}

/**
   This function is only for testing that the kernel works correctly
   This is for the reduced reads version
*/
std::vector<unsigned int> createTransactionsReducedReadsTester(std::vector<std::vector<float>*>* data, unsigned int centroid, float width,size_t smem_size){
	uint size = data->size();
	uint dim = data->at(0)->size();
	uint size_of_data = size*dim*sizeof(float);
	size_t size_of_output = size*ceilf((float)dim/32)*sizeof(unsigned int);
	float* data_h;
	unsigned int* output_h;
	cudaMallocHost((void**) &data_h, size_of_data);
	cudaMallocHost((void**) &output_h, size_of_output);
	for(int i = 0; i < size; i++){
		for(int j = 0; j < dim; j++){
			data_h[i*dim+j] = data->at(i)->at(j);
		}
	}
	float* data_d;
	cudaMalloc((void**) &data_d, size_of_data);
	cudaMemcpy(data_d, data_h, size_of_data, cudaMemcpyHostToDevice);
		
	unsigned int* output_d;
	cudaMalloc((void**) &output_d, size_of_output);

	unsigned int dimBlock = 1024;
	unsigned int dimGrid = ceilf((float)size/dimBlock);
	
	createTransactionsReducedReads<<<dimGrid, dimBlock, smem_size>>>(data_d, dim, size, centroid, width, output_d);

	
	
	cudaMemcpy(output_h, output_d, size_of_output, cudaMemcpyDeviceToHost);
	std::vector<unsigned int> res;
	for(int i = 0; i < ceilf((float)dim/32)*size;i++){
		res.push_back(output_h[i]);
	}
	return res;
}



