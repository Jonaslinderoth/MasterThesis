#include "HashTable.h"
#include "assert.h"
#include "iostream"
#include "../randomCudaScripts/Utils.h"
//https://nosferalatu.com/SimpleGPUHashTable.html

__device__ unsigned long long int hashFun(unsigned long long int k)
{
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return k;
}

__device__ unsigned int hashFun(unsigned int k)
{
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return k;
}


// Create a hash table. For linear probing, this is just an array of KeyValues
KeyValue* create_hashtable(unsigned int size, cudaStream_t stream) 
{
    // Allocate memory
    KeyValue* hashtable;
    checkCudaErrors(cudaMalloc((void**) &hashtable, sizeof(KeyValue) * size));

    // Initialize hash table to empty
    static_assert(kEmpty == 0xffffffffffffffff, "memset expected kEmpty=0xffffffffffffffff");
    checkCudaErrors(cudaMemsetAsync(hashtable, 0xff, sizeof(KeyValue) * size, stream));

    return hashtable;
}


KeyValue* create_hashtable(unsigned int size) 
{
    // Allocate memory
    KeyValue* hashtable;
    checkCudaErrors(cudaMalloc((void**) &hashtable, sizeof(KeyValue) * size));

    // Initialize hash table to empty
    static_assert(kEmpty == 0xffffffffffffffff, "memset expected kEmpty=0xffffffffffffffff");
    checkCudaErrors(cudaMemset(hashtable, 0xff, sizeof(KeyValue) * size));

    return hashtable;
}



__global__ void insertHashTable(KeyValue* hashTable, unsigned int* candidates, unsigned int hashTableSize, unsigned int numberOfCandidates, unsigned int dim, bool* alreadyDeleted){
	unsigned int candidate = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int numberOfBlocks = ceilf((float)dim/32);
	unsigned int primes[10] = {3,5,7,11,13,17,19,23,27,31};
	//unsigned int primes[10] = {1,2,3,4,5,6,7,8,9,10};
	if(candidate < numberOfCandidates && !alreadyDeleted[candidate]){
		unsigned int hash = 0;
		unsigned long long int key = 0;
		unsigned int value = candidate;
		unsigned int block = 0;
		unsigned int multiplier = 0;
		for(unsigned int i = 0; i < numberOfBlocks; i++){
			block = candidates[candidate+numberOfCandidates*i];
			multiplier = hashFun(i+1);
			key += (unsigned long long int)hashFun((unsigned long long int)block*multiplier); // i should really be a prime
			hash ^= hashFun(block);
		}
		hash = hash & (hashTableSize-1); // most significant bits are removed to make it an int
		unsigned int count =0;
		while (true){
			assert(hash < hashTableSize);
			unsigned long long int prev = atomicCAS(&hashTable[hash].key, kEmpty, key);
			
			// printf("prev %llu \n", prev);
			if (prev == kEmpty){
				// printf("Inserted: hash %u, value %u, key %llu \n", hash, value, key);
				hashTable[hash].value = value;
				return;
			}
			if (prev == key){
				// printf("Already inserted: hash %u, value %u, key %llu \n", hash, value,key);
				return;
			}
			count++;
			// printf("Collision %u, candidate %u, hash %u, key %llu\n", count, candidate, hash, key);
			hash = (hash + 1) & (hashTableSize-1);
		}
	}
}



__global__ void lookupHashTable(KeyValue* hashTable, unsigned int* candidates, unsigned int hashTableSize, unsigned int numberOfCandidates, unsigned int dim, bool* output){
	unsigned int candidate = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int numberOfBlocks = ceilf((float)dim/32);
	unsigned int primes[10] = {3,5,7,11,13,17,19,23,27,31};
	//unsigned int primes[10] = {1,2,3,4,5,6,7,8,9,10};
	if(candidate < numberOfCandidates){
		unsigned int hash = 0;
		unsigned long long int key = 0;
		unsigned int block = 0;
		unsigned int multiplier = 1;
		for(unsigned int i = 0; i < numberOfBlocks; i++){
			multiplier = hashFun(i+1);
			block = candidates[candidate+numberOfCandidates*i];
			key += (unsigned long long int)hashFun((unsigned long long int)block*multiplier); // i should really be a prime
			hash ^= hashFun(block);
		}
		hash = hash & (hashTableSize-1); // most significant bits are removed to make it an int

		while (true){
            if (hashTable[hash].key == key){
				output[candidate] = candidate != hashTable[hash].value;
                return;
            }
            if (hashTable[hash].key == kEmpty){
                output[candidate] = false;
                return;
            }
            hash = (hash + 1) & (hashTableSize - 1);
        }
	}
}


void findDublicatesHashTableTester(unsigned int dimGrid, unsigned int dimBlock,
								   unsigned int* candidates, unsigned int numberOfCandidates,
								   unsigned int dim, bool* alreadyDeleted_d, bool* output_d){
	
	unsigned int sizeOfHashTable = pow(2,ceilf(log2(numberOfCandidates))+1);
	assert(sizeOfHashTable > numberOfCandidates);
	auto hashTable = create_hashtable(sizeOfHashTable);
	insertHashTable<<<dimGrid, dimBlock>>>(hashTable, candidates, sizeOfHashTable, numberOfCandidates, dim, alreadyDeleted_d);
	lookupHashTable<<<dimGrid, dimBlock>>>(hashTable, candidates, sizeOfHashTable, numberOfCandidates, dim, output_d);
}


void findDublicatesHashTableWrapper(unsigned int dimGrid, unsigned int dimBlock,cudaStream_t stream,
								   unsigned int* candidates, unsigned int numberOfCandidates,
								   unsigned int dim, bool* alreadyDeleted_d, bool* output_d){
	assert(dimGrid>=1);
	unsigned int sizeOfHashTable = max((unsigned int)pow(2,ceilf(log2(numberOfCandidates))+1), 4096 /*2^10*/);
	//std::cout << "sizeOfHashTable " << sizeOfHashTable << " what it sould be: " << ceilf(log2(numberOfCandidates)) << " number of candidates: " << numberOfCandidates << std::endl;
	assert(sizeOfHashTable > numberOfCandidates);
	auto hashTable = create_hashtable(sizeOfHashTable, stream);
	insertHashTable<<<dimGrid, dimBlock, 0, stream>>>(hashTable, candidates, sizeOfHashTable, numberOfCandidates, dim, alreadyDeleted_d);
	lookupHashTable<<<dimGrid, dimBlock, 0, stream>>>(hashTable, candidates, sizeOfHashTable, numberOfCandidates, dim, output_d);
	checkCudaErrors(cudaStreamSynchronize(stream));
	checkCudaErrors(cudaFree(hashTable));
}
