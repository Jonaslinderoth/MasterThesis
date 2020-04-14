#ifndef HASHTABLE_H
#define HASHTABLE_H
//https://nosferalatu.com/SimpleGPUHashTable.html
struct KeyValue
{
    unsigned long long int key;
    unsigned int value;
};

KeyValue* create_hashtable(unsigned int sizeOfHashTable, cudaStream_t stream);
KeyValue* create_hashtable(unsigned int sizeOfHashTable);

void findDublicatesHashTableTester(unsigned int dimGrid, unsigned int dimBlock,
							  unsigned int* candidates, unsigned int numberOfCandidates,
								   unsigned int dim, bool* alreadyDeleted_d, bool* output_d);

void findDublicatesHashTableWrapper(unsigned int dimGrid, unsigned int dimBlock,cudaStream_t stream,
							  unsigned int* candidates, unsigned int numberOfCandidates,
								   unsigned int dim, bool* alreadyDeleted_d, bool* output_d);

const size_t kEmpty = 0xffffffffffffffff;




#endif
