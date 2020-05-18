__global__ void dummyKernel(){
	unsigned int count = 0;
	for(unsigned int i = 0; i < 1000; i++){
		count += i;
	}
}


void dummyKernelWrapper(){
	dummyKernel<<<1,1>>>();
}