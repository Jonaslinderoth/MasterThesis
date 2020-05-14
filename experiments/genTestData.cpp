#include "../src/testingTools/DataGeneratorBuilder.h"
#include <vector>
#include <iostream>     // std::cout
#include <fstream>      // std::ifstream

void inline genTestData(){
	if(system("mkdir testData  >>/dev/null 2>>/dev/null")){
		
	};
	// 10 clusters of 10000 points in 100 dimesnions 5% outliers ~40 mb
	{
		DataGeneratorBuilder dgb;
		bool res = dgb.buildUClusters("testData/smallDataSet",10000,10,15,100,10,5, false);
		if(!res) std::cout << "ERROR" << std::endl;
	}
	
	// 10 clusters of 10000 points in 1000 dimesnions 5% outliers ~400 mb
	{
		DataGeneratorBuilder dgb;
		bool res = dgb.buildUClusters("testData/mediumDataSet",10000,10,15,1000,10,5, false);
		if(!res) std::cout << "ERROR" << std::endl;
	}

	// // 10 clusters of 100000 points in 1000 dimesnions 5% outliers ~4 gb
	// {
	// 	DataGeneratorBuilder dgb;
	// 	bool res = dgb.buildUClusters("testData/largeDataSet",100000,10,15,1000,10,5, false);
	// 	if(!res) std::cout << "ERROR" << std::endl;
	// }

	// // 10 clusters of 100000 points in 1000 dimesnions 5% outliers ~10 gb
	// {
	// 	DataGeneratorBuilder dgb;
	// 	bool res = dgb.buildUClusters("testData/hugeDataSet",100000,25,15,1000,10,5, false);
	// 	if(!res) std::cout << "ERROR" << std::endl;
	//
};	

int inline reverseInt (int i) 
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
};

inline std::vector<std::vector<float>*>* getMnist(){
	system("mkdir testData >>/dev/null 2>>/dev/null");
	system("wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -O testData/t10k-images-idx3-ubyte.gz >>/dev/null 2>>/dev/null");
	system("gunzip -f -q testData/t10k-images-idx3-ubyte.gz >>/dev/null 2>>/dev/null");
	std::ifstream file (/*full_path*/"testData/t10k-images-idx3-ubyte");
	auto result = new std::vector<std::vector<float>*>;
    if (file.is_open())
		{
			int magic_number=0;
			int number_of_images=0;
			int n_rows=0;
			int n_cols=0;
			file.read((char*)&magic_number,sizeof(magic_number)); 
			magic_number= reverseInt(magic_number);
			file.read((char*)&number_of_images,sizeof(number_of_images));
			number_of_images= reverseInt(number_of_images);
			file.read((char*)&n_rows,sizeof(n_rows));
			n_rows= reverseInt(n_rows);
			file.read((char*)&n_cols,sizeof(n_cols));
			n_cols= reverseInt(n_cols);
			for(int i=0;i<number_of_images;++i)
				{
					auto point = new std::vector<float>;
					for(int r=0;r<n_rows;++r)
						{
							for(int c=0;c<n_cols;++c)
								{
									unsigned char temp=0;
									file.read((char*)&temp,sizeof(temp));
									point->push_back(temp);
								}
						}
					result->push_back(point);
				}
		}
	return result;
};


