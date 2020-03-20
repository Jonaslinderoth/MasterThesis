
#include <iostream>
#include <chrono>
#include <random>
#include <iostream>
#include "src/DOC_GPU/DOCGPU.h"
#include "src/Fast_DOCGPU/Fast_DOCGPU.h"
#include "src/Clustering.h"
#include "test/testData.h"
#include <vector>
#include "src/testingTools/DataGeneratorBuilder.h"
#include "src/dataReader/Cluster.h"
#include "src/dataReader/DataReader.h"
#include "src/testingTools/MetaDataFileReader.h"
#include "src/DOC/HyperCube.h"




int main ()
{
	for(int i = 0; i < 5; i++){
		std::cout << i << std::endl;
		DataGeneratorBuilder dgb;
		dgb.setSeed(3);
		{	Cluster small;
			small.setAmmount(2000);
			small.addDimension(normalDistribution, {-10000,10000}, {50,2});
			small.addDimension(uniformDistribution, {-10000,10000});
			small.addDimension(normalDistribution, {-10000,10000}, {50,2});
			small.addDimension(uniformDistribution, {-10000,10000});
			small.addDimension(normalDistribution, {-10000,10000}, {50,2});
			small.addDimension(uniformDistribution, {-10000,10000});
			small.addDimension(normalDistribution, {-10000,10000}, {50,2});
			small.addDimension(uniformDistribution, {-10000,10000});
			small.addDimension(normalDistribution, {-10000,10000}, {50,2});
			small.addDimension(uniformDistribution, {-10000,10000});
			small.addDimension(normalDistribution, {-10000,10000}, {50,2});
			small.addDimension(uniformDistribution, {-10000,10000});	
			dgb.addCluster(small);
		};

		
		
		{	Cluster small2;
			small2.setAmmount(200);

			small2.addDimension(uniformDistribution, {-10000,10000});
			small2.addDimension(uniformDistribution, {-10000,10000});
			small2.addDimension(uniformDistribution, {-10000,10000});
			small2.addDimension(uniformDistribution, {-10000,10000});
			small2.addDimension(uniformDistribution, {-10000,10000});
			small2.addDimension(uniformDistribution, {-10000,10000});
			small2.addDimension(uniformDistribution, {-10000,10000});
			small2.addDimension(uniformDistribution, {-10000,10000});
			small2.addDimension(uniformDistribution, {-10000,10000});
			small2.addDimension(uniformDistribution, {-10000,10000});
			small2.addDimension(uniformDistribution, {-10000,10000});
			small2.addDimension(uniformDistribution, {-10000,10000});
			dgb.addCluster(small2);};
		
		


		{
			Cluster small;
			small.setAmmount(1000);
			small.addDimension(uniformDistribution, {-10000,10000});
			small.addDimension(uniformDistribution, {-10000,10000});
			small.addDimension(uniformDistribution, {-10000,10000});
			small.addDimension(uniformDistribution, {-10000,10000});
			small.addDimension(uniformDistribution, {-10000,10000});
			small.addDimension(uniformDistribution, {-10000,10000});
			small.addDimension(uniformDistribution, {-10000,10000});
			small.addDimension(normalDistribution, {-10000,10000}, {500,2});
			small.addDimension(normalDistribution, {-10000,10000}, {500,2});
			small.addDimension(normalDistribution, {-10000,10000}, {0,2});
			small.addDimension(uniformDistribution, {-10000,10000});
			small.addDimension(uniformDistribution, {-10000,10000});	
			dgb.addCluster(small);
		}


	
		{
			Cluster small;
			small.setAmmount(1000);
			small.addDimension(uniformDistribution, {-10000,10000});
			small.addDimension(uniformDistribution, {-10000,10000});
			small.addDimension(uniformDistribution, {-10000,10000});
			small.addDimension(normalDistribution, {-10000,10000}, {100,2});
			small.addDimension(uniformDistribution, {-10000,10000});
			small.addDimension(uniformDistribution, {-10000,10000});
			small.addDimension(uniformDistribution, {-10000,10000});
			small.addDimension(uniformDistribution, {-10000,10000});
			small.addDimension(uniformDistribution, {-10000,10000});
			small.addDimension(normalDistribution, {-10000,10000}, {100,2});
			small.addDimension(uniformDistribution, {-10000,10000});
			small.addDimension(uniformDistribution, {-10000,10000});
	
			dgb.addCluster(small);
		}

		{
			Cluster small;
			small.setAmmount(1000);
			small.addDimension(normalDistribution, {-10000,10000}, {200,2});
			small.addDimension(uniformDistribution, {-10000,10000});
			small.addDimension(uniformDistribution, {-10000,10000});
			small.addDimension(normalDistribution, {-10000,10000}, {200,2});
			small.addDimension(uniformDistribution, {-10000,10000});
			small.addDimension(normalDistribution, {-10000,10000}, {200,2});
			small.addDimension(normalDistribution, {-10000,10000}, {200,2});
			small.addDimension(uniformDistribution, {-10000,10000});
			small.addDimension(uniformDistribution, {-10000,10000});
			small.addDimension(normalDistribution, {-10000,10000}, {200,2});
			small.addDimension(uniformDistribution, {-10000,10000});
			small.addDimension(normalDistribution, {-10000,10000}, {200,2});	
			dgb.addCluster(small);
			}
		
	
		dgb.setFileName("test/testData/test6");
		dgb.build(true);
		DataReader* dr = new DataReader("test/testData/test6");
	
		Fast_DOCGPU d(dr);
		d.setSeed(2);
		d.setWidth(10);

		std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> > res = d.findKClusters(5);
	}

	return 0;
}
