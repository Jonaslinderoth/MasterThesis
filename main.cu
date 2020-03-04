
#include <iostream>
#include <chrono>
#include <random>
#include <iostream>
#include "src/DOC_GPU/DOCGPU.h"
#include "src/Clustering.h"
#include "test/testData.h"
#include <vector>
#include "src/testingTools/DataGeneratorBuilder.h"
#include "src/dataReader/Cluster.h"
#include "src/dataReader/DataReader.h"
#include "src/testingTools/MetaDataFileReader.h"
int main ()
{
	DataGeneratorBuilder dgb;
	dgb.setSeed(1);
	for(int i = 0; i < 5; i++){
		Cluster small;
		small.setAmmount(100);
		for(int j = 0; j < 5; j++){
			if((i) == j%5){
				small.addDimension(uniformDistribution, {10000,10002});
			}else{
				small.addDimension(uniformDistribution, {-1000,1000});
			}
		}
		std::cout << small.getOutLierPercentage() << std::endl;
		dgb.addCluster(small);		
	}

	dgb.setFileName("test/testData/benchmark1");
	dgb.build(true);

	
	DataReader* dr2 = new DataReader("test/testData/benchmark1");
	int c = 0;
	while(dr2->isThereANextPoint()){
		auto a = dr2->nextPoint();
		if((a->at(0) >= 1000 && a->at(0) <= 10002) || (a->at(1) >= 1000 && a->at(1) <= 10002) || (a->at(2) >= 1000 && a->at(2) <= 10002) || (a->at(3) >= 1000 && a->at(3) <= 10002)){
			//std::cout << "inside" << std::endl;
		}else{
			for(int j = 0; j < a->size(); j++){
				std::cout << a->at(j) << ", " ;
				//if(j >= 2){break;}
			}
			std::cout << std::endl;			
		}

		/*c++;
		if(c == 1){
			for(int j = 0; j < a->size(); j++){
				std::cout << a->at(j) << ", " ;
				if(j >= 2){break;}
			}
			std::cout << std::endl;				
		}
		*/


	}


	DataReader* dr = new DataReader("test/testData/benchmark1");	
	DOCGPU d = DOCGPU(dr);
	d.setSeed(2);
	std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*>> res = d.findKClusters(10);
	std::cout << "found " << res.size() << "clusters" << std::endl;
	for(int i = 0; i < res.size(); i++){
		std::cout << res.at(i).first->size() << std::endl;
		for(int j = 0; j < res.at(i).second->size(); j++){
			std::cout << res.at(i).second->at(j) << ", ";
		}
		std::cout << std::endl;

		/*
		std::cout << "[ ";
		for(int j = 0; j < res.at(i).first->size(); j++){
			std::cout << "[ ";
			for(int k = 0; k < res.at(i).first->at(j)->size(); k++){
				if(k == res.at(i).first->at(j)->size()-1){
					std::cout << res.at(i).first->at(j)->at(k);					
				}else{
					std::cout << res.at(i).first->at(j)->at(k) << ", ";
				}
			}
			std::cout << " ]," << std::endl;
		}
		std::cout << " ]," << std::endl;
		*/

		
	}
	std::cout << "found " << res.size() << "clusters" << std::endl;
	for(int i = 0; i < res.size(); i++){
		std::cout << res.at(i).first->size() << std::endl;
		for(int j = 0; j < res.at(i).second->size(); j++){
			std::cout << res.at(i).second->at(j) << ", ";
		}
		std::cout << std::endl;
	}

	
}