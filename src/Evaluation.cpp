#include "Evaluation.h"
#include "../src/testingTools/DataGeneratorBuilder.h"
#include "../src/dataReader/Cluster.h"
#include "../src/dataReader/DataReader.h"
#include "../src/testingTools/MetaDataFileReader.h"
#include "../src/testingTools/RandomFunction.h"

bool Evaluation::pointEq(std::vector<float>* a, std::vector<float>* b){
	bool result = true;
	result &= a->size() == b->size();
	for(unsigned int i = 0; i < a->size(); i++){
		result &= abs(a->at(i) - b->at(i)) <= 0.000001;
		if (!result) return result;
	}
	return result;
}

bool Evaluation::pointInCluster(std::vector<std::vector<float>*>* cluster, std::vector<float>* point){
	for(unsigned int i = 0; i < cluster->size(); i++){
		if(pointEq(cluster->at(i), point)){
			return true;
		}
	}
	return false;
}


std::vector<std::vector<unsigned int>> Evaluation::confusion(std::vector<std::vector<std::vector<float>*>*>* labels,std::vector<std::vector<std::vector<float>*>*>* clusters){
	std::vector<std::vector<unsigned int>> result = std::vector<std::vector<unsigned int>>(
		labels->size(),
		std::vector<unsigned int>(clusters->size(),0));

	for(unsigned int i = 0; i < labels->size(); i++){ // for each cluster
		for(unsigned int l = 0; l < clusters->size(); l++){ // for each cluster
			//entry i,l in confusion matrix
			unsigned int count = 0;
			for(unsigned int j = 0; j < clusters->at(l)->size(); j++){
				count += pointInCluster(labels->at(i), clusters->at(l)->at(j));
			}
			result.at(i).at(l) = count;
		}		
	}
	return result;
};

std::vector<std::vector<unsigned int>> Evaluation::confusion(std::vector<std::vector<std::vector<float>*>*>* labels,
												 std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*>> clusters){
	std::vector<std::vector<unsigned int>> result = std::vector<std::vector<unsigned int>>(
		labels->size(),
		std::vector<unsigned int>(clusters.size(),0));

	for(unsigned int i = 0; i < labels->size(); i++){ // for each cluster
		for(unsigned int l = 0; l < clusters.size(); l++){ // for each cluster
			//entry i,l in confusion matrix
			unsigned int count = 0;
			for(unsigned int j = 0; j < clusters.at(l).first->size(); j++){
				count += pointInCluster(labels->at(i), clusters.at(l).first->at(j));
			}
			result.at(i).at(l) = count;
		}		
	}
	return result;
};

float Evaluation::accuracy(std::vector<std::vector<unsigned int>> confusion){
	float result = 0;
	unsigned int trues = 0;
	unsigned int total = 0;
	for(unsigned int i = 0; i < confusion.size(); i++){ // for each label
		unsigned int maxIndex = 0; 
		for(unsigned int l = 0; l < confusion.at(i).size(); l++){ // for each cluster
			if(confusion.at(i).at(l) > confusion.at(i).at(maxIndex)){
				maxIndex = l;
			}
			total += confusion.at(i).at(l);
		}

		trues += confusion.at(i).at(maxIndex);
	}
	result += (float)trues/total;
	return result;	
}



std::vector<std::vector<std::vector<float>*>*>* Evaluation::getCluster(std::string path){
	auto dr = new DataReader(path);
	auto mdr = new MetaDataFileReader(path);

	std::vector<std::vector<std::vector<float>*>*>* labels = new std::vector<std::vector<std::vector<float>*>*>;
	for(unsigned int i = 0; i < mdr->getClusterLines().size(); i++){
		labels->push_back(new std::vector<std::vector<float>*>);		
	}
	while(dr->isThereANextPoint()){
		labels->at(mdr->nextCheat())->push_back(dr->nextPoint());
	}
	delete dr;
	delete mdr;
	return labels;
}

