#ifndef EVALUATION_H
#define EVALUATION_H
#include <vector>
#include<string>
class Evaluation{
 public:
	static bool pointEq(std::vector<float>* a, std::vector<float>* b);
	static bool pointInCluster(std::vector<std::vector<float>*>* cluster, std::vector<float>* point);
	static std::vector<std::vector<unsigned int>> confusion(std::vector<std::vector<std::vector<float>*>*>* labels,
															std::vector<std::vector<std::vector<float>*>*>* clusters);
	
	static std::vector<std::vector<unsigned int>> confusion(std::vector<std::vector<std::vector<float>*>*>* labels,
															std::vector<std::pair<std::vector<std::vector<float>*>*,std::vector<bool>*>> clusters);

	static float accuracy(std::vector<std::vector<unsigned int>> confusion);

	
	// Path is the path is for a cluster and a metadata file
	static std::vector<std::vector<std::vector<float>*>*>* getCluster(std::string path);


 private:
	Evaluation(){};
};

#endif
