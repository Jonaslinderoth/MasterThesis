#ifndef CLUSTERING_H
#define CLUSTERING_H
#include <vector>

class Clustering{
 public:
	Clustering(){};
	virtual std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> findCluster(){};
	virtual std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*>> findKClusters(int k){};
	void setSeed(int s){
		this->gen.seed(s);
	}
	virtual ~Clustering(){};
 private:
	std::mt19937 gen;

 protected:
	size_t seed = std::random_device()();
	
	std::vector<int> randInt(int upper, int n){
		return randInt(0,upper, n);
	};
	std::vector<int> randInt(int lower, int upper, int n){
		std::vector<int> res = std::vector<int>();
				
		//std::random_device rd;  //Will be used to obtain a seed for the random number engine
		//std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
		std::uniform_int_distribution<> dis(lower, upper); //inclusive

		for(int i = 0; i < n; i++){
			auto a = dis(gen);

			res.push_back(a);
		}
		return res;
	};
	
	
};

#endif
