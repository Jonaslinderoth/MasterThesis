#ifndef DOC_h
#define DOC_h
#include <vector>
#include <math.h>

class DOC{
public:
	DOC();
	DOC(std::vector<std::vector<float>> input);
	bool addPoint(std::vector<float> point);
	std::pair<std::vector<std::vector<float>>, std::vector<bool>> findCluster();
	std::vector<float> findKClusters(int k);
	int size();
	std::vector<float> pickRandom();
	std::vector<bool> findDimensions(std::vector<float> centroid, std::vector<std::vector<float>> points, float width);
	float mu(int a, int b){
		return pow(a*((float) 1/0.5),b);
	};
private:
	std::vector<std::vector<float>> data;
};


#endif
