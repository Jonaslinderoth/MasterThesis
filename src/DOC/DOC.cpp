#include "DOC.h"
#include "HyperCube.h"
#include <iostream>
#include <math.h>       /* log */
#include <algorithm>

DOC::DOC(){

	DOC(std::vector<std::vector<float>>());
}

DOC::DOC(std::vector<std::vector<float>> input){
	data = input;
}

std::pair<std::vector<std::vector<float>>, std::vector<bool>> DOC::findCluster() {
	float a = 0.3;
	float b = 0.3;
	float w = 5;
	float d = this->data.at(0).size();
	float r = log2(2*d)/log2(1/(2*b));
	float m = pow((2/a),2) * log(4);

	auto resD = std::vector<bool>();
	auto resC = std::vector<std::vector<float>>();
	float maxValue = 0;

	for(int i = 1; i < (float)2/a; i++){
		auto p = this->pickRandom();
		for(int j = 1; j < m; j++){


			std::vector<std::vector<float>> X = std::vector<std::vector<float>>();
			for(int k = 0; k < r; k++){

				X.push_back(this->pickRandom());
			}
			auto D = this->findDimensions(p,X,w);
			auto C = std::vector<std::vector<float>>();
			auto b_pD = HyperCube(p,w,D);
			for(int l = 0; l < this->data.size(); l++){
				auto point = this->data.at(l);
				if (b_pD.pointContained(point)){
					C.push_back(point);
				}

			}
			int Dsum = 0;
			std::for_each(D.begin(), D.end(), [&] (int n) {
			    Dsum += 1;
			});
			float current = mu(C.size(), Dsum);
			if(current > maxValue){
				resD = D;
				resC = C;
				maxValue = current;
			}

		}
	}

	auto result = std::make_pair(resC, resD);
	return result;
}

std::vector<float> DOC::findKClusters(int k) {
	std::vector<float> res;
	return res;
}

bool DOC::addPoint(std::vector<float> point) {
	data.push_back(point);
	return true;
}

int DOC::size() {
	return data.size();
}

std::vector<float> DOC::pickRandom() {
	int i = this->randInt(0,this->size()-1,1).at(0);
	return this->data.at(i);
}

std::vector<bool> DOC::findDimensions(std::vector<float> centroid,
		std::vector<std::vector<float> > points, float width) {
	std::vector<bool> result = std::vector<bool>(centroid.size(), true);
	for(int i = 0; i < points.size(); i++){
		for(int j = 0; j < centroid.size(); j++){
			result.at(j) = result.at(j) && (centroid.at(j)-points.at(i).at(j) < width);
		}
	}
	return result;
}


