#include <iostream>
#include <vector>
#include "src/DOC/DOC.h"
int main(){
	std::vector<std::vector<float>*>* data = new std::vector<std::vector<float>*>;
		int a = 0;
		for(float i = 9; i <= 12; i++){
			for(float j = 9; j <= 12; j++){
				std::vector<float>* point1 = new std::vector<float>{i,j};
				data->push_back(point1);
				a++;
			}
		}

		int b = 0;
		for(float i = 60; i <= 65; i++){
			for(float j = 0; j <= 50; j++){
				std::vector<float>* point1 = new std::vector<float>{i,j};
				data->push_back(point1);
				b++;
			}
		}


		DOC d = DOC(data, 0.1, 0.25, 5);
		auto res = d.findCluster();




		delete res.first;
		delete res.second;
		for(int i = 0; i< data->size(); i++){
			delete data->at(i);
		}
		delete data;

}
