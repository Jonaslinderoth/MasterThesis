#ifndef EXPERIMENTDOCACCURACY_H
#define EXPERIMENTDOCACCURACY_H

#include "Experiment.h"

class ExperimentDOCAccuracy: public Experiment{
 public:
 ExperimentDOCAccuracy(std::string name, std::string dir): Experiment(0, name, dir, "Number of Clusters, Number of Points, Dim, Dim in Cluster, Number of samples, Time, Clusters Found, Accuracy"){
		//this->addTests(numberOfTests);
	};
	void start() override;
 private:
};


#endif
