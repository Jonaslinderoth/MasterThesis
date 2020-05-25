#ifndef EXPERIMENTMINECLUSSPEEDLARGE_H
#define EXPERIMENTMINECLUSSPEEDLARGE_H

#include "Experiment.h"

class ExperimentClusteringSpeed: public Experiment{
 public:
 ExperimentClusteringSpeedLarge(std::string name, std::string dir): Experiment(0, name, dir, "Number of Clusters, Number of Points, Dim, Dim in Cluster, Version, Time, Clusters Found, Accuracy"){
		//this->addTests(numberOfTests);
	};
	void start() override;
 private:
};


#endif
