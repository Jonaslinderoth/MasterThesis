/*
 * breakingIntervallExperiment.h
 *
 *  Created on: May 11, 2020
 *      Author: mikkel
 */

#ifndef BREAKINGINTERVALLEXPERIMENT_H_
#define BREAKINGINTERVALLEXPERIMENT_H_


#include "Experiment.h"
class breakingIntervallExperiment: public Experiment{
 public:
	breakingIntervallExperiment(std::string name, std::string dir): Experiment(0,name, dir, "number of points, dim, version, passedTest, time"){
	}
	void start() override;
};




#endif /* BREAKINGINTERVALLEXPERIMENT_H_ */
