/*
 * fakeExperiment.h
 *
 *  Created on: May 13, 2020
 *      Author: mikkel
 */

#ifndef FAKEEXPERIMENT_H_
#define FAKEEXPERIMENT_H_


#include "Experiment.h"
class fakeExperiment: public Experiment{
 public:
	fakeExperiment(std::string name, std::string dir): Experiment(0,name, dir, ""){
	}
	void start() override;
};




#endif /* FAKEEXPERIMENT_H_ */
