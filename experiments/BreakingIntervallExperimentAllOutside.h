/*
 * BreakingIntervallExperimentAllOutside.h
 *
 *  Created on: May 14, 2020
 *      Author: mikkel
 */

#ifndef BREAKINGINTERVALLEXPERIMENTALLOUTSIDE_H_
#define BREAKINGINTERVALLEXPERIMENTALLOUTSIDE_H_


#include "Experiment.h"
class BreakingIntervallExperimentAllOutside: public Experiment{
 public:
	BreakingIntervallExperimentAllOutside(std::string name, std::string dir): Experiment(0,name, dir, "number of points, dim, version, break interval, time"){
	}
	void start() override;
};





#endif /* BREAKINGINTERVALLEXPERIMENTALLOUTSIDE_H_ */
