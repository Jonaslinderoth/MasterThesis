#include "ExperimentStub.h"
#include <unistd.h>
#include <iostream>

void ExperimentStub::start(){
	Experiment::start();
	for(unsigned int i = 0; i < this->getNumberOfTests(); i++){
		usleep(delay);
		if(i%23==0){
            this->repportError("Error stub (Fake error)", this->getName());
		}
		this->writeLineToFile("Hello world");
        Experiment::testDone();
	}
	Experiment::stop();
};




