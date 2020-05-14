#include "TerminalComandExperiment.h"
#include <unistd.h>
#include <iostream>

void TerminalComandExperiment::start(){
	Experiment::start();

	Experiment::testDone();
	unsigned int i = 6;
	//std::string a = "cd bin; nvprof --csv --log-file ../output/MineClusNaiveSmallTest__"+std::to_string(i)+".csv ./experimentMineClusNaive";
	const char* buffer2 = this->command.c_str();
	if(system(buffer2)){
		Experiment::repportError("ERROR command not working", this->getName());
	};
	
	Experiment::testDone();

	Experiment::stop();
};
