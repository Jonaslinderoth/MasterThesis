#include "Runner.h"
#include "ExperimentStub.h"
#include "createTransactions.h"
#include "mergeCandidates.h"
#include "mergeCandidatesSmem.h"
#include "ExperimentCountSupport.h"
<<<<<<< HEAD
#include "ExperimentRemoveDuplicates.h"
#include "TerminalComandExperiment.h"

int main() {
    auto runner = new Runner("Main");
	if(system("mkdir output   >>/dev/null 2>>/dev/null")){
		
	};
	for(unsigned int i = 0; i < 10; i++){
		auto ex = new CreateTransactionsExperiments("CreateTransactions", "output");
		auto ex2 = new MergeCandidatesExperiment("MergeCandidates", "output", 11);
		auto ex3 = new ExperimentCountSupport("CountSupport", "output");
		auto ex4 = new ExperimentRemoveDuplicates("findDuplicates", "output");
		auto ex5 = new MergeCandidatesExperimentSmem("MergeCandidatesSmem", "output", 11);
		auto ex6 = new TerminalComandExperiment("cd bin; nvprof --csv --log-file ../output/MineClusNaiveSmallTest_"+std::to_string(i)+".csv ./experimentMineClusNaive", "", "output");
		auto ex7 = new TerminalComandExperiment("cd bin; nvprof --csv --log-file ../output/MineClusNaiveMnistTest_"+std::to_string(i)+".csv ./experimentMineClusNaiveMnist", "", "output");
		// Too slow.... :(
		//auto ex7 = new TerminalComandExperiment("cd bin; nvprof --csv --log-file ../output/MineClusNaiveMediumTest_"+std::to_string(i)+".csv ./experimentMineClusNaiveMedium", "", "output");
		
		runner->addExperiment(ex5);
		runner->addExperiment(ex6);
		runner->addExperiment(ex7);
		runner->addExperiment(ex2);
		runner->addExperiment(ex4);
		runner->addExperiment(ex3);
		runner->addExperiment(ex);

	}
	runner->start();
    delete runner;
=======
#include "breakingIntervallExperiment.h"
#include "BreakingIntervallExperimentAllOutside.h"
#include "fakeExperiment.h"

int main() {
    auto runner = new Runner("Main");
    //jonas if it is false then i forgot to change it back to true for you , just change it back to true.
    bool canRunFancyStuff = false;

    //fake experiment
    auto mex0 = new fakeExperiment("fakeExperiment", "output");
    runner->addExperiment(mex0);

    if(canRunFancyStuff){
    	auto ex = new CreateTransactionsExperiments("CreateTransactions", "output");
		auto ex2 = new MergeCandidatesExperiment("MergeCandidates", "output", 11);
		auto ex3 = new ExperimentCountSupport("CountSupport", "output");

		runner->addExperiment(ex3);
		runner->addExperiment(ex2);
		runner->addExperiment(ex);
    }

>>>>>>> 71b2f6c0af5e7fdd7af15c4f24125b648ea00ac6

    bool experimentsThatIDontTestRightNow = false;
    if(experimentsThatIDontTestRightNow){
    	auto mex1 = new breakingIntervallExperiment("BreakingIntervallExperiment", "output");
		runner->addExperiment(mex1);
    }
    auto mex2 = new BreakingIntervallExperimentAllOutside("BreakingIntervallExperimentAllOutside", "output");
    runner->addExperiment(mex2);

    runner->start();
    delete runner;

	return 0;
}

