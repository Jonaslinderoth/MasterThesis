#include "Runner.h"
#include "ExperimentStub.h"
#include "createTransactions.h"
#include "mergeCandidates.h"
#include "ExperimentCountSupport.h"
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
