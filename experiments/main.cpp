#include "Runner.h"
#include "ExperimentStub.h"
#include "createTransactions.h"
#include "mergeCandidates.h"
#include "ExperimentCountSupport.h"
#include "ExperimentRemoveDuplicates.h"

int main() {
    auto runner = new Runner("Main");
	for(unsigned int i = 0; i < 10; i++){
		auto ex = new CreateTransactionsExperiments("CreateTransactions", "output");
		auto ex2 = new MergeCandidatesExperiment("MergeCandidates", "output", 11);
		auto ex3 = new ExperimentCountSupport("CountSupport", "output");
		auto ex4 = new ExperimentRemoveDuplicates("findDuplicates", "output");

		runner->addExperiment(ex2);
		runner->addExperiment(ex4);
		runner->addExperiment(ex3);
		runner->addExperiment(ex);

	}
	runner->start();
    delete runner;


	return 0;
}
