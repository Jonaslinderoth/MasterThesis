#include "Runner.h"
#include "ExperimentStub.h"
#include "createTransactions.h"
#include "mergeCandidates.h"
#include "ExperimentCountSupport.h"

int main() {
    auto runner = new Runner("Main");
    auto ex = new CreateTransactionsExperiments("CreateTransactions", "output");
	auto ex2 = new MergeCandidatesExperiment("MergeCandidates", "output", 11);
	auto ex3 = new ExperimentCountSupport("CountSupport", "output");


	runner->addExperiment(ex3);
	runner->addExperiment(ex2);
	runner->addExperiment(ex);
	runner->start();
    delete runner;


	return 0;
}
