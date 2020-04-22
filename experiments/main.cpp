#include "Runner.h"
#include "ExperimentStub.h"
#include "createTransactions.h"
#include "mergeCandidates.h"

int main() {
    auto runner = new Runner("Main");
    auto ex = new CreateTransactionsExperiments("CreateTransactions", "output");
	auto ex2 = new MergeCandidatesExperiment("MergeCandidates", "output", 11);
	
	runner->addExperiment(ex2);
	runner->addExperiment(ex);
	runner->start();
    delete runner;


	return 0;
}
