#include "Runner.h"
#include "ExperimentStub.h"
#include "createTransactions.h"

int main() {
    auto runner = new Runner("Main");
    auto ex = new CreateTransactionsExperiments("CreateTransactionsSmall", "output");
	runner->addExperiment(ex);
	runner->start();
    delete runner;


	return 0;
}
