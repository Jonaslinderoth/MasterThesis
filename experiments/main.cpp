#include "Runner.h"
#include "ExperimentStub.h"
#include "createTransactions.h"
#include "mergeCandidates.h"
#include "mergeCandidatesSmem.h"
#include "ExperimentCountSupport.h"
#include "ExperimentRemoveDuplicates.h"
#include "TerminalComandExperiment.h"
#include "breakingIntervallExperiment.h"
#include "BreakingIntervallExperimentAllOutside.h"
#include "fakeExperiment.h"
#include "ExperimentClusteringSpeed.h"

int main() {
    auto runner = new Runner("Main");
	if(system("mkdir output   >>/dev/null 2>>/dev/null")){
	};
	auto mex0 = new fakeExperiment("fakeExperiment", "output");
    runner->addExperiment(mex0);
	for(unsigned int i = 0; i < 10; i++){
		auto runner_inner = new Runner("iteration " + std::to_string(i));
		auto ex = new CreateTransactionsExperiments("CreateTransactions", "output");
		auto ex2 = new MergeCandidatesExperiment("MergeCandidates", "output", 11);
		auto ex3 = new ExperimentCountSupport("CountSupport", "output");
		auto ex4 = new ExperimentRemoveDuplicates("findDuplicates", "output");
		auto ex5 = new MergeCandidatesExperimentSmem("MergeCandidatesSmem", "output", 11);
		auto mex1 = new breakingIntervallExperiment("BreakingIntervallExperiment", "output");
		// auto mex2 = new BreakingIntervallExperimentAllOutside("BreakingIntervallExperimentAllOutside", "output");

		auto ex6 = new TerminalComandExperiment("cd bin; nvprof --print-gpu-trace --print-api-trace --csv --log-file ../output/MineClusNaiveSmallTest_"+std::to_string(i)+".csv ./experimentMineClusNaive", "MineClusNaiveSmall", "output");
		auto ex7 = new TerminalComandExperiment("cd bin; nvprof --print-gpu-trace --print-api-trace --csv --log-file ../output/MineClusNaiveMediumTest_"+std::to_string(i)+".csv ./experimentMineClusNaiveMedium", "MineClusNaiveMedium", "output");

		auto ex6_2 = new TerminalComandExperiment("cd bin; nvprof --print-gpu-trace --print-api-trace --csv --log-file ../output/MineClusBestSmallTest_"+std::to_string(i)+".csv ./experimentMineClusNaive", "MineClusBestSmall", "output");
		auto ex7_2 = new TerminalComandExperiment("cd bin; nvprof --print-gpu-trace --print-api-trace --csv --log-file ../output/MineClusBestMediumTest_"+std::to_string(i)+".csv ./experimentMineClusNaiveMedium", "MineClusBestMedium", "output");

		auto ex8 = new TerminalComandExperiment("cd bin; nvprof --print-gpu-trace --print-api-trace --csv --log-file ../output/DOCNaiveSmallTest_"+std::to_string(i)+".csv ./experimentDOCNaive", "DOCNaiveSmall", "output");
		auto ex9 = new TerminalComandExperiment("cd bin; nvprof --print-gpu-trace --print-api-trace --csv --log-file ../output/DOCNaiveMediumTest_"+std::to_string(i)+".csv ./experimentDOCNaiveMedium", "DOCNaiveMedium", "output");


		auto ex10 = new TerminalComandExperiment("cd bin; nvprof --print-gpu-trace --print-api-trace --csv --log-file ../output/FastDOCNaiveSmallTest_"+std::to_string(i)+".csv ./experimentFastDOCNaive", "FastDOCNaive", "output");
		auto ex11 = new TerminalComandExperiment("cd bin; nvprof --print-gpu-trace --print-api-trace --csv --log-file ../output/FastDOCNaiveMediumTest_"+std::to_string(i)+".csv ./experimentFastDOCNaiveMedium", "FastDOCNaiveMedium", "output");

		auto ex12 = new ExperimentClusteringSpeed("ExperimentClusteringSpeed", "output"); 

		
		// auto ex7 = new TerminalComandExperiment("cd bin; nvprof --csv --log-file ../output/MineClusNaiveMnistTest_"+std::to_string(i)+".csv ./experimentMineClusNaiveMnist", "", "output");
		//runner->addExperiment(mex1);		
		// runner->addExperiment(mex2);		
		runner_inner->addExperiment(ex5);
		runner_inner->addExperiment(ex12);
		runner_inner->addExperiment(ex6);
		runner_inner->addExperiment(ex6_2);
		runner_inner->addExperiment(ex8);
		runner_inner->addExperiment(ex10);

		runner_inner->addExperiment(ex7);
		runner_inner->addExperiment(ex7_2);
		runner_inner->addExperiment(ex9);
		runner_inner->addExperiment(ex11);
		runner_inner->addExperiment(ex2);
		runner_inner->addExperiment(ex4);
		runner_inner->addExperiment(ex3);
		runner_inner->addExperiment(ex);
		runner->addExperiment(runner_inner);
	}
	
	runner->start();			
    delete runner;
	runner->start();
	return 0;
}

