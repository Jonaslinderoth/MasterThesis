
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
#include "BreakingIntervallExperimentNormalData.h"
#include "fakeExperiment.h"
#include "PointsContainedDeviceNormalData.h"
#include "ExperimentClusteringSpeed.h"
#include "ExperimentDOCAccuracy.h"
#include "ExperimentFindDimensions.h"
#include "ExperimentDelete.h"
#include "PointsContainedDeviceNormalDataSharedMem.h"
int main() {
    auto runner = new Runner("Main");
	if(system("mkdir output   >>/dev/null 2>>/dev/null")){
	};
	auto mex0 = new fakeExperiment("fakeExperiment", "output");
    runner->addExperiment(mex0);
	auto a1 = new PointsContainedDeviceNormalDataSharedMem("PointsContainedDeviceSharedMemSize", "output");
	runner->addExperiment(a1);
	
	
	for(unsigned int i = 0; i < 10; i++){
		auto runner_inner = new Runner("iteration " + std::to_string(i));
		auto ex123 = new ExperimentDelete("deleteExperiments", "output");
		runner_inner->addExperiment(ex123);
		auto ex = new CreateTransactionsExperiments("CreateTransactions", "output");
		auto ex2 = new MergeCandidatesExperiment("MergeCandidates", "output", 11);
		auto ex3 = new ExperimentCountSupport("CountSupport", "output");
		auto ex4 = new ExperimentRemoveDuplicates("findDuplicates", "output");
		auto ex5 = new MergeCandidatesExperimentSmem("MergeCandidatesSmem", "output", 11);



		auto ex6 = new TerminalComandExperiment("cd bin; nvprof --print-gpu-trace --print-api-trace --csv --log-file ../output/MineClusNaiveSmallTest_"+std::to_string(i)+".csv ./experimentMineClusNaive", "MineClusNaiveSmall", "output");
		auto ex7 = new TerminalComandExperiment("cd bin; nvprof --print-gpu-trace --print-api-trace --csv --log-file ../output/MineClusNaiveMediumTest_"+std::to_string(i)+".csv ./experimentMineClusNaiveMedium", "MineClusNaiveMedium", "output");


		auto ex6_2 = new TerminalComandExperiment("cd bin; nvprof --print-gpu-trace --print-api-trace --csv --log-file ../output/MineClusBestSmallTest_"+std::to_string(i)+".csv ./experimentMineClusNaive", "MineClusBestSmall", "output");
		auto ex7_2 = new TerminalComandExperiment("cd bin; nvprof --print-gpu-trace --print-api-trace --csv --log-file ../output/MineClusBestMediumTest_"+std::to_string(i)+".csv ./experimentMineClusNaiveMedium", "MineClusBestMedium", "output");

		auto ex8 = new TerminalComandExperiment("cd bin; nvprof --print-gpu-trace --print-api-trace --csv --log-file ../output/DOCNaiveSmallTest_"+std::to_string(i)+".csv ./experimentDOCNaive", "DOCNaiveSmall", "output");
		auto ex9 = new TerminalComandExperiment("cd bin; nvprof --print-gpu-trace --print-api-trace --csv --log-file ../output/DOCNaiveMediumTest_"+std::to_string(i)+".csv ./experimentDOCNaiveMedium", "DOCNaiveMedium", "output");
		auto ex8_2 = new TerminalComandExperiment("cd bin; nvprof --print-gpu-trace --print-api-trace --csv --log-file ../output/DOCBestSmallTest_"+std::to_string(i)+".csv ./experimentDOCBest", "DOCBestSmall", "output");
		auto ex9_2 = new TerminalComandExperiment("cd bin; nvprof --print-gpu-trace --print-api-trace --csv --log-file ../output/DOCBestMediumTest_"+std::to_string(i)+".csv ./experimentDOCBestMedium", "DOCBestMedium", "output");

		 
		auto ex10 = new TerminalComandExperiment("cd bin; nvprof --print-gpu-trace --print-api-trace --csv --log-file ../output/FastDOCNaiveSmallTest_"+std::to_string(i)+".csv ./experimentFastDOCNaive", "FastDOCNaive", "output");
		auto ex11 = new TerminalComandExperiment("cd bin; nvprof --print-gpu-trace --print-api-trace --csv --log-file ../output/FastDOCNaiveMediumTest_"+std::to_string(i)+".csv ./experimentFastDOCNaiveMedium", "FastDOCNaiveMedium", "output");
		auto ex10_2 = new TerminalComandExperiment("cd bin; nvprof --print-gpu-trace --print-api-trace --csv --log-file ../output/FastDOCBestSmallTest_"+std::to_string(i)+".csv ./experimentFastDOCBest", "FastDOCBest", "output");
		auto ex11_2 = new TerminalComandExperiment("cd bin; nvprof --print-gpu-trace --print-api-trace --csv --log-file ../output/FastDOCBestMediumTest_"+std::to_string(i)+".csv ./experimentFastDOCBestMedium", "FastDOCBestMedium", "output");


		// auto ex7 = new TerminalComandExperiment("cd bin; nvprof --csv --log-file ../output/MineClusNaiveMnistTest_"+std::to_string(i)+".csv ./experimentMineClusNaiveMnist", "", "output");
		//runner->addExperiment(mex1);
		// runner->addExperiment(mex2);
		auto ex12 = new ExperimentClusteringSpeed("ExperimentClusteringSpeed", "output");
		auto mex1 = new breakingIntervallExperiment("BreakingIntervallExperiment", "output");
		runner_inner->addExperiment(mex1);
		auto mex2 = new BreakingIntervallExperimentAllOutside("BreakingIntervallExperimentAllOutside", "output");
		runner_inner->addExperiment(mex2);
		auto mex3 = new BreakingIntervallExperimentNormalData("BreakingIntervallExperimentNormalData", "output");
		runner_inner->addExperiment(mex3);
		auto mex4 = new PointsContainedDeviceNormalData("PointsContainedDeviceNormalData", "output");
		runner_inner->addExperiment(mex4);
		// runner_inner->addExperiment(ex5);
		// runner_inner->addExperiment(ex12);
		// runner_inner->addExperiment(ex6);
		// runner_inner->addExperiment(ex6_2);
		// runner_inner->addExperiment(ex8);
		// runner_inner->addExperiment(ex10);

		// runner_inner->addExperiment(ex7);
		// runner_inner->addExperiment(ex7_2);
		// runner_inner->addExperiment(ex9);
		// runner_inner->addExperiment(ex11);
		// runner_inner->addExperiment(ex2);
		// runner_inner->addExperiment(ex4);
		// runner_inner->addExperiment(ex3);
		// runner_inner->addExperiment(ex);



		auto ex13 = new ExperimentDOCAccuracy("DOCAccuracy", "output"); 

	
	    
		

		runner_inner->addExperiment(ex13);
		runner_inner->addExperiment(ex5);
		
		runner_inner->addExperiment(ex6);
		runner_inner->addExperiment(ex6_2);
		runner_inner->addExperiment(ex8);
		runner_inner->addExperiment(ex8_2);
		runner_inner->addExperiment(ex10);
		runner_inner->addExperiment(ex10_2);

		runner_inner->addExperiment(ex7);
		runner_inner->addExperiment(ex7_2);
		runner_inner->addExperiment(ex9);
		runner_inner->addExperiment(ex9_2);
		runner_inner->addExperiment(ex11);
		runner_inner->addExperiment(ex11_2);
		
		runner_inner->addExperiment(ex2);
		runner_inner->addExperiment(ex4);
		runner_inner->addExperiment(ex3);
		runner_inner->addExperiment(ex);


		auto ex111 = new ExperimentFindDimensions("FindDim","output");
		runner_inner->addExperiment(ex111);
		runner->addExperiment(runner_inner);

		
	}

	for(unsigned int i = 0; i < 10; i++){
		auto runner_inner = new Runner("iteration " + std::to_string(i));
		auto ex12 = new ExperimentClusteringSpeed("ClusteringSpeed", "output");
		runner_inner->addExperiment(ex12);
		runner->addExperiment(runner_inner);




	}
	runner->start();			
	delete runner;
	
	return 0;
}

