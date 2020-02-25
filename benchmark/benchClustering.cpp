#include <benchmark/benchmark.h>
#include "../src/DOC/DOC.h"
#include "../src/DOC_GPU/DOCGPU.h"
#include "../test/testData.h"
#include "../src/testingTools/DataGeneratorBuilder.h"
#include "../src/dataReader/Cluster.h"
#include "../src/dataReader/DataReader.h"
#include "../src/testingTools/MetaDataFileReader.h"

static void BM_DOC(benchmark::State& state) {
	for (auto _ : state){
		std::vector<std::vector<float>*>* data = data_4dim2cluster();
		DOC d = DOC(data, 0.1, 0.25, 5);
		d.setSeed(1);
		std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> res = d.findCluster();
	}
}
// Register the function as a benchmark
BENCHMARK(BM_DOC)->Unit(benchmark::kMillisecond);

static void BM_DOCGPU(benchmark::State& state) {
	for (auto _ : state){
		std::vector<std::vector<float>*>* data = data_4dim2cluster();
		DOCGPU d = DOCGPU(data, 0.1, 0.25, 5);
		d.setSeed(1);
		std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*> res = d.findCluster();
	}
}
// Register the function as a benchmark
BENCHMARK(BM_DOCGPU)->Unit(benchmark::kMillisecond);




static void BM_DOC_find2(benchmark::State& state) {
	for (auto _ : state){
		std::vector<std::vector<float>*>* data = data_4dim2cluster();
		DOC d = DOC(data, 0.1, 0.25, 5);
		d.setSeed(1);
		std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*>> res = d.findKClusters(2);
	}
}
// Register the function as a benchmark
BENCHMARK(BM_DOC_find2)->Unit(benchmark::kMillisecond);

static void BM_DOCGPU_find2(benchmark::State& state) {
	for (auto _ : state){
		std::vector<std::vector<float>*>* data = data_4dim2cluster();
		DOCGPU d = DOCGPU(data, 0.1, 0.25, 5);
		d.setSeed(1);
		std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*>> res = d.findKClusters(2);
	}
}
// Register the function as a benchmark
BENCHMARK(BM_DOCGPU_find2)->Unit(benchmark::kMillisecond);


static void BM_DOC_find10(benchmark::State& state) {
	DataGeneratorBuilder dgb;
	for(int i = 0; i < 3; i++){
		Cluster small;
		small.setAmmount(100);
		for(int j = 0; j < 10; j++){
			if(i == j){
				small.addDimension(uniformDistribution, {10000,10001});
			}else{
				small.addDimension(uniformDistribution, {-10000,10000});
			}
		}
		dgb.addCluster(small);		
	}

	
	dgb.setFileName("test/testData/benchmark1");
	dgb.build(false);
	DataReader* dr = new DataReader("test/testData/benchmark1");
	for (auto _ : state){
		DOC d = DOC(dr);
		d.setSeed(1);
		std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*>> res = d.findKClusters(10);
		std::cout << "found " << res.size() << "clusters" << std::endl;
		for(int i = 0; i < res.size(); i++){
			std::cout << res.at(i).first->size() << std::endl;
		}
	}
}

// Register the function as a benchmark
BENCHMARK(BM_DOC_find10)->Unit(benchmark::kMillisecond);



static void BM_DOCGPU_find10(benchmark::State& state) {
	DataGeneratorBuilder dgb;
	for(int i = 0; i < 3; i++){
		Cluster small;
		small.setAmmount(100);
		for(int j = 0; j < 10; j++){
			if(i == j){
				small.addDimension(uniformDistribution, {10000,10001});
			}else{
				small.addDimension(uniformDistribution, {-10000,10000});
			}
		}
		dgb.addCluster(small);		
	}

	
	dgb.setFileName("test/testData/benchmark1");
	dgb.build(false);
	DataReader* dr = new DataReader("test/testData/benchmark1");
	for (auto _ : state){
		DOCGPU d = DOCGPU(dr);
		d.setSeed(1);
		std::vector<std::pair<std::vector<std::vector<float>*>*, std::vector<bool>*>> res = d.findKClusters(10);
		std::cout << "found " << res.size() << "clusters" << std::endl;
		for(int i = 0; i < res.size(); i++){
			std::cout << res.at(i).first->size() << std::endl;
		}

	}
}
// Register the function as a benchmark
BENCHMARK(BM_DOCGPU_find10)->Unit(benchmark::kMillisecond);
