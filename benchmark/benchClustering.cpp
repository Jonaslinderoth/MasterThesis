#include <benchmark/benchmark.h>
#include "../src/DOC/DOC.h"
#include "../src/DOC_GPU/DOCGPU.h"
#include "../test/testData.h"
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


