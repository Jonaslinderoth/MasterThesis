#include <gtest/gtest.h>
#include <vector>
#include "../src/IterativeClustering.h"
#include "testData.h"
#include "../src/DOC/DOC.h"


TEST(testIterativeClustering, testConstructor){
  std::vector<std::vector<float>*>* data = data_4dim2cluster();
  DOC* d = new DOC();
  IterativeClustering c(d,data);
  SUCCEED();
  
  EXPECT_EQ(d->size(), 0);
  c.findKClusters(0);
  EXPECT_EQ(d->size(), 400);
}


TEST(testIterativeClustering, testConstructor2){
  std::vector<std::vector<float>*>* data = data_4dim2cluster();
  DOC* d = new DOC();
  IterativeClustering c(d,data);
  SUCCEED();
  
  EXPECT_EQ(d->size(), 0);
  auto res = c.findKClusters(1);

  EXPECT_EQ(res.size(), 1);
  EXPECT_EQ(res.at(0).first->size(), 400);
  EXPECT_EQ(res.at(0).second->size(), 4);


  
  EXPECT_EQ(d->size(), 0);
}
