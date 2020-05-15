#include <gtest/gtest.h>
#include <vector>



std::vector<std::vector<unsigned int>> accuracy(std::vector<std::vector<std::vector<float>*>*>* labels,std::vector<std::vector<std::vector<float>*>*>* cluster){
	std::vector<std::vector<unsigned int>> result = std::vector<std::vector<unsigned int>>(
		labels->size(),
		std::vector<unsigned int>(labels->size(),0));

	for(unsigned int i = 0; i < labels->size(); i++){ // for each cluster
		for(unsigned int l = 0; l < labels->size(); l++){ // for each cluster
			//entry i,l in confusion matrix
			unsigned int count = 0; 
			for(unsigned int j = 0; j < labels->at(i)->size(); j++){ // for each point
				bool equal = true;
				for(unsigned int k = 0; k < labels->at(i)->at(j)->size(); k++){ // for each dim
					equal &= labels->at(i)->at(j)->at(k) == cluster->at(l)->at(j)->at(k);
					if(!equal){
						break;
					}
				}
				count += equal;
			}
			result.at(i).at(l) = count;
		}		
	}
	return result;
};

TEST(testAccuracy, testSimple){
	auto clusters = new std::vector<std::vector<std::vector<float>*>*>;
	auto labels = new std::vector<std::vector<std::vector<float>*>*>;

	{
		auto point1 = new std::vector<float>{1,2};
		auto point2 = new std::vector<float>{2,3};
		auto cluster1 = new std::vector<std::vector<float>*>{point1,point2};
		auto point3 = new std::vector<float>{3,4};
		auto point4 = new std::vector<float>{4,5};
		auto cluster2 = new std::vector<std::vector<float>*>{point3,point4};
		clusters->push_back(cluster1);
		clusters->push_back(cluster2);
	}

	{
		auto point1 = new std::vector<float>{1,2};
		auto point2 = new std::vector<float>{2,3};
		auto cluster1 = new std::vector<std::vector<float>*>{point1,point2};
		auto point3 = new std::vector<float>{3,4};
		auto point4 = new std::vector<float>{4,5};
		auto cluster2 = new std::vector<std::vector<float>*>{point3,point4};
		labels->push_back(cluster1);
		labels->push_back(cluster2);
	}

	auto res = accuracy(labels, clusters);

	EXPECT_EQ(res.at(0).at(0), 2);
	EXPECT_EQ(res.at(1).at(1), 2);
	
	EXPECT_EQ(res.at(0).at(1), 0);
	EXPECT_EQ(res.at(1).at(0), 0);
}


TEST(testAccuracy, testSimple2){
	auto clusters = new std::vector<std::vector<std::vector<float>*>*>;
	auto labels = new std::vector<std::vector<std::vector<float>*>*>;

	{
		auto point1 = new std::vector<float>{1,2};
		auto point2 = new std::vector<float>{2,3};
		auto cluster1 = new std::vector<std::vector<float>*>{point1,point2};
		auto point3 = new std::vector<float>{3,4};
		auto point4 = new std::vector<float>{4,5};
		auto cluster2 = new std::vector<std::vector<float>*>{point3,point4};
		auto point5 = new std::vector<float>{5,6};
		auto point6 = new std::vector<float>{6,7};
		auto cluster3 = new std::vector<std::vector<float>*>{point5,point6};
		
		clusters->push_back(cluster1);
		clusters->push_back(cluster2);
		clusters->push_back(cluster3);
	}

	{
		auto point1 = new std::vector<float>{1,2};
		auto point2 = new std::vector<float>{2,3};
		auto cluster1 = new std::vector<std::vector<float>*>{point1,point2};
		auto point3 = new std::vector<float>{3,4};
		auto point4 = new std::vector<float>{4,5};
		auto cluster2 = new std::vector<std::vector<float>*>{point3,point4};
		auto point6 = new std::vector<float>{6,7};
		auto point5 = new std::vector<float>{5,6};
		auto cluster3 = new std::vector<std::vector<float>*>{point5,point6};
		labels->push_back(cluster1);
		labels->push_back(cluster2);
		labels->push_back(cluster3);
	}

	auto res = accuracy(labels, clusters);

	EXPECT_EQ(res.at(0).at(0), 2);
	EXPECT_EQ(res.at(1).at(1), 2);
	EXPECT_EQ(res.at(2).at(2), 2);
	
	EXPECT_EQ(res.at(0).at(1), 0);
	EXPECT_EQ(res.at(0).at(2), 0);
	EXPECT_EQ(res.at(1).at(0), 0);
	EXPECT_EQ(res.at(1).at(2), 0);
	EXPECT_EQ(res.at(2).at(0), 0);
	EXPECT_EQ(res.at(2).at(1), 0);
}


TEST(testAccuracy, DISABLED_testSimple3){
	auto clusters = new std::vector<std::vector<std::vector<float>*>*>;
	auto labels = new std::vector<std::vector<std::vector<float>*>*>;

	{
		auto point1 = new std::vector<float>{1,2};
		auto point2 = new std::vector<float>{2,3};
		auto cluster1 = new std::vector<std::vector<float>*>{point1,point2};
		auto point3 = new std::vector<float>{3,4};
		auto point4 = new std::vector<float>{4,5};
		auto cluster2 = new std::vector<std::vector<float>*>{point3,point4};
		auto point5 = new std::vector<float>{5,6};
		auto point6 = new std::vector<float>{6,7};
		auto cluster3 = new std::vector<std::vector<float>*>{point5,point6};
		
		clusters->push_back(cluster1);
		clusters->push_back(cluster2);
		clusters->push_back(cluster3);
	}

	{
		auto point2 = new std::vector<float>{2,3};
		auto point1 = new std::vector<float>{1,2};
		auto point1_1 = new std::vector<float>{9,9};
		auto cluster1 = new std::vector<std::vector<float>*>{point1,point2,point1_1};
		auto point3 = new std::vector<float>{3,4};
		auto point4 = new std::vector<float>{4,5};
		auto cluster2 = new std::vector<std::vector<float>*>{point3,point4};
		auto point6 = new std::vector<float>{6,7};
		auto point5 = new std::vector<float>{5,6};
		auto cluster3 = new std::vector<std::vector<float>*>{point5,point6};
		labels->push_back(cluster1);
		labels->push_back(cluster2);
		labels->push_back(cluster3);
	}

	auto res = accuracy(labels, clusters);

	EXPECT_EQ(res.at(0).at(0), 2);
	EXPECT_EQ(res.at(1).at(1), 2);
	EXPECT_EQ(res.at(2).at(2), 2);
	
	EXPECT_EQ(res.at(0).at(1), 0);
	EXPECT_EQ(res.at(0).at(2), 0);
	EXPECT_EQ(res.at(1).at(0), 0);
	EXPECT_EQ(res.at(1).at(2), 0);
	EXPECT_EQ(res.at(2).at(0), 0);
	EXPECT_EQ(res.at(2).at(1), 0);
}
