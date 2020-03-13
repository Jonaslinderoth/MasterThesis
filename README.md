# MasterThesis


## Notes for Iterative Projected Clustering by Subspace Mining
The idea behind the algorithm is quiet simple.
It improves upon fastDOC. 
It computes the HyperCube dimensions for each point compared to a medioid, to produce a bunch of dimension patterns
Each of these patterns is then used to build a FP-tree, and from this a version of the FP_Grow algorithm is used to find the hypercube that should give the best score. 
Since this is in FastDOC we only compute the accual cluster once. 
THis algorithm have only the number of mediod iterations, and each of these (around 20) iterations will traverse the data to create the hybercubes for each point. 


An idea here can be to use apriori algorithm in stead of FP-trees
