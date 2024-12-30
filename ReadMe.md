There are 9 main scripts:

Predicting_Neighbors_4a: Compares the predicted number of neighbors of an individual with the true number of neighbors based on homogeneous mixing. This experiment corresponds to figure 4a. 

PageRank_Population_4b: Predicts mobility of individuals using PageRank to form the transition matrix between grids. The true population of a grid is compared with the predicted population of a grid. This experiment corresponds to figure 4b.

Predict_CP_True_Mobility_5a: Compares true and predicted CP and infectivity based on accurate mobility data between grids. This experiment corresponds to figure 5a.

Predict_CP_Predict_Mobility_5b: Compares true and predicted CP and infectivity based on predicted mobility data between grids. The mobility between grids is predicted using the PageRank process. This experiment corresponds to figure 5b.

Learn_Infection_Parameters_6: Learns individual infeciton parameters based on sampling individuals for testing. This experiment corresponds to figure 6.

Optimized_Assignments_7: Compares the impact of greedy and optimized assignments on mean CP and infection spread under one wave of infection. Heterogeneous infection parameters are learned over time. This experiment corresponds to figure 7.

Optimized_Assigments_Two_Waves_8 and 9: Compares the impact of greedy and optimized assignments on infection spread under two waves of infection. Heterogeneous infection parameters are learned over time. This experiment corresponds to figures 8 and 9.

Periodic_Optimization_10: Compares the impact of greedy and optimized assignments on infection spread under two waves of infection with periodic optmized assignments. Heterogeneous infection parameters are also learned periodically. This experiment corresponds to figures 10.

Gravity_Model_11: Model is carried out with a gravity mobility model. Compares the impact of greedy and optimized assignments on infection spread under two waves of infection with periodic optmized assignments. Heterogeneous infection parameters are also learned periodically. This experiment corresponds to figures 11.
