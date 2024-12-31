
# Dynamic Contagion Potential Framework for Optimizing Infection Control in Healthcare

## Abstract

<p align="justify">Healthcare-acquired infections (HAIs) caused by bacterial and viral pathogens continue to affect millions of individuals annually, posing a significant challenge to healthcare systems. Traditional infection control strategies often fall short due to their inability to assess real-time spatial and movement data within healthcare facilities dynamically. To address this gap, this study leverages *contagion potential* (CP), a metric that quantifies infection risk based on individual characteristics and behavior, to propose a framework for minimizing the incidence of HAIs. CP accounts for the infection susceptibility and transmissibility of individuals, incorporating their movement patterns and interactions across various units within a healthcare facility. The proposed framework integrates approximate location data, modeling the infection risk landscape without requiring precise tracking. Through continuous learning, the CP parameters are inferred and refined over time, enabling accurate infection risk assessments at both the individual and unit levels. This framework also includes an optimization approach for patient-to-unit assignments, using CP to minimize contagion risk while ensuring that patients receive the appropriate clinical care. The efficacy of the framework is validated through experiments, both on individual modules and in an integrated evaluation, using synthetic and real-world datasets. The results demonstrate that leveraging CP significantly enhances infection control efforts, optimizes healthcare resource allocation, and improves patient safety. Overall, this dynamic, data-driven approach offers a robust strategy to combat HAIs, contributing to improved healthcare environments and patient outcomes.</p>

---

## Experiments and Corresponding Figures

### Predicting_Neighbors_4a
<p align="justify">Compares the predicted number of neighbors of an individual with the true number of neighbors based on homogeneous mixing. This experiment corresponds to ** Figure 4a **.</p>

### PageRank_Population_4b
<p align="justify">Predicts mobility of individuals using PageRank to form the transition matrix between grids. The true population of a grid is compared with the predicted population of a grid. This experiment corresponds to **Figure 4b**.</p>

### Predict_CP_True_Mobility_5a
<p align="justify">Compares true and predicted CP and infectivity based on accurate mobility data between grids. This experiment corresponds to **Figure 5a**.</p>

### Predict_CP_Predict_Mobility_5b
<p align="justify">Compares true and predicted CP and infectivity based on predicted mobility data between grids. The mobility between grids is predicted using the PageRank process. This experiment corresponds to **Figure 5b**.</p>

### Learn_Infection_Parameters_6
<p align="justify">Learns individual infection parameters based on sampling individuals for testing. This experiment corresponds to **Figure 6**.</p>

### Optimized_Assignments_7
<p align="justify">Compares the impact of greedy and optimized assignments on mean CP and infection spread under one wave of infection. Heterogeneous infection parameters are learned over time. This experiment corresponds to **Figure 7**.</p>

### Optimized_Assignments_Two_Waves_8 and 9
<p align="justify">Compares the impact of greedy and optimized assignments on infection spread under two waves of infection. Heterogeneous infection parameters are learned over time. This experiment corresponds to **Figures 8 and 9**.</p>

### Periodic_Optimization_10
<p align="justify">Compares the impact of greedy and optimized assignments on infection spread under two waves of infection with periodic optimized assignments. Heterogeneous infection parameters are also learned periodically. This experiment corresponds to **Figure 10**.</p>

### Gravity_Model_11
<p align="justify">The model is carried out with a gravity mobility model. Compares the impact of greedy and optimized assignments on infection spread under two waves of infection with periodic optimized assignments. Heterogeneous infection parameters are also learned periodically. This experiment corresponds to **Figure 11**.</p>
