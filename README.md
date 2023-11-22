# Alternating between Surrogate Model Construction and Search for Configurations of an Autonomous Delivery System
Autonomous robots are emerging as a solution to various challenges of last mile goods delivery, like reducing traffic congestion, pollution, and costs. The configuration of an autonomous delivery robots system requires balancing aspects like delivery rate, cost of robots' operation, and required monitoring efforts. Our industry partner Panasonic is employing a search-based approach to find the configurations of the system that optimise these three aspects for a given set of customers' orders. The approach uses a simulator to assess the different configurations in the fitness functions' computation. Due to the high cost of the simulation, the whole search-based approach is computationally expensive. A classic approach to speed up such approaches is to use surrogate models trained on example simulation data that allow to approximate the results of a simulated configuration with negligible computational cost. A risk when using such approaches is to underestimate the cost of building the surrogate model itself, that can exceed the computational gain obtained during the search, thus making the adoption of surrogate models detrimental. In this work, we propose an approach in which the surrogate model is not trained before the search; instead, the approach alternates between training the model on subsets of data of increasing size, and searching using these cheaper models until the search stagnates. Experiments over 144,000 settings of the search show that the proposed approach can significantly reduce the cost of searching for configurations, while having an acceptable impact on the quality of the configurations it finds.

## How to reproduce the results
1. Run the batch file **run_var_search.bat** in the folder **src** to get the results of the approach
2. Run **AnalysisSANER2024.ipynb** to trigger the reevaluation of the Pareto Fronts
3. Run **run_reeval_parallel.py** that uses the simulator to generate the reevaluated Pareto Fronts
4. Run **AnalysisSANER2024.ipynb** to generate results **re_eval_quality_indicators.csv** and **threshold_min_quality_indicators.csv**
5. Run **stop_criterion_each_run.py** in the folder **each_run_stop_criterion** to apply all the stopping criteria to on each run (total 30 runs) and each approach
6. Run **stats_compare.py** to do statistical comparison with Mann-Whitney and A12. For each pair of approaches *approach1* and *approach2* and *metric* (*cost* and *IGD*), the following results are produced:
```python 
f'A1_{approach1}_A2_{approach2}_{metric}.csv'
```
7. Run **plot_heatmap.py** to obtain the heatmaps in 
```python 
f'A1_{approach1}_A2_{approach2}_{metric}.csv'
```

## People
* Chin-Hsuan Sun
* Thomas Laurent https://laurenttho3.github.io/
* Paolo Arcaini http://group-mmm.org/~arcaini/
* Fuyuki Ishikawa http://research.nii.ac.jp/~f-ishikawa/en/