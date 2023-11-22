import pandas as pd
import os
import numpy as np
from a12 import a12
from scipy.stats import mannwhitneyu
from itertools import product
from joblib import Parallel, delayed
from tqdm import tqdm

def judge_stats_level(a12_stats: float, utest_pvalue: float) -> dict:
    judge_results = {'SAME': 0, 'worseNegl': 0, 'worseSmall': 0, 'worseMedium': 0, 'worseLarge': 0,
                     'betterNegl': 0, 'betterSmall': 0, 'betterMedium': 0, 'betterLarge': 0}
    if utest_pvalue > 0.05:
        judge_results['SAME'] = 1
        return judge_results
    else:
        if (a12_stats > 0.5) & (a12_stats < 0.556):
            judge_results['worseNegl'] = 1
            return judge_results
        elif (a12_stats >= 0.556) & (a12_stats < 0.638):
            judge_results['worseSmall'] = 1
            return judge_results
        elif (a12_stats >= 0.638) & (a12_stats < 0.714):
            judge_results['worseMedium'] = 1
            return judge_results
        elif (a12_stats >= 0.714):
            judge_results['worseLarge'] = 1
            return judge_results
        elif (a12_stats > 0.494) & (a12_stats < 0.5):
            judge_results['betterNegl'] = 1
            return judge_results
        elif (a12_stats > 0.362) & (a12_stats <= 0.494):
            judge_results['betterSmall'] = 1
            return judge_results
        elif (a12_stats > 0.286) & (a12_stats <= 0.362):
            judge_results['betterMedium'] = 1
            return judge_results
        elif (a12_stats <= 0.286):
            judge_results['betterLarge'] = 1
            return judge_results
    
def stats_compare(stop_criterion, indicator, approach_1, approach_2, stop_idx):
    C1 = []
    C2 = []
    for run_idx in range(1,31):
        C1.append(stop_criterion[f'{approach_1}_run_{run_idx}_{indicator}'].iloc[stop_idx])
        C2.append(stop_criterion[f'{approach_2}_run_{run_idx}_{indicator}'].iloc[stop_idx])
        
    a12_stats = a12(C1, C2)
    utest_stats, utest_pvalue = mannwhitneyu(C1, C2)
    judge_results = judge_stats_level(a12_stats=a12_stats,
                                    utest_pvalue=utest_pvalue)
    
    C1 = np.array(C1)
    C2 = np.array(C2)
    if (judge_results['betterMedium']==1) or (judge_results['betterLarge']==1):
        diff_cost = np.mean(C1 - C2)
    else:
        diff_cost = 0
    
    row = {'diff_cost': diff_cost, 'a12_stats': a12_stats, 'utest_stats': utest_stats, 'utest_pvalue': utest_pvalue, **judge_results}
    return row

    

if __name__ == "__main__":
    stop_criterion = pd.read_csv("stop_criterion.csv")
    
    
    approaches = ['H_1', 'H_2', 'H_3', 'H_4', 'standard']
    indicators = ['reeval_IGD','budget']
    
    for indicator in indicators:
        for approach_1 in approaches:
            for approach_2 in approaches:
                if approach_1 != approach_2:
                    statistical_comparison_data = Parallel(n_jobs=-5)(delayed(stats_compare)(stop_criterion, indicator, approach_1, approach_2, stop_idx) for stop_idx in tqdm(range(144000)))
                    stats_df = pd.DataFrame(statistical_comparison_data)
                    stats_df.to_csv(f"A1_{approach_1}_A2_{approach_2}_{indicator}.csv", index=False)
                    

