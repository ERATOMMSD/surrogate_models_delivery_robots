from joblib import Parallel, delayed
import pandas as pd
import os
import numpy as np
from tqdm import tqdm

look_up_table = pd.read_csv("re_eval_quality_indicators.csv")
get_best_individual = pd.read_csv("min_threshold.csv")

def from_df_to_numpy(df):
    delivery_rate_worst = df['delivery_rate_best'].to_list()
    utilization_rate_worst = df['utilization_rate_best'].to_list()
    num_risk_worst = df['num_risk_best'].to_list()

    return np.array([delivery_rate_worst, utilization_rate_worst, num_risk_worst])

def apply_stop_criterion(H, window, delivery_rate_threshold, utilization_rate_threshold, num_risk_threshold):
    row = {}
    for idx, generation in enumerate(generations):
        if generation == 'standard':
            for run_idx in range(1,31):
                for standard_generation in range(window+1,21):
                    based_approach = f'standard_H_0_M_{standard_generation - window}' 
                    now_approach = f'standard_H_0_M_{standard_generation}'
                
                
                    based_data = get_best_individual[(get_best_individual['approach'] == based_approach) & (get_best_individual['run'] == run_idx)]
                    now_data = get_best_individual[(get_best_individual['approach'] == now_approach) & (get_best_individual['run'] == run_idx)]
                    
                    based_obj = from_df_to_numpy(based_data) + 1e-9
                    now_obj = from_df_to_numpy(now_data) + 1e-9       
                    assert np.where(now_obj <= based_obj, 1, 0).all()
                    
                    change_obj = (now_obj - based_obj)/np.abs(based_obj)*100
                    if (change_obj[0] > delivery_rate_threshold) & (change_obj[1] > utilization_rate_threshold) & (change_obj[2] > num_risk_threshold):
                        row[f'standard_run_{run_idx}_Stop_Generation'] = standard_generation
                        row[f'standard_run_{run_idx}_reeval_IGD'] = look_up_table[(look_up_table['approach'] == now_approach) & (look_up_table['run'] == run_idx)]['IGD'].item()
                        row[f'standard_run_{run_idx}_budget'] = standard_generation*12 + 12
                        break
                    else:
                        row[f'standard_run_{run_idx}_Stop_Generation'] = standard_generation
                        row[f'standard_run_{run_idx}_reeval_IGD'] = look_up_table[(look_up_table['approach'] == now_approach) & (look_up_table['run'] == run_idx)]['IGD'].item()
                        row[f'standard_run_{run_idx}_budget'] = standard_generation*12 + 12
        else:
            M = [i for i in range(window+1, int(generation))]
            using_H = H[idx-1]
            for run_idx in range(1, 31):
                for using_M in M:
                    based_approach = f'incremental-data_lab28_special_utilization_model12.pth_H_{using_H}_M_{using_M - window}' 
                    now_approach = f'incremental-data_lab28_special_utilization_model12.pth_H_{using_H}_M_{using_M}'

                
                    based_data = get_best_individual[(get_best_individual['approach'] == based_approach) & (get_best_individual['run'] == run_idx)]
                    now_data = get_best_individual[(get_best_individual['approach'] == now_approach) & (get_best_individual['run'] == run_idx)]
                    
                    based_obj = from_df_to_numpy(based_data) + 1e-9
                    now_obj = from_df_to_numpy(now_data) + 1e-9       
                    assert np.where(now_obj <= based_obj, 1, 0).all()
                    
                    change_obj = (now_obj - based_obj)/np.abs(based_obj)*100
                    if (change_obj[0] > delivery_rate_threshold) & (change_obj[1] > utilization_rate_threshold) & (change_obj[2] > num_risk_threshold):
                        row[f'H_{using_H}_run_{run_idx}_Stop_Generation'] = using_M
                        row[f'H_{using_H}_run_{run_idx}_reeval_IGD'] = look_up_table[(look_up_table['approach'] == now_approach) & (look_up_table['run'] == run_idx)]['IGD'].item()
                        if using_M < using_H*2:
                            row[f'H_{using_H}_run_{run_idx}_budget'] = 32
                        elif (using_M >= using_H*2) & (using_M < using_H*5):
                            row[f'H_{using_H}_run_{run_idx}_budget'] = 63
                        elif (using_M >= using_H*5) & (using_M < using_H*10):
                            row[f'H_{using_H}_run_{run_idx}_budget'] = 126
                        elif (using_M >= using_H*10) & (using_M < using_H*21):
                            row[f'H_{using_H}_run_{run_idx}_budget'] = 252
                        break
                    else:
                        row[f'H_{using_H}_run_{run_idx}_Stop_Generation'] = using_M
                        row[f'H_{using_H}_run_{run_idx}_reeval_IGD'] = look_up_table[(look_up_table['approach'] == now_approach) & (look_up_table['run'] == run_idx)]['IGD'].item()
                        if using_M < using_H*2:
                            row[f'H_{using_H}_run_{run_idx}_budget'] = 32
                        elif (using_M >= using_H*2) & (using_M < using_H*5):
                            row[f'H_{using_H}_run_{run_idx}_budget'] = 63
                        elif (using_M >= using_H*5) & (using_M < using_H*10):
                            row[f'H_{using_H}_run_{run_idx}_budget'] = 126
                        elif (using_M >= using_H*10) & (using_M < using_H*21):
                            row[f'H_{using_H}_run_{run_idx}_budget'] = 252
                    
    row['delivery_rate_threshold'] = delivery_rate_threshold
    row['utilization_rate_threshold'] = utilization_rate_threshold
    row['num_risk_threshold'] = num_risk_threshold
    row['window'] = window
    
    return row

if __name__ == "__main__":
    delivery_rate_threshold_set = np.linspace(-10, -0.1, 30) # percentage
    utilization_rate_threshold_set = np.linspace(-10, -0.1, 30) # percentage
    num_risk_threshold_set = np.linspace(-50, -10, 20) # percentage
    window_set = [10, 9, 8, 7, 6, 5, 4, 3]
    generations = ['standard', '21', '42', '63', '84']
    H = [1,2,3,4]

    products = []
    for window in window_set:
        for delivery_rate_threshold in delivery_rate_threshold_set:
            for utilization_rate_threshold in utilization_rate_threshold_set:
                for num_risk_threshold in num_risk_threshold_set:
                    products.append((window, delivery_rate_threshold, utilization_rate_threshold, num_risk_threshold))


    data = Parallel(n_jobs=-5)(delayed(apply_stop_criterion)(H, window, delivery_rate_threshold, utilization_rate_threshold, num_risk_threshold) for window, delivery_rate_threshold, utilization_rate_threshold, num_risk_threshold in tqdm(products))

    data = pd.DataFrame(data)
    data.to_csv("stop_criterion.csv", index=False)