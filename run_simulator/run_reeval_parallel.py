# %%
import os
import pandas as pd
import numpy as np
import shutil
import json
import pickle
import datetime
import random
from joblib import Parallel, delayed
import tqdm
from glob import glob

def run_one_sim(seed, id, num_customer_requests, num_robots, robot_speed_kmph, robot_loading_capacity, utilization_time_period, sim_name="ref"):
    command = "run/run "
    command += f"--seed {seed} "
    command += f"--sim_name {sim_name} "
    command += f"--sim_id {id} "
    command += "--start_time 2021-01-01T09:00:00 "
    command += "--end_time 2021-01-01T14:00:00 "
    command += f"--num_customer_requests {num_customer_requests} "
    command += f"--num_robots {num_robots} "
    command += f"--robot_speed_kmph {robot_speed_kmph} "
    command += f"--robot_loading_capacity {robot_loading_capacity} "
    command += f"--utilization_time_period {utilization_time_period} "
    command += f">> sim_id{id}_log.txt"
    print(f'The usage of command is : {command} \n\n')

    currpath = os.getcwd()
    print(f'current work directory : {currpath}')
    os.chdir('simulator')
    os.system(command)
    print('-------------------finish--------------------')

    with open(f"sim_id{id}_log.txt", "a") as f:
        f.write(command)
    oldpath = os.path.join(currpath, 'simulator', f"sim_id{id}_log.txt")
    newpath = os.path.join(currpath, 'simulator', "result", sim_name, f"{id}")
    command = f'mv {oldpath} {newpath}'
    os.system(command)
    # shutil.move(oldpath, newpath)

    config = {'seed': seed,
              'sim_name': sim_name,
              'num_customer_requests': num_customer_requests,
              'num_robots': num_robots,
              'robot_speed_kmph': robot_speed_kmph,
              'robot_loading_capacity': robot_loading_capacity,
              'utilization_time_period' : utilization_time_period,
              'sim_id': id}
    with open(os.path.join("result", sim_name, f"{id}", "config.json"), "w") as f:
        json.dump(config, f)
    os.chdir(currpath)

def format_utilization_time_period(utilization_time_period):
    if utilization_time_period:
        utilization_time_str = []
        for (start, end) in utilization_time_period:
            assert 0 <= start < end < 24, f"The start time should be earlier than end time [Given: ({start}, {end})]"
            utilization_time_str.append(f'{datetime.time(hour=start).strftime("%H:%M")}-{datetime.time(hour=end).strftime("%H:%M")}')
        return ','.join(utilization_time_str)
    return ''

def simulations(seed, reeval_row, row_idx):
    utilization_time_period = []
    for i in range(10):
        robot_start = int(reeval_row[f'robot_{i}_start'])
        robot_end = int(reeval_row[f'robot_{i}_end'])
        if robot_start==0:
            break
        utilization_time_period.append((robot_start, robot_end))
    num_customer_requests = 250
    num_robots = len(utilization_time_period)
    robot_speed_kmph = reeval_row['speed_kmh']
    robot_loading_capacity = 5
    utilization_time_period = format_utilization_time_period(utilization_time_period)
    run_one_sim(seed, row_idx+1, num_customer_requests, num_robots,
                            robot_speed_kmph, robot_loading_capacity, utilization_time_period, sim_name)

def gen_dataset(seed, sim_name, reeval_file):
    # %% generate dataset
    # seed = 59713643048
    # start = 9
    # end = 14
    reeval_file = pd.read_csv(reeval_file)
    Parallel(n_jobs=-5)(delayed(simulations)(seed, reeval_row, row_idx) for row_idx, reeval_row in tqdm.tqdm(reeval_file.iterrows()))

def check_lost_file(datapath):
    lost_file = []
    datapath = os.path.join("simulator", "result", datapath)
    folders = glob(os.path.join(datapath, '*'))
    for id in range(1,len(folders)+1):
        if (not os.path.isfile(os.path.join(datapath,str(id),'cost.csv'))) | (not os.path.isfile(os.path.join(datapath,str(id),'risk.csv'))):
            lost_file.append(id)
            print(f'lost reeval folder {id}')   
    
    return lost_file  

def regenerate_lost_file(seed, lost_file, reeval_file):
    reeval_file = pd.read_csv(reeval_file)
    # regenerate lost file 
    for file_id in lost_file:
        reeval_row = reeval_file.iloc[file_id-1]
        simulations(seed, reeval_row, file_id-1)
       

if __name__ == "__main__":
    seed = 59713643048
    
    sim_name = 'dataset30_reeval_parallel_incremental_data'
    reeval_file = 'need_reeval.csv'
    
    gen_dataset(seed, sim_name, reeval_file)
    
    lost_file = ['init']
    while len(lost_file) != 0:
        lost_file = check_lost_file(sim_name)
        print(f'lost files : {lost_file}')
        regenerate_lost_file(seed, lost_file, reeval_file)
    
    print('All is done!')
