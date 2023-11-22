from torch.utils.data import Dataset
import os
import json
import csv
import numpy as np
import pandas as pd


def GetSimResult(id: str, DATAPATH: str):
    # extract x
    num_customer_requests = 0
    num_robots = 0
    robot_speed_kmph = 0
    robot_loading_capacity = 0
    with open(os.path.join(DATAPATH, id, f"sim_id{id}_log.txt"), 'r') as f:
        file = f.read()
    start_time = file.split("--start_time")
    start_time = start_time[-1]
    start_time = start_time.split("2021-01-01T")[1]
    start_hour = int(start_time.split(":")[0])
    start_min = int(start_time.split(":")[1])

    end_time = file.split("--end_time")
    end_time = end_time[-1]
    end_time = end_time.split("2021-01-01T")[1]
    end_hour = int(end_time.split(":")[0])
    end_min = int(end_time.split(":")[1])

    duration = (end_hour*60 + end_min) - (start_hour*60 + start_min)
    duration = duration//60
    
    utilization_time_period = file.split('--utilization_time_period ')[-1]
    utilization_time_period = utilization_time_period.split()[0]
    utilization_time_period = utilization_time_period.split(',')

    utilization_time_period_deformat = [0 for i in range(10)]
    utilization_time_period_start = [0 for i in range(10)]
    for i in range(len(utilization_time_period)):
        t1 = utilization_time_period[i].split('-')[0]
        t1_hour = int(t1.split(':')[0])
        t1_min = int(t1.split(':')[1])
        t2 = utilization_time_period[i].split('-')[1]
        t2_hour = int(t2.split(':')[0])
        t2_min = int(t2.split(':')[1])
        utilization_time_period_deformat[i] = ((t2_hour*60 + t2_min) - (t1_hour*60 + t1_min))/60
        utilization_time_period_start[i] = (t1_hour*60 + t1_min)/60

    with open(os.path.join(DATAPATH, id, "config.json"), "r") as f:
        sim_config = json.load(f)
        num_customer_requests = sim_config["num_customer_requests"]//duration
        num_robots = sim_config["num_robots"]
        robot_speed_kmph = sim_config["robot_speed_kmph"]
        robot_loading_capacity = sim_config["robot_loading_capacity"]

    # for simplify problem
    simulation_duration = 5
    working_hours_per_robot = utilization_time_period_deformat
    working_percentage = pd.Series(
        working_hours_per_robot) / simulation_duration

    df_cost = pd.read_csv(os.path.join(DATAPATH, id, "cost.csv"))
    df_risk = pd.read_csv(os.path.join(DATAPATH, id, "risk.csv"))
    num_delivered = df_cost['num_delivered']
    utilization_rate = df_cost['utilization_rate']
    num_customer_requests
    num_delivered = df_cost['num_delivered'].sum()
    num_risks = df_risk.index.size
    NUM_DELIVERED = num_delivered/simulation_duration
    UTILIZATION_RATE = (utilization_rate / working_percentage).mean()
    DELIVERY_RATE = num_delivered / \
        (num_customer_requests * simulation_duration)
    NUM_RISKS = (num_risks / simulation_duration)

    return num_customer_requests, utilization_time_period_start, utilization_time_period_deformat, robot_speed_kmph, robot_loading_capacity, NUM_DELIVERED, DELIVERY_RATE, UTILIZATION_RATE, NUM_RISKS


class myDataset(Dataset):
    def __init__(self, ds, size_x) -> None:
        super().__init__()
        self.x = ds[:, :size_x]
        self.y = ds[:, size_x:]

        self.len = len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len
