import subprocess
import random
import time
import logging as log
from typing import Tuple, List, Dict
from itertools import count
import multiprocessing
import pandas as pd
import math
import numpy as np
import datetime
from joblib import Parallel, delayed

# shawn modify
import torch
from simulator.bin.surrogate_model.Surrogate_Model import DNN, DNN3, DNN_utilization
import os


class Surrogate:
    NUM_DELIVERED = 'num_delivered'
    NUM_RISKS = 'num_risks'
    UTILIZATION_RATE = 'utilization_rate'
    DELIVERY_RATE = 'delivery_rate'
    # if true, the missing simulations are run at the end. Otherwise, they are run immediatly (but only considered if needed)
    reconcilation_at_the_end = False

    METRICS = [NUM_RISKS, NUM_DELIVERED, UTILIZATION_RATE, DELIVERY_RATE]

    def __init__(self, min_customer_requests_per_hour: int,
                 max_customer_requests_per_hour: int,
                 robot_loading_capacity: int, simulation_duration: int,
                 threads: int = None, number_of_simulations: int = 1, seed: int = 48979683312,
                 results_folder: str = 'default',
                 year: int = 2021, month: int = 1, day: int = 1, hour: int = 9, minutes: int = 0,
                 simulator: str = 'simulator',
                 surrogate: DNN3 = None):
        self.surrogate = surrogate
        assert 0 < min_customer_requests_per_hour <= max_customer_requests_per_hour, 'Upper bound should be more than equal than lower bound.'
        self.robot_loading_capacity = robot_loading_capacity
        self.hour = hour
        self.minutes = minutes
        self.start_time = datetime.datetime(
            year=year, month=month, day=day, hour=hour, minute=minutes)
        self.simulation_duration = simulation_duration
        self.results_folder = results_folder
        self.threads = threads if threads else multiprocessing.cpu_count()
        random.seed(seed)
        if number_of_simulations > 1:
            self.seeds = [seed] + [random.randint(10 ** 11, 10 ** 12 - 1)
                          for _ in range(1,number_of_simulations)]
        else:
            self.seeds = [seed]
        self.request_per_hour = list(map(math.ceil,
                                         np.linspace(min_customer_requests_per_hour,
                                                     max_customer_requests_per_hour,
                                                     number_of_simulations)))
        self.selected_simulations = set()
        self.sim_counter = count()
        self.memo = {}

        if simulator.find("special_surrogate") != -1:
            self.METRICS = [self.DELIVERY_RATE,
                            self.UTILIZATION_RATE, self.NUM_RISKS]
        elif simulator.find("general_surrogate") != -1:
            self.METRICS = [self.DELIVERY_RATE,
                            self.UTILIZATION_RATE, self.NUM_RISKS]
        elif simulator.find("special_utilization") != -1:
            self.METRICS = [self.DELIVERY_RATE,
                            self.UTILIZATION_RATE, self.NUM_RISKS]
        self.simulator = simulator

    def get_unique_id(self):
        return self.sim_counter.__next__()

    @staticmethod
    def format_utilization_time_period(utilization_time_period):
        if utilization_time_period:
            utilization_time_str = []
            for (start, end) in utilization_time_period:
                assert 0 <= start < end < 24, f"The start time should be earlier than end time [Given: ({start}, {end})]"
                utilization_time_str.append(
                    f'{datetime.time(hour=start).strftime("%H:%M")}-{datetime.time(hour=end).strftime("%H:%M")}')
            return ','.join(utilization_time_str)
        return ''

    def calculate_working_hours(self, num_robots, utilization_time_period):
        working_hours = []
        start_hours = []
        if utilization_time_period:
            for (start, end) in utilization_time_period:
                working_hours_of_robot = end - start
                assert working_hours_of_robot > 0, "The working hours must be non-zero values"
                working_hours.append(working_hours_of_robot)
                start_hours.append(start)

        while len(working_hours) < num_robots:
            working_hours.append(self.simulation_duration)
            start_hours.append((60*self.hour + self.minutes)/60)

        assert len(
            working_hours) == num_robots, "Each robot must have the number of hours that works."

        return working_hours, start_hours

    def update_simulation_duration(self, simulation_duration):
        self.simulation_duration = simulation_duration

    def get_end_time(self):
        return self.start_time + datetime.timedelta(hours=self.simulation_duration)

    def set_number_of_simulations(self, number_of_simulations):
        new_simulations = list(
            set(range(len(self.seeds))) - self.selected_simulations)
        random.shuffle(new_simulations)
        target_simulations = min(
            number_of_simulations - len(self.selected_simulations), len(self.seeds))
        self.selected_simulations = self.selected_simulations.union(
            new_simulations[:target_simulations])

    def get_current_simulations(self):
        return set(range(len(self.seeds))) if len(self.selected_simulations) == 0 else self.selected_simulations

    def get_all_simulations(self):
        return set(range(len(self.seeds)))

    def run(self, num_robots: int, robot_speed_kmh: float, utilization_time_period: List[Tuple[int, int]] = None):
        assert type(
            num_robots) == int and num_robots > 0, "The number of robots must be a positive integer."
        assert type(
            robot_speed_kmh) == float and robot_speed_kmh > 0.0, "The speed must be a positive value."
        start = time.time()
        formatted_utilization_time_period = self.format_utilization_time_period(
            utilization_time_period)
        working_hours_per_robot, start_hours_per_robot = self.calculate_working_hours(
            num_robots, utilization_time_period)

        if self.reconcilation_at_the_end:
            selected_simulations = self.get_current_simulations()
            number_of_simulations = len(selected_simulations)
            log.getLogger().info(
                f'Number of simulations for this generation: {number_of_simulations} ({self.selected_simulations})')
            simulations = [(num_robots,
                            robot_speed_kmh,
                            formatted_utilization_time_period,
                            sim_id) for sim_id in selected_simulations]

            metrics = []

            # Getting previous results from cache and checking which simulations are missing
            missing_simulations = []
            for (num_robots, robot_speed_kmh, utilization_time_period, sim_id) in simulations:
                # the sim_id determines the seed and the requests per hour
                key = (num_robots, robot_speed_kmh, utilization_time_period,
                       self.simulation_duration, sim_id)
                if key in self.memo:
                    metrics.append(self.memo[key])
                else:
                    missing_simulations.append((num_robots,
                                                robot_speed_kmh,
                                                utilization_time_period,
                                                working_hours_per_robot,
                                                start_hours_per_robot,
                                                sim_id,
                                                self.get_unique_id()))

            # Adding the missing data to the cache
            missing_metrics = Parallel(n_jobs=self.threads)(
                delayed(self.run_single_simulation)(*params) for params in missing_simulations)
            for metric in missing_metrics:
                key = (num_robots, robot_speed_kmh, utilization_time_period,
                       self.simulation_duration, metric['sim_id'])
                self.memo[key] = metric
                metrics.append(metric)

            metrics_values = [[m[label]
                               for label in self.METRICS] for m in metrics]
            labels = self.METRICS
            response = dict(zip(labels, np.apply_along_axis(
                np.mean, 0, np.array(metrics_values))))

            sequential_execution_time = 0.0
            response['simulations'] = number_of_simulations
            for metrics_single in metrics:
                i = metrics_single['sim_id']
                for k, v in metrics_single.items():
                    if k != 'sim_id':
                        response[f'{k}_{i}'] = v
                sequential_execution_time += metrics_single['execution_time']

            response['sequential_execution_time'] = sequential_execution_time
            response['execution_time'] = time.time() - start

            return response
        else:
            selected_simulations = self.get_current_simulations()
            all_simulations = self.get_all_simulations()
            # number_of_simulations = len(selected_simulations)
            number_of_selected_simulations = len(selected_simulations)
            number_of_simulations = len(all_simulations)
            # log.getLogger().info(f'Number of simulations for this generation: {number_of_simulations} ({self.selected_simulations})')
            log.getLogger().info(
                f'Number of simulations for this generation: {number_of_simulations} ({all_simulations})')
            simulations = [(num_robots,
                            robot_speed_kmh,
                            formatted_utilization_time_period,
                            sim_id) for sim_id in
                           # selected_simulations
                           all_simulations
                           ]

            metrics = []

            # Getting previous results from cache and checking which simulations are missing
            missing_simulations = []
            for (num_robots, robot_speed_kmh, utilization_time_period, sim_id) in simulations:
                # the sim_id determines the seed and the requests per hour
                key = (num_robots, robot_speed_kmh, utilization_time_period,
                       self.simulation_duration, sim_id)
                if key in self.memo:
                    print("found in memo")
                    metrics.append(self.memo[key])
                else:
                    missing_simulations.append((num_robots,
                                                robot_speed_kmh,
                                                utilization_time_period,
                                                working_hours_per_robot,
                                                start_hours_per_robot,
                                                sim_id,
                                                self.get_unique_id()))

            # Adding the missing data to the cache
            missing_metrics = Parallel(n_jobs=self.threads)(
                delayed(self.run_single_simulation)(*params) for params in missing_simulations)
            for metric in missing_metrics:
                key = (num_robots, robot_speed_kmh, utilization_time_period,
                       self.simulation_duration, metric['sim_id'])
                self.memo[key] = metric
                metrics.append(metric)

            # print("metrics")
            # print(metrics)
            selected_metrics = []
            for metric in metrics:
                if metric['sim_id'] in selected_simulations:
                    selected_metrics.append(metric)
            # print("selected_metrics")
            # print(selected_metrics)

            # metrics_values = [[m[label] for label in self.METRICS] for m in metrics]
            metrics_values = [[m[label] for label in self.METRICS]
                              for m in selected_metrics]
            labels = self.METRICS
            response = dict(zip(labels, np.apply_along_axis(
                np.mean, 0, np.array(metrics_values))))

            # print("response")
            # print(metrics_values)
            # print("selected_simulations")
            # print(selected_simulations)
            # print("all_simulations")
            # print(all_simulations)

            sequential_execution_time = 0.0
            # response['simulations'] = number_of_simulations
            response['simulations'] = number_of_selected_simulations
            # for metrics_single in metrics:
            for metrics_single in selected_metrics:
                i = metrics_single['sim_id']
                for k, v in metrics_single.items():
                    if k != 'sim_id':
                        response[f'{k}_{i}'] = v
                sequential_execution_time += metrics_single['execution_time']

            response['sequential_execution_time'] = sequential_execution_time
            response['execution_time'] = time.time() - start

            return response

    def run_no_robots(self, robot_speed_kmh: float):
        assert type(
            robot_speed_kmh) == float and robot_speed_kmh > 0.0, "The speed must be a positive value."
        selected_simulations = self.get_current_simulations()
        number_of_simulations = len(selected_simulations)
        log.getLogger().info(
            f'Number of simulations for this generation: {number_of_simulations} ({self.selected_simulations})')

        labels = self.METRICS
        
        if self.simulator == 'simulator':
            response = dict(zip(labels, [100000, -1, -1, -1]))
        elif self.simulator.find("special_surrogate") != -1:
            response = dict(zip(labels, [-1, -1, 100000]))
        elif self.simulator.find("general_surrogate") != -1:
            response = dict(zip(labels, [-1, -1, 100000]))
        elif self.simulator.find('special_utilization') != -1:
            response = dict(zip(labels, [-1, -1, 100000]))

        response['simulations'] = number_of_simulations
        response['sequential_execution_time'] = 0
        response['execution_time'] = 0

        return response

    def run_single_simulation(self, num_robots, robot_speed_kmh, utilization_time_period, working_hours_per_robot, start_hours_per_robot, sim_id, unique_id):
        try:
            start = time.time()
            requests_per_hour = self.request_per_hour[sim_id]
            seed = self.seeds[sim_id]
            if self.simulator == "simulator":
                command = (f'run/run --seed {seed} --sim_name {self.results_folder} --sim_id {unique_id}'
                           f' --start_time {self.start_time.isoformat()}'
                           f' --end_time {self.get_end_time().isoformat()}'
                           f' --num_customer_requests {requests_per_hour * self.simulation_duration}'
                           f' --num_robots {num_robots}'
                           f' --robot_speed_kmph {robot_speed_kmh}'
                           f' --robot_loading_capacity {self.robot_loading_capacity}')

                if utilization_time_period:
                    command += f' --utilization_time_period {utilization_time_period}'
            elif self.simulator.find("special_surrogate") != -1:
                command = np.array(
                    [num_robots, robot_speed_kmh], dtype=np.float32)
                command = torch.Tensor(command)
            elif self.simulator.find("general_surrogate") != -1:
                command = np.array([requests_per_hour, num_robots,
                                    robot_speed_kmh, self.robot_loading_capacity], dtype=np.float32)
                command = torch.Tensor(command)
            elif self.simulator.find('special_utilization') != -1:
                working_hours_per_robot_DNN = [0 for i in range(10)]
                start_hours_per_robot_DNN = [0 for i in range(10)]
                for i in range(len(working_hours_per_robot)):
                    start_hours_per_robot_DNN[i] = start_hours_per_robot[i]
                    working_hours_per_robot_DNN[i] = working_hours_per_robot[i]
                    
                command_in = start_hours_per_robot_DNN + working_hours_per_robot_DNN + [robot_speed_kmh]
                command = np.array(command_in, dtype=np.float32)
                command = torch.Tensor(command)
            else:
                raise Exception(
                    "Something wrong in your simulator name, should be simulator, special_surrogate or general_surrogate !")

            result = self.execute_command(
                command, requests_per_hour, unique_id, working_hours_per_robot)
            result['sim_id'] = sim_id
            result['unique_id'] = unique_id
            result['seed'] = seed
            result['requests_per_hour'] = requests_per_hour
            result['execution_time'] = time.time() - start

            return result

        except subprocess.CalledProcessError as e:
            log.getLogger().error(e, e.output, e.stdout, e.stderr)
            raise RuntimeError(
                "The simulator is not available or does not have execution permission.")

    def execute_command(self, command: str, requests_per_hour: int, sim_id: int, working_hours_per_robot: List[int]) -> \
            Dict[str, float]:
        if self.simulator == "simulator":
            subprocess.run(command, shell=True, check=True,
                           capture_output=True, cwd="simulator/bin")
            try:
                df_cost = pd.read_csv(
                    f'simulator/bin/result/{self.results_folder}/{sim_id}/cost.csv')
                df_risk = pd.read_csv(
                    f'simulator/bin/result/{self.results_folder}/{sim_id}/risk.csv')
            except:
                print("Error reading files cost.csv or risk.csv from folder "
                      + f'simulator/bin/result/{self.results_folder}/{sim_id}')
                return self.worst_response()

            num_delivered = df_cost['num_delivered']
            utilization_rate = df_cost['utilization_rate']
            num_risks = df_risk.index.size

            return self.create_response(num_delivered, num_risks, requests_per_hour, utilization_rate,
                                        working_hours_per_robot)
        elif self.simulator.find("special_surrogate") != -1:
            output = self.surrogate(command)
            output = output.detach().cpu().numpy()
            output[output < 0] = 0

            return {Surrogate.UTILIZATION_RATE: output[1],
                    Surrogate.DELIVERY_RATE: output[0],
                    Surrogate.NUM_RISKS: output[2]}
        elif self.simulator.find("general_surrogate") != -1:
            output = self.surrogate(command)
            output = output.detach().cpu().numpy()
            output[output < 0] = 0

            return {Surrogate.UTILIZATION_RATE: output[1],
                    Surrogate.DELIVERY_RATE: output[0],
                    Surrogate.NUM_RISKS: output[2]}
            
        elif self.simulator.find("special_utilization") != -1:
            output = self.surrogate(command)
            output = output.detach().cpu().numpy()
            output[output < 0] = 0

            return {Surrogate.UTILIZATION_RATE: output[1],
                    Surrogate.DELIVERY_RATE: output[0],
                    Surrogate.NUM_RISKS: output[2]}
            
            # return {Surrogate.UTILIZATION_RATE: 1,
            #         Surrogate.DELIVERY_RATE: 1,
            #         Surrogate.NUM_RISKS: 0}
            
    def create_response(self, num_delivered, num_risks, requests_per_hour, utilization_rate, working_hours_per_robot):
        working_percentage = pd.Series(
            working_hours_per_robot) / self.simulation_duration
        adjusted_utilization_rate = utilization_rate / working_percentage

        return {Surrogate.NUM_DELIVERED: num_delivered.sum() / self.simulation_duration,
                Surrogate.UTILIZATION_RATE: adjusted_utilization_rate.mean(),
                Surrogate.DELIVERY_RATE: num_delivered.sum() / (requests_per_hour * self.simulation_duration),
                Surrogate.NUM_RISKS: num_risks / self.simulation_duration}

    @staticmethod
    def worst_response():
        return {Surrogate.NUM_DELIVERED: 0,
                Surrogate.UTILIZATION_RATE: 0.0,
                Surrogate.DELIVERY_RATE: 0.0,
                Surrogate.NUM_RISKS: 99999999999}


class MockSimulator(Surrogate):

    def execute_command(self, command, requests_per_hour: int, sim_id: int, working_hours_per_robot: List[int]):
        log.getLogger().debug(command)
        log.getLogger().debug(
            f'simulator/bin/result/{self.results_folder}/{sim_id}/cost.csv')

        utilization_rate = pd.Series([random.random(
        ) * (wh / self.simulation_duration) for wh in working_hours_per_robot])
        num_delivered = pd.Series([random.randint(10, 100)
                                  for _ in working_hours_per_robot])
        num_risks = random.randint(0, 100)

        return self.create_response(num_delivered, num_risks, requests_per_hour, utilization_rate,
                                    working_hours_per_robot)
