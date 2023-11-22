#!/usr/bin/env python
# coding: utf-8
import logging as log
import os
import time

import jmetal.config
from jmetal.util.evaluator import MapEvaluator

from archive.deap_problem import DeapDeliverySearchProblem
from deliverysearch.problem import RobotQuantityProblem, RobotScheduleAssignmentProblem, WeightedProblem, \
    RobotScheduleAssignmentProblemVariableRobots, RobotQuantitySimpleProblem, RobotScheduleVariableRobotsIncrementalEpochs, RobotScheduleVariableRobotsBeforeAfter, RobotScheduleVariableRobotsIncrementalData
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator.crossover import IntegerSBXCrossover
from jmetal.operator.mutation import IntegerPolynomialMutation
from jmetal.util.termination_criterion import StoppingByEvaluations
from deliverysearch.job import DeliveryJob

from simulator.simulator import Simulator, MockSimulator
from simulator.surrogate import Surrogate

from optparse import OptionParser

from deliverysearch.wrappers import IncrementalSimulationsWrapper, IncrementalTimeWrapper, AlgorithmWrapper, BeforeAfterWrapper, IncrementalEpochWrapper, IncrementalDatahWrapper

from simulator.bin.surrogate_model.Surrogate_Model import DNN, DNN3, DNN_utilization
import torch

if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option('--tool', action="store", dest="tool", default='jmetalpy',
                      help="Algorithm used for the search (underlying tool jmetalpy or deap)")
    parser.add_option('--mock', action="store_true", dest="mock", default=False,
                      help="it uses the mock simulator for testing purposes")
    parser.add_option('--generations', action="store", type="int", dest="generations", default=5,
                      help="number of generations that the evolutionary algorithm")
    parser.add_option('--population', action="store", type="int", dest="population", default=12,
                      help="population size per generation of the evolutionary algorithm (multiples of 4?)")
    parser.add_option('--offspring-population', action="store", type="int", dest="offspring_population_size",
                      default=8,
                      help="population size of the offspring of the evolutionary algorithm (jmetalpy)")
    parser.add_option('--crossover-prob', action="store", type="float", dest="crossover_prob", default=0.9,
                      help="crossover probability")
    parser.add_option('--low-bound', action="store", type="int", dest="low_bound", default=1,
                      help="lower bound for the search (both robots and speed)")
    parser.add_option('--speed', action="store", type="int", dest="speed", default=10,
                      help="maximum speed")
    parser.add_option('--speed-factor', action="store", type="int", dest="speed_factor", default=1,
                      help="changes the scale of the speed to increase granularity")
    parser.add_option('--max-requests', action="store", type="int", dest="max_requests", default=100,
                      help="minimum number of requests per hour to be received")
    parser.add_option('--min-requests', action="store", type="int", dest="min_requests", default=20,
                      help="maximum number of requests per hour to be received")
    parser.add_option('--capacity', action="store", type="int", dest="capacity", default=5,
                      help="number of packages that a robot can load")
    parser.add_option('--duration', action="store", type="int", dest="duration", default=3,
                      help="duration of the simulation")
    parser.add_option('--start-time', action="store", type="int", dest="start_time", default=9,
                      help="time when the simulation starts in hours (e.g, 9 means 9:00)")
    parser.add_option('--threads', action="store", type="int", dest="threads", default=None,
                      help="number of threads to be used (by default it uses the total number of cores")
    parser.add_option('--run-tag', action="store", type="int", dest="run_tag", default=0,
                      help="number of runs of the experiment")
    parser.add_option('--seed', action="store", type="int", dest="seed", default=59713643048,
                      help="seed that is used to initialize both the simulator and the search")
    parser.add_option('--jmetal-seed', action="store", type="int", dest="jmetal_seed", default=73912398701,
                      help="seed that is used to initialize both the simulator and the search")
    parser.add_option('--problem', action="store", type="str", dest="problem", default="robot-quantity",
                      help="It can be 'robot-quantity' or 'schedule-assignment'")
    parser.add_option('--robots', action="store", type="int", dest="robots", default=10,
                      help="maximum number of robots")
    parser.add_option('--approach', action="store", type="str", dest="approach", default="standard",
                      help="Simulator's precision strategy (standard, incremental-time, incremental-simulations).")
    parser.add_option('--simulations', action="store", type="int", dest="simulations", default=1,
                      help="number of samples in which an individual is evaluated")
    parser.add_option('--results-dir', action="store", type="str", dest="result_dir", default="results",
                      help="name of folder in which results are stored")
    parser.add_option('--weighted', action="store_true", dest="weighted", default=False,
                      help="If weighted, it uses a single objective search.")
    parser.add_option('--parallel', action="store", type="int", dest="parallel", default=1,
                      help="number of parallel threads")
    parser.add_option('--simulator', action="store", type="str", dest="simulator", default="simulator",
                      help="choosed simulator you want to use. e.g. surrogate.pth or simulator")
    parser.add_option('--switch', action="store", type="int", dest="switch", default=0,
                      help="change simulator from surrogate to Panasonic simulator")
    parser.add_option('--surrogate', action="store", type="str", dest="surrogate", default=None,
                    help="choosed surrogate you want to use. e.g. surrogate.pth or simulator")
    parser.add_option('--surrogate_type', action="store", type="str", dest="surrogate_type", default=None,
                    help="choosed surrogate type you want to use. e.g. surrogate100, surrogate50, surrogate25 or surrogate12")
    parser.add_option('--epoch', action="store", type="int", dest="epoch", default=None,
                      help="change epoch for each M generation you want to train")
    parser.add_option('--H', action="store", type="int", dest="H", default=None,
                      help="train each H*M generation")

    options, args = parser.parse_args()
    log.getLogger().setLevel(log.INFO)

    if options.mock:
        Sim = MockSimulator
        Sur = MockSimulator
    else:
        Sim = Simulator
        Sur = Surrogate
        

    # TODO: Lower and upper bounds should be independent
    if options.tool == 'deap':

        sim = Sim(min_customer_requests_per_hour=options.min_requests,
                  max_customer_requests_per_hour=options.max_requests,
                  robot_loading_capacity=options.capacity,
                  simulation_duration=options.duration,
                  hour=options.start_time,
                  number_of_simulations=options.simulations,
                  threads=options.threads,
                  seed=options.seed,
                  results_folder=f'{options.problem}')

        deap_optimization = DeapDeliverySearchProblem(sim=sim,
                                                      generations=options.generations,
                                                      population_size=options.population,
                                                      low_bound=options.low_bound,
                                                      upper_bound=options.speed,
                                                      dimensions=2,
                                                      crossover_probability=options.crossover_prob)
        deap_optimization.start(seed=options.seed)

    elif options.tool == 'jmetalpy':
        if options.approach == 'before-after':
            results_dir = os.path.join(
                options.result_dir, f'{options.approach}_{options.surrogate}_switch_{options.switch}')
        elif options.approach == 'incremental-epoch':
            results_dir = os.path.join(
                options.result_dir, f'{options.approach}_{options.simulator}_switch_{options.switch}')
        elif options.approach == 'incremental-Data':
            results_dir = os.path.join(
                options.result_dir, f'{options.approach}_{options.simulator}_{options.H}')
        else:  
            results_dir = os.path.join(
                options.result_dir, f'{options.approach}_{options.simulator}')

        log.info(f"Problem: {options.problem}")
        log.info(f"Robots capacity: {options.capacity}")
        log.info(
            f"Customer requests per hour: {options.min_requests} - {options.max_requests}")
        log.info(f"Simulator strategy: {options.approach}")
        log.info(f"Max simulation duration (hours): {options.duration}")
        log.info(f"Results folder: {results_dir}")
        log.info(f"Population size: {options.population}")
        log.info(
            f"Offspring population size: {options.offspring_population_size}")
        log.info(f"Generations: {options.generations}")
        log.info(f"Crossover probability: {options.crossover_prob}")
        log.info(f"Robots: [{options.low_bound}, {options.robots}]")
        log.info(f"Speed: [{options.low_bound}, {options.speed}]")
        log.info(f"simulator: {options.simulator}")
        log.info(f"switch generation : {options.switch}")

        log.info("Starting simulation in 5 seconds...")

        time.sleep(5)
        
        # load surrogate model in advance. in void recurrsively load model.
        if options.approach == "before-after":
            simulator = None
            surrogate = DNN_utilization(21, 3)
            surrogate.load_state_dict(torch.load(os.path.join(
                "simulator", "bin", "surrogate_model", f'{options.duration}hours_{options.min_requests}_{options.max_requests}', options.surrogate), map_location=torch.device('cpu')))
        elif options.approach == "incremental-epoch":
            simulator = DNN_utilization(21, 3)
            surrogate = None
        elif options.approach == 'incremental-data':
            simulator = DNN_utilization(21, 3)
            surrogate = None
        elif options.simulator != 'simulator':
            simulator = DNN_utilization(21, 3)
            simulator.load_state_dict(torch.load(os.path.join(
                "simulator", "bin", "surrogate_model", f'{options.duration}hours_{options.min_requests}_{options.max_requests}', options.simulator), map_location=torch.device('cpu')))
        else:
            simulator = None
            surrogate = None

        sim = Sim(min_customer_requests_per_hour=options.min_requests,
                  max_customer_requests_per_hour=options.max_requests,
                  robot_loading_capacity=options.capacity,
                  simulation_duration=options.duration,
                  hour=options.start_time,
                  number_of_simulations=options.simulations,
                  threads=options.threads,
                  seed=options.seed,
                  results_folder=f'{options.problem}/{options.approach}_{options.simulator}/gens_{options.generations}/reqs_{options.min_requests}_{options.max_requests}/{options.run_tag}',
                  simulator=options.simulator,
                  surrogate=simulator)
        
        if options.approach == 'before-after':
            sur = Sur(min_customer_requests_per_hour=options.min_requests,
                    max_customer_requests_per_hour=options.max_requests,
                    robot_loading_capacity=options.capacity,
                    simulation_duration=options.duration,
                    hour=options.start_time,
                    number_of_simulations=options.simulations,
                    threads=options.threads,
                    seed=options.seed,
                    results_folder=f'{options.problem}/{options.approach}_{options.simulator}/gens_{options.generations}/reqs_{options.min_requests}_{options.max_requests}/{options.run_tag}',
                    simulator=options.surrogate,
                    surrogate=surrogate)

        if options.problem == 'schedule-assignment':
            end_time = options.start_time + options.duration
            problem = RobotScheduleAssignmentProblem(sim=sim,
                                                     number_of_robots=options.robots,
                                                     speed_bounds=(
                                                         options.low_bound, options.speed),
                                                     business_hours=(
                                                         options.start_time, end_time),
                                                     speed_factor=options.speed_factor)
        elif options.problem == 'schedule-assignment-var-robots':
            end_time = options.start_time + options.duration
            problem = RobotScheduleAssignmentProblemVariableRobots(sim=sim,
                                                                   switch_generation=options.switch,
                                                                   max_number_of_robots=options.robots,
                                                                   speed_bounds=(
                                                                       options.low_bound, options.speed),
                                                                   business_hours=(
                                                                       options.start_time, end_time),
                                                                   speed_factor=options.speed_factor)
        
        elif options.problem == 'schedule-assignment-var-robots-Before-After':
            end_time = options.start_time + options.duration
            problem = RobotScheduleVariableRobotsBeforeAfter(sim=sim,
                                                            sur=sur,
                                                            switch_generation=options.switch,
                                                            max_number_of_robots=options.robots,
                                                            speed_bounds=(
                                                                options.low_bound, options.speed),
                                                            business_hours=(
                                                                options.start_time, end_time),
                                                            speed_factor=options.speed_factor)
        
        elif options.problem == 'schedule-assignment-var-robots-Inc-Epoch':
            end_time = options.start_time + options.duration
            problem = RobotScheduleVariableRobotsIncrementalEpochs(sim=sim,
                                                                   switch_generation=options.switch,
                                                                   max_number_of_robots=options.robots,
                                                                   speed_bounds=(
                                                                       options.low_bound, options.speed),
                                                                   business_hours=(
                                                                       options.start_time, end_time),
                                                                   speed_factor=options.speed_factor,
                                                                   surrogate_type=options.surrogate_type,
                                                                   epoch=options.epoch)
        
        elif options.problem == 'schedule-assignment-var-robots-Inc-Data':
            end_time = options.start_time + options.duration
            problem = RobotScheduleVariableRobotsIncrementalData(sim=sim,
                                                                 run_tag=options.run_tag,
                                                                    max_number_of_robots=options.robots,
                                                                    speed_bounds=(
                                                                        options.low_bound, options.speed),
                                                                    business_hours=(
                                                                        options.start_time, end_time),
                                                                    speed_factor=options.speed_factor,
                                                                    surrogate_type=options.surrogate_type,
                                                                    epoch=options.epoch)
            
        else:
            raise RuntimeError(
                f"{options.problem} is not a valid problem input")

        if options.weighted == 'weighted-classic':
            problem = WeightedProblem(
                problem=problem, weights=[1000.0, 100.0, 1.0])

        max_evaluations = options.population + \
            options.offspring_population_size * options.generations

        if options.parallel == 1:
            pop_evaluator = jmetal.config.store.default_evaluator
        else:
            pop_evaluator = MapEvaluator(processes=options.parallel)

        algorithm = NSGAII(
            problem=problem,
            population_size=options.population,
            offspring_population_size=options.offspring_population_size,
            mutation=IntegerPolynomialMutation(probability=1.0 / problem.number_of_variables,
                                               distribution_index=20),
            crossover=IntegerSBXCrossover(
                probability=options.crossover_prob, distribution_index=20),
            termination_criterion=StoppingByEvaluations(
                max_evaluations=max_evaluations),
            population_evaluator=pop_evaluator
        )

        if options.approach == 'incremental-data':
            algorithm = IncrementalDatahWrapper(algorithm=algorithm, simulator=sim, H=options.H, generations=options.generations)
        elif options.approach == 'incremental-epoch':
            algorithm = IncrementalEpochWrapper(algorithm=algorithm, simulator=sim, switch_generation=options.switch, 
                                         single_evaluator=jmetal.config.store.default_evaluator, multi_evaluator=MapEvaluator(processes=options.parallel), generations=options.generations)
        elif options.approach == 'before-after':
            algorithm = BeforeAfterWrapper(algorithm=algorithm, simulator=sim, surrogate=sur, switch_generation=options.switch, 
                                         single_evaluator=jmetal.config.store.default_evaluator, multi_evaluator=MapEvaluator(processes=options.parallel), generations=options.generations)
            
        elif options.approach == 'standard':
            algorithm = AlgorithmWrapper(
                algorithm=algorithm, simulator=sim, generations=options.generations)
        else:
            raise RuntimeError(f"{options.approach}  is not a valid options")

        # setting random seed based on the run tag to obtain different NSGAII searches
        # if not, the result of different runs would be the same because the random seed is initialized in the simulator
        jmetal_seed = options.jmetal_seed + options.run_tag

        # Adding the seeds and request size of each simulation
        config = options.__dict__
        config['simulations_info'] = {
            'requests_per_hour': sim.request_per_hour, 'seeds': sim.seeds}
        config['jmetal_seed'] = jmetal_seed

        job = DeliveryJob(
            algorithm=algorithm, run_tag=options.run_tag, seed=jmetal_seed, config=config)
        
        
        output_path = os.path.join(results_dir, job.algorithm_tag, job.problem_tag,
                                f'{options.generations}_generations',
                                f'requests_{options.min_requests}_{options.max_requests}')
        
        job.execute(output_path)
