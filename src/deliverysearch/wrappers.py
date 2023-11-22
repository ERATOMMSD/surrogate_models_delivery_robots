import math
import logging as log
from jmetal.core.algorithm import Algorithm
from simulator.simulator import Simulator
from simulator.surrogate import Surrogate
import torch
from torch.utils.data import DataLoader
import os



class AlgorithmWrapper:

    def __init__(self, algorithm: Algorithm, simulator: Simulator, generations: int):
        self.algorithm = algorithm
        self.original_step = self.algorithm.step
        self.algorithm.__setattr__('step', self.step)
        self.simulator = simulator
        self.generation = 0
        self.generations = generations

    def step(self):
        self.generation += 1
        self.algorithm.problem.update_generation(generation=self.generation)
        return self.original_step()

    def __getattr__(self, called_method):
        if called_method != 'step':
            return getattr(self.algorithm, called_method)
        else:
            return self.step()

class BeforeAfterWrapper(AlgorithmWrapper):
    
    def __init__(self, algorithm: Algorithm, simulator: Simulator, surrogate: Surrogate, switch_generation: int, single_evaluator, multi_evaluator, generations: int):
        super(BeforeAfterWrapper, self).__init__(algorithm, simulator, generations)
        self.surrogate = surrogate
        self.switch_generation = switch_generation
        # self.algorithm.problem.sim = self.surrogate
        
        self.single_evaluator = single_evaluator
        self.multi_evaluator = multi_evaluator
        
        # avoid using both multiprocess and GPU together
        self.algorithm.population_evaluator = self.single_evaluator
        
    def step(self):
        res = super().step()
        if self.generation >= (self.switch_generation-1):
        #     # self.algorithm.problem.sim = self.simulator
        #     # self.algorithm.population_evaluator = self.single_evaluator
            self.algorithm.population_evaluator = self.multi_evaluator
            
        return res


class IncrementalEpochWrapper(AlgorithmWrapper):
    
    def __init__(self, algorithm: Algorithm, simulator: Simulator, switch_generation: int, single_evaluator, multi_evaluator, generations: int):
        super(IncrementalEpochWrapper, self).__init__(algorithm, simulator, generations)

        self.switch_generation = switch_generation
        # self.algorithm.problem.sim = self.surrogate
        
        self.single_evaluator = single_evaluator
        self.multi_evaluator = multi_evaluator
        
        # avoid using both multiprocess and GPU together
        self.algorithm.population_evaluator = self.multi_evaluator
        
    def step(self):
        res = super().step()
        if self.generation % self.switch_generation == 0:
            self.algorithm.problem.sim.surrogate = self.algorithm.problem.train_surrogate.run_train()
        # if self.generation >= (self.switch_generation-1):
        # #     # self.algorithm.problem.sim = self.simulator
        # #     # self.algorithm.population_evaluator = self.single_evaluator
        #     self.algorithm.population_evaluator = self.multi_evaluator
            
        return res
    
class IncrementalDatahWrapper(AlgorithmWrapper):
    
    def __init__(self, algorithm: Algorithm, simulator: Simulator, H: int, generations: int):
        super(IncrementalDatahWrapper, self).__init__(algorithm, simulator, generations)
        self.H = H
        # self.algorithm.problem.sim = self.surrogate
        # avoid using both multiprocess and GPU together
        self.surrogate_types = ['surrogate25_13', 'surrogate50_25', 'surrogate100_50']
        
    def step(self):
        res = super().step()
        if (self.generation == 2*self.H) | (self.generation == 5*self.H) | (self.generation == 9*self.H) :
            surrogate_type = self.surrogate_types.pop(0)
            self.algorithm.problem.train_surrogate.type = surrogate_type
            self.algorithm.problem.train_surrogate.train_ds = torch.load(os.path.join(os.getcwd(), 'surrogate', 'dataset', surrogate_type,'train_ds.pt'))
            self.algorithm.problem.train_surrogate.train_loader = DataLoader(dataset=self.algorithm.problem.train_surrogate.train_ds, batch_size=self.algorithm.problem.train_surrogate.batch_size, shuffle=True)
            self.algorithm.problem.sim.surrogate = self.algorithm.problem.train_surrogate.run_train()
        # if self.generation >= (self.switch_generation-1):
        # #     # self.algorithm.problem.sim = self.simulator
        # #     # self.algorithm.population_evaluator = self.single_evaluator
        #     self.algorithm.population_evaluator = self.multi_evaluator
            
        return res   

class IncrementalSimulationsWrapper(AlgorithmWrapper):

    def __init__(self, algorithm: Algorithm, simulator: Simulator, generations: int, max_simulations: int):
        self.max_simulations = max_simulations
        super(IncrementalSimulationsWrapper, self).__init__(algorithm, simulator, generations)
        self.simulator.set_number_of_simulations(1)

    def update_simulations(self):
        simulations = math.ceil(self.generation / self.generations * self.max_simulations)
        log.getLogger().info(f'Starting generation {self.generation} with {simulations} simulations.')
        self.simulator.set_number_of_simulations(simulations)

    def step(self):
        res = super().step()
        self.update_simulations()
        return res


class IncrementalTimeWrapper(AlgorithmWrapper):

    def __init__(self, algorithm: Algorithm, simulator: Simulator, generations: int, max_time: int, min_time: int = 1):
        self.max_time = max_time
        self.min_time = min_time
        super(IncrementalTimeWrapper, self).__init__(algorithm, simulator, generations)
        self.update_simulation_duration()

    def update_simulation_duration(self):
        simulation_duration = max(self.min_time, math.ceil(self.generation / self.generations * self.max_time))
        log.getLogger().info(f'Starting generation {self.generation} with {simulation_duration} hours simulation.')
        self.simulator.update_simulation_duration(simulation_duration)

    def step(self):
        res = super().step()
        self.update_simulation_duration()
        return res
