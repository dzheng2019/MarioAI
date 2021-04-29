# Trains an agent from scratch (no existing AI) using evolution
# GA with no cross-over, just mutation, and random tournament selection
# Not optimized for speed, and just uses a single CPU (mainly for simplicity)

from slimevolleygym.slimevolley import BaselinePolicy
import numpy as np
import pandas as pd
from numpy import tanh 
import random
import gym
import slimevolleygym.mlp as mlp
from slimevolleygym.mlp import Model
from slimevolleygym import multiagent_rollout as rollout

from typing import Tuple, List, Dict


class MyFastModel():

    def __init__(self, include_randomness = False):
        self.randomness = include_randomness
        
        self.dense1 = np.empty((12,13))
        self.dense2 = np.empty((12,13))
        self.dense3 = np.empty((12,13))
        self.dense4 = np.empty((3,13))
        
        self.l1 = self.dense1.size
        self.l2 = self.l1 + self.dense2.size
        self.l3 = self.l2 + self.dense3.size
        self.l4 = self.l3 + self.dense4.size
        
        self.param_count = self.l4
        
        self.variance1 = self.find_var(self.dense1, .1)
        self.variance2 = self.find_var(self.dense2, .1)
        self.variance3 = self.find_var(self.dense3, .1)
        self.variance4 = self.find_var(self.dense4, .1)
    
    def find_var(self, layer, std):
        dimensions = layer.shape
        shape = (dimensions[0], dimensions[1])
        var = np.random.normal(1, std, (dimensions[1], dimensions[1]))
        return var;
        
    def predict(self, inputs):
        inputs = np.array([inputs]).T

        # hidden
        x1 = np.append(inputs, [[1]], axis=0)
        z1 = self.dense1 @ self.variance1 @ x1 if self.randomness else self.dense1 @ x1
        a1 = tanh(z1)
        
        # hidden
        x2 = np.append(a1, [[1]], axis=0)
        z2 = self.dense2 @ self.variance2 @ x2 if self.randomness else self.dense2 @ x2
        a2 = tanh(z2)
        
        # hidden
        x3 = np.append(a2, [[1]], axis=0)
        z3 = self.dense3 @ self.variance3 @ x3 if self.randomness else self.dense3 @ x3
        a3 = tanh(z3)
        
        # hidde
        x4 = np.append(a3, [[1]], axis=0)
        z4 = self.dense4 @ self.variance4 @ x4 if self.randomness else self.dense4 @ x4
        a4 = tanh(z4)
        
        return(a4)

    def set_model_params(self,model_params):
        layer1_params = model_params[0:self.l1]
        layer2_params = model_params[self.l1:self.l2]
        layer3_params = model_params[self.l2:self.l3]
        layer4_params = model_params[self.l3:self.l4]
        
        self.dense1 = np.array(layer1_params).reshape((12,13))
        self.dense2 = np.array(layer2_params).reshape((12,13))
        self.dense3 = np.array(layer3_params).reshape((12,13))
        self.dense4 = np.array(layer4_params).reshape((3,13))
        
        if random.random() < .1:
            self.variance1 = self.find_var(self.dense1, .1)
            self.variance2 = self.find_var(self.dense2, .1)
            self.variance3 = self.find_var(self.dense3, .1)
            self.variance4 = self.find_var(self.dense4, .1)

# Settings
random_seed = 612
population_size = 128
generations = 1125

# Create two instances of a feed forward policy we may need later.
policy_left = MyFastModel()
policy_right = MyFastModel()
param_count = policy_left.param_count # number of parameters
print("Number of parameters of the neural net policy:", param_count) # 273 for slimevolleylite

# create the gym environment, and seed it
env = gym.make("SlimeVolley-v0")
env.seed(random_seed)
np.random.seed(random_seed)

# Bracket Tournament Evolution (Hunger Games)
#
# Top 25% are used as parents for the next generation
# Tournament uses bracket system to reach top 25% players
#
#  [   *           *  ...] - top 25% cut off
#    /   \       /   \
#   *     *     *     * ...
#  / \   / \   / \   / \
# *   * *   * *   * *   * ...
#
# we have 250,000 games pool
# games per tournament: (128 + 128/2 + 128/4) = 224 games
# number of generations: 250,000/224 = 1125 generations

# select top 25% of the population based on the bracket, need 3 levels
def tournament(population):
    all_times: List[int] = []
    top_25: List[Tuple[int, np.array]] = []

    levels_max = 7
    level_for_top_25 = 2
    levels_count = 0

    best_agent = {}

    while len(population) > 1:
        # this devides population into 2
        pairs = zip(population[::2], population[1::2])
        levels_count += 1

        population = []

        for agent_a, agent_b in pairs:
            policy_left.set_model_params(agent_a)
            policy_right.set_model_params(agent_b)
            score, time = rollout(env, policy_right, policy_left)

            all_times.append(time)

            # if score is positive, it means policy_right won
            # on a tie delete agent_a
            if score > 0:
                population.append(agent_a)

                if levels_count == level_for_top_25:
                    top_25.append((time, agent_a))

                if levels_count == levels_max:
                    best_agent = (time, agent_a)

            elif score < 0:
                population.append(agent_b)

                if levels_count == level_for_top_25:
                    top_25.append((time, agent_b))

                if levels_count == levels_max:
                    best_agent = (time, agent_b)

            else:
                population.append(agent_b)

                if levels_count == level_for_top_25:
                    top_25.append((time, agent_b))

                if levels_count == levels_max:
                    best_agent = (time, agent_b)

    return best_agent, top_25, all_times


# continue evolution for given number of generations
def evolution(generations):
    print(f"Starting evolution with {generations} generations")

    data_point_freq = int(generations / 200)
    df = pd.DataFrame(columns=[
        'best time ',
        'mean time for top 25%',
        'standart deviation for top 25%',
        'mean time for all',
        'standart deviation for all'
    ])
    data_point_count = 0
    print(f"Data point frequency is {data_point_freq}")

    best_agent_snapshot_freq = int(generations / 10)
    best_agent_count = 1
    print(f"Best agent frequency is {best_agent_snapshot_freq}")

    # initial parent population, randomly initialized
    population = np.random.normal(size=(population_size, param_count)) * 0.5

    for current_gen in range(generations):
        print(f"Generation N{current_gen}")

        # conduct a tournament to get the best, top 25% of the population, and all times
        best_agent, top_25, all_times = tournament(population)

        population = []

        # for each parent, create 4 mutated children 
        for parent in [agent for (_, agent) in top_25]:
            for _ in range(4):
                child = np.copy(parent)
                child += np.random.normal(size=param_count) * 0.1
                population.append(child)

        # save best agent to a seperate file
        if current_gen % best_agent_snapshot_freq == 0:
            best_agent = best_agent[1]
            np.save(f"complex_bracket_agent_N{best_agent_count}", best_agent)
            best_agent_count += 1

        # save data points to pandas dataframe
        if current_gen % data_point_freq == 0:
            best_time = best_agent[0]
            top_25_times = [time for (time, _) in top_25]
            df.loc[data_point_count] = [
                best_time,
                np.mean(top_25_times),
                np.std(top_25_times),
                np.mean(all_times),
                np.std(all_times)
            ]
            data_point_count += 1

    # save dataframe as csv file
    df.to_csv("bracket_agent_datapoints_complex_model.csv")

evolution(generations)