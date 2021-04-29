# Trains an agent from scratch (no existing AI) using evolution
# GA with no cross-over, just mutation, and random tournament selection
# Not optimized for speed, and just uses a single CPU (mainly for simplicity)

import itertools
from slimevolleygym.slimevolley import BaselinePolicy
import numpy as np
import pandas as pd
import gym
import slimevolleygym.mlp as mlp
from slimevolleygym.mlp import Model
from slimevolleygym import multiagent_rollout as rollout

from typing import Tuple, List, Dict

# Settings
random_seed = 612
population_size = 128
generations = 2000

# Create two instances of a feed forward policy we may need later.
policy_left = Model(mlp.games['slimevolleylite'])
policy_right = Model(mlp.games['slimevolleylite'])
param_count = policy_left.param_count # number of parameters
print("Number of parameters of the neural net policy:", param_count) # 273 for slimevolleylite

# create the gym environment, and seed it
env = gym.make("SlimeVolley-v0")
env.seed(random_seed)
np.random.seed(random_seed)

# Cooperative Evolution (Olympic Games)
#
# Top 25% are used as parents for the next generation
# Longest play among all players, pick top 25% 
# 
# [ *   * ] - top 25%  *   *   *   *   *   *   ...
#   |   |              |   |   |   |   |   |
#   *   *              *   *   *   *   *   *   ...
#
# longest <-------------------------------> fastest
#
# we have 250,000 games pool
# games per tournament: 128 games
# number of generations: 250,000/128 = 2000 generations

cutoff = int(population_size * 0.25)

# select top 25% of the population based on the longest time in play
def tournament(population) -> Tuple[List[Tuple[int, np.array]], List[int]]:
    results: List[Tuple[int, np.array]] = []
    all_times: List[int] = []

    # print("before for loop")

    # run game for each agent and record the time
    for agent in population:
        policy_left.set_model_params(agent)
        policy_right.set_model_params(agent)
        _, time = rollout(env, policy_right, policy_left)
        results.append((time, agent))
        all_times.append(time)

    # print("after for loop before sort")

    top_25 = sorted(results, key=lambda tup: tup[0], reverse=True)[:cutoff]

    # print("after sort")

    return top_25, all_times


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
    # print(f"Initialized first population to random")

    for current_gen in range(generations):
        print(f"Generation N{current_gen}")

        # conduct a tournament to get top 25% of the population
        top_25, all_times = tournament(population)

        population = []

        # print("before population creation")
        # for each parent, create 4 mutated children 
        for parent in [agent for (_, agent) in top_25]:
            for _ in range(4):
                child = np.copy(parent)
                child += np.random.normal(size=param_count) * 0.1
                population.append(child)
        # print("before population creation")

        # save best agent to a seperate file
        if current_gen % best_agent_snapshot_freq == 0:
            # print("before file")
            best_agent = top_25[0][1]
            np.save(f"simple_friendly_agent_N{best_agent_count}", best_agent)
            best_agent_count += 1
            # print("after file")

        # save data points to pandas dataframe
        if current_gen % data_point_freq == 0:
            # print("before df")
            best_time = top_25[0][0]
            top_25_times = [time for (time, _) in top_25]
            df.loc[data_point_count] = [
                best_time,
                np.mean(top_25_times),
                np.std(top_25_times),
                np.mean(all_times),
                np.std(all_times)
            ]
            data_point_count += 1
            # print("after df")

    # save dataframe as csv file
    df.to_csv("frindly_agent_datapoints_simple_model.csv")

evolution(generations)