# Trains an agent from scratch (no existing AI) using evolution
# GA with no cross-over, just mutation, and random tournament selection
# Not optimized for speed, and just uses a single CPU (mainly for simplicity)

import os
import json
import numpy as np
import gym
import slimevolleygym
import slimevolleygym.mlp as mlp
from slimevolleygym.mlp import Model
from slimevolleygym import multiagent_rollout as rollout
from myFastSlimeAgent import MyFastModel as ComplexModel
from time import sleep
from pyglet.window import key
import itertools
import math
import pandas as pd

# Settings
random_seed = 612
population_size = 128
total_tournaments = 250000
save_freq = 1000

modeltype = ComplexModel

policy_left = modeltype()
policy_right = modeltype()

param_count = modeltype().param_count
print("Number of parameters of the neural net policy:", param_count)

# store our population here
population = np.random.normal(size=(population_size, param_count)) * 0.5

    
winning_streak = [0] * population_size # store the number of wins for this agent (including mutated ones)

# create the gym environment, and seed it
env = gym.make("SlimeVolley-v0")
env.seed(random_seed)
np.random.seed(random_seed)

history = []
stats = pd.DataFrame(columns=['tournament', 'best_winning_streak','mean_duration','stdev'])

for tournament in range(1, total_tournaments+1):
  m, n = np.random.choice(population_size, 2, replace=False)

  policy_left.set_model_params(population[m])
  policy_right.set_model_params(population[n])

  # the match between the mth and nth member of the population
  score, length = rollout(env, policy_right, policy_left)

  history.append(length)

  # if score is positive, it means policy_right won.
  if score == 0: # if the game is tied, add noise to the left agent.
    population[m] += np.random.normal(size=param_count) * 0.1

  if score > 0:
    population[m] = population[n] + np.random.normal(size=param_count) * 0.1
    winning_streak[m] = winning_streak[n]
    winning_streak[n] += 1

  if score < 0:
    population[n] = population[m] + np.random.normal(size=param_count) * 0.1
    winning_streak[n] = winning_streak[m]
    winning_streak[m] += 1

    
  if (tournament ) % 1000 == 0:
    record_holder = np.argmax(winning_streak)
    record = winning_streak[record_holder]
    print("tournament:", tournament,
          "best_winning_streak:", record,
          "mean_duration", np.mean(history),
          "stdev:", np.std(history),
         )
    row = [tournament, record, np.mean(history),np.std(history)]
    stats.loc[len(stats.index)] = row
    history = []
  
  # Save best every 1000 games
  if (tournament) % 25000 == 0:
    print("Saving Best")
    record_holder = np.argmax(winning_streak)
    best_player = population[record_holder]
    best_player = np.array(best_player)
    np.save(f'complex_models/tournament{tournament}.npy',best_player)
  
stats.to_csv('stats.csv')
