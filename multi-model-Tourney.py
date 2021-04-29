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
total_tournaments = 100000
save_freq = 1000

# Create two instances of a feed forward policy we may need later.

# policy_left = Model(mlp.games['slimevolleylite'])
# policy_right = Model(mlp.games['slimevolleylite'])
# param_count = policy_left.param_count



param_count = ComplexModel().param_count
print("Number of parameters of the neural net policy:", ComplexModel().param_count, Model(mlp.games['slimevolleylite']).param_count)

# store our population here
half = int(population_size/2)
popC = np.random.normal(size=(half, ComplexModel().param_count)) * 0.5 # each row is an agent.
popS = np.random.normal(size=(half, Model(mlp.games['slimevolleylite']).param_count)) * 0.5 # each row is an agent.



population = popC.tolist() + popS.tolist() 
agents = []

for weights in popC:
    agent = ComplexModel()
    agent.set_model_params(weights)
    agents.append(agent)

for weights in popS:
    agent = Model(mlp.games['slimevolleylite'])
    agent.set_model_params(weights)
    agents.append(agent)

    
winning_streak = [0] * population_size # store the number of wins for this agent (including mutated ones)

# create the gym environment, and seed it
env = gym.make("SlimeVolley-v0")
env.seed(random_seed)
np.random.seed(random_seed)

history = []
stats = pd.DataFrame(columns=['tournament', 'best_winning_streak','mean_duration','stdev','complex_ratio'])
for tournament in range(1, total_tournaments+1):
  left, right = np.random.choice(population_size, 2, replace=False)
  policy_left = agents[left]
  policy_right= agents[right]
  # the match between the mth and nth member of the population

  score, length = rollout(env, policy_right, policy_left)

  history.append(length)

  if score == 0: # if the game is tied, add noise to the left agent.
    num_params = agents[left].param_count
    noise = (np.random.normal(size=num_params) * 0.1).tolist()
    population[left] =  [population[left][i] + noise[i] for i in range(num_params)]  
    agents[left].set_model_params(population[left])

  if score > 0: # policy right WON
    agents[left] = agents[right]

    num_params = agents[right].param_count
    noise = (np.random.normal(size=num_params) * 0.1).tolist()

    population[left] = [population[right][i] + noise[i] for i in range(num_params)]    
    agents[left].set_model_params(population[left])


    winning_streak[left] = winning_streak[right]
    winning_streak[right] += 1

  if score < 0: # policy left WON
    agents[right] = agents[left]

    num_params = agents[left].param_count
    noise = (np.random.normal(size=num_params) * 0.1).tolist()

    population[right] = [population[left][i] + noise[i] for i in range(num_params)]    
    agents[right].set_model_params(population[right])


    winning_streak[right] = winning_streak[left]
    winning_streak[left] += 1
    
  if (tournament ) % 1000 == 0:
    record_holder = np.argmax(winning_streak)
    record = winning_streak[record_holder]
    complex_count = 0
    for agent in agents:
      if type(agent) is ComplexModel:
        complex_count+=1
    complex_ratio = complex_count/population_size

    print("tournament:", tournament,
          "best_winning_streak:", record,
          "mean_duration", np.mean(history),
          "stdev:", np.std(history),
          "complex_ratio", complex_ratio
         )

    row = [tournament, record, np.mean(history),np.std(history), complex_ratio]
    stats.loc[len(stats.index)] = row
    history = []

  # Save best every 1000 games
  if (tournament) % 25000 == 0:
    record_holder = np.argmax(winning_streak)
    best_player = population[record_holder]
    best_player = np.array(best_player)
    np.save(f'models/tournament{tournament}.npy',best_player)

stats.to_csv('stats_both.csv')

# record_holder = np.argmax(winning_streak)
# best_player = population[record_holder]
# policy_left.set_model_params(population[m])


# env = gym.make("SlimeVolleySurvivalNoFrameskip-v0")
# obs = env.reset()

# while True:    
#     action = policy_left.predict(obs)
#     obs, reward, done, info = env.step(action)
#     sleep(0.02)
#     env.render()
#     if done:
#       obs = env.reset()