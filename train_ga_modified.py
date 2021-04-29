import os
import json
from slimevolleygym.slimevolley import BaselinePolicy
import numpy as np
import gym
import slimevolleygym
import slimevolleygym.mlp as mlp
from slimevolleygym.mlp import Model
from slimevolleygym import multiagent_rollout as rollout

# Settings
random_seed = 612
population_size = 128
save_freq = 1000
generations = 2000

# Log results
logdir = "ga_selfplay"
if not os.path.exists(logdir):
  os.makedirs(logdir)


# Create two instances of a feed forward policy we may need later.
policy_left = Model(mlp.games['slimevolleylite'])
policy_right = Model(mlp.games['slimevolleylite'])
param_count = policy_left.param_count # number of parameters
print("Number of parameters of the neural net policy:", param_count) # 273 for slimevolleylite

# create the gym environment, and seed it
env = gym.make("SlimeVolley-v0")
env.seed(random_seed)
np.random.seed(random_seed)

def tournament_select(population, num_tournaments):
  
  # initially, there is no winner
  winner = None
  maxScore = 0

  finalLength = 0

  for tournament in range(1, num_tournaments+1):

    # get competing parents from population size
    p1, p2 = np.random.choice(population_size, 2, replace=False) 

    # set policies to be the competing parents
    policy_left.set_model_params(population[p1])
    policy_right.set_model_params(population[p2])

    # the match between the mth and nth member of the population
    score, length = rollout(env, policy_right, policy_left)

    if score == 0: # tied
      currWinner = population[p1]
      
    if score > 0: # right agent won
      currWinner = population[p2]

    if score < 0: # left agent won
      currWinner = population[p1]

    # if there was no previous winner, initialize it with the current winner
    if winner is None:
      winner = currWinner
      maxScore = abs(score)
      finalLength = length
    # otherwise set the two winners to compete and get the better agent
    else: 
      policy_left.set_model_params(winner)
      policy_right.set_model_params(currWinner)

      score, length = rollout(env, policy_right, policy_left)
      
      # if the right agent won, we want to replace the overall winner. otherwise don't do anything
      if score > 0: # right agent won
        winner = currWinner
        maxScore = abs(score)
        finalLength = length

  return winner, maxScore, finalLength



def get_optimal_agent(parent_population, population_size):
  # get the best of the child population (not necessarily the best, but highest probability)
  agent_1, score_1, length_1 = tournament_select(parent_population, int(population_size / 2))

  agent_2, score_2, length_2 = tournament_select(parent_population, int(population_size / 2))

  policy_left.set_model_params(agent_1)
  policy_right.set_model_params(agent_2)

  score, length = rollout(env, policy_right, policy_left)

  agent_3 = None

  if score > 0:
    agent_3 = agent_2
  else: 
    agent_3 = agent_1

  agent_4, score_4, length_4 = tournament_select(parent_population, int(population_size / 2))

  policy_left.set_model_params(agent_3)
  policy_right.set_model_params(agent_4)

  score_final, length = rollout(env, policy_right, policy_left)

  optimal_agent = None
  if score_final > 0:
    optimal_agent = agent_4
  else: 
    optimal_agent = agent_3
  
  policy_right.set_model_params(optimal_agent)

  score_against_base, length = rollout(env, BaselinePolicy(), policy_left)
  return optimal_agent, score_against_base

# parent population, randomly initialized
parent_population = np.random.normal(size=(population_size, param_count)) * 0.5 # each row is an agent. (parent population)

# unique winners
unique_winners = np.array([])

avg_length = np.array([])
avg_scores = np.array([])
std_scores = np.array([])

# initialize optimal agent
optimal_agent = None

# genetic algorithm, evolution over generations
for generation in range(0, generations):
  
  # initial empty child population to populate
  child_population = []

  # list of (parent, score)
  parents = []

  # calculate average time taken for this generation
  avg_length_gen = np.array([])

  # populate the list of (parent, score) using tournament selection
  while (len(parents) < population_size):

    # binary tournament selection
    # length is how long game took, calculate average game time of every generation
    parent1, score1, length1 = tournament_select(parent_population, 1)
    parent2, score2, length2 = tournament_select(parent_population, 1)

    # likely to have duplicate parents here because of how often we're picking
    parents.append((parent1, score1))
    parents.append((parent2, score2))

    # STATS
    # keep track # of unique winners in every generation (store as numpy array)
    if not(parent1 in unique_winners):
      unique_winners = np.append(unique_winners, parent1)
    
    if not(parent2 in unique_winners):
      unique_winners = np.append(unique_winners, parent2)
    
    # add length to generation
    avg_length_gen = np.append(avg_length_gen, length1)
    avg_length_gen = np.append(avg_length_gen, length2)
  
  # STATS
  # ADD AVERAGE GAME TIME 
  if generation % 10 == 0:
    print("testing avg length for " + str(generation))
    avg_length = np.append(avg_length, np.mean(avg_length_gen))

  # our elite agents
  elite_size = int(population_size * 0.25)
  elite_agents = []
  
  # STATS:
  elite_agents_scores = np.array([])

  # sort our list of (parent, score) tuples by the score in descending order
  parents.sort(key = lambda x: x[1], reverse=True) 
  print(parents)

  for i in range(elite_size):

    parent = parents[i] # (parent, score) tuple

    elite_agents.append(parent[0])
    elite_agents_scores = np.append(elite_agents_scores, parent[1])

  if generation % 10 == 0:
    # STATS
    # keep track of the average score and std of the top 25%
    avg_scores = np.append(avg_scores, np.mean(elite_agents_scores))
    std_scores = np.append(std_scores, np.std(elite_agents_scores))

    print(avg_scores)
    print(std_scores)

  # add the elite agents to the new population
  for idx in range(len(elite_agents)):
    child_population.append(elite_agents[idx])

  idx = 0
  # we want to take the top 25 percent of the parents and mutate on them
  while (len(child_population) < population_size):
    child = elite_agents[idx % elite_size]
    child += np.random.normal(size=param_count) * 0.1
    child_population.append(child)

    idx += 1

  # initialize new parents for next generation
  parent_population = child_population


  # save best agent at this generation
  if generation % 200 == 0:
    # get optimal agent and its score against the baseline
    optimal_agent, score = get_optimal_agent(parent_population, population_size)
    print("saving for " + str(generation))
    # save agent for every generations
    np.save(f'slimevolleygym-master/training_scripts/ga_saved_agents_gens/ga_agent' + str(generation), optimal_agent)

# saved at the end
# STATS: 
# write average length, score and standard deviations to a file
np.save(f'slimevolleygym-master/training_scripts/ga_stats/avg_length', avg_length)
np.save(f'slimevolleygym-master/training_scripts/ga_stats/avg_scores', avg_scores)
np.save(f'slimevolleygym-master/training_scripts/ga_stats/std_scores', std_scores)
    

def get_winner():
  return optimal_agent

# policy_left.set_model_params(optimal_agent_final)

# score = rollout(env, BaselinePolicy(), policy_left)

# # if score is positive, then policy_left lost
# print(score)