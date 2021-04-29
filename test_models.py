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

env = gym.make("SlimeVolley-v0")

genes0 = np.load('complex_models/tournament250000.npy').tolist()
complex_pol = ComplexModel()
complex_pol.set_model_params(genes0)

genes1 = np.load('simple_models/tournament250000.npy').tolist()
simple_pol = Model(mlp.games['slimevolleylite'])
simple_pol.set_model_params(genes1)

obs0 = env.reset()
obs1 = obs0 # same observation at the very beginning for the other 
# env.render()

defaultAction=[0,0,0]

complex_rec = 0
simple_rec = 0
t = 0
history = []
while True:
    done = False
    score, length = rollout(env, complex_pol, simple_pol)
    if score > 0:
        complex_rec +=1
    if score < 0:
        simple_rec +=1
    history.append(length)
    if t % 100 == 0:
        print(t)
        print('Complex', complex_rec)
        print('Simple', simple_rec)
        print('Avg Time', np.average(history))
        print('STD Time', np.std(history))
    t+=1

viewer.close()
env.close()



# while True:
#     action0 = policy0.predict(obs0)
#     action1 = policy1.predict(obs1) 

#     # left is policy, or the simple model
#     #                                   right,   left 
#     obs0, reward, done, info = env.step(defaultAction, action0)
#     obs1 = info['otherObs']
    
#     # 1 if right wins
#     # -1 if left wins
#     print(reward)
#     sleep(0.01)
#     env.render()

#     if done:
#         obs = env.reset()