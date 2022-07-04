#!/usr/bin/env python

## MonteCarlo Policy Gradients (REINFORCE algorithm)
#
# Implementing MonteCarlo policy gradient algorithm
# (REINFORCE, [Williams, 1992]) to learn a mapping for a
# given communication graph representing an application.

import numpy  as np

import decimal
import time
import pdb
import json
import os
import sys
sys.path.append('./env')
sys.path.append('./agent/REINFORCE')
sys.path.append('./utils')

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from torch.distributions import Categorical, Multinomial

from environ    import MPIMapEnv
from agent      import Agent
from graph      import adjacency
from config     import read_config


##########   MAIN   ##########

# Read problem description from JSON file
config = read_config()

# Create Environment instance
env    = MPIMapEnv(params=config)

# Create Agent instance
agent  = Agent(env=env, params=config)


######################
# Main Learning code:
######################

start = time.time()

n_episodes = config["Hyperparameters"]["n_episodes"]

for episode in range(n_episodes):

	s = env.reset()
	agent.reset()

	# Generate a complete trajectory (on-policy)
	a = agent.generate_trajectory(s)

	# Get rewards of the trajectory
	s_, r, info = env.steps(a)

	# Save values to perform learning
	agent.save_trajectory(s, a, r, s_, info)

	# Learn the policy
	agent.learn()


end = time.time()
print("Wallclock time: ", end - start)

s = env.reset()
a, _ = agent.predict(s)
print("Final Predicted trajectory: ", a)
