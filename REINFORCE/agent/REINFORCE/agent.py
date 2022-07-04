#!/usr/bin/env python

## REINFORCE agent

import numpy  as np
import decimal

import os
import sys
sys.path.append('../Env')
sys.path.append('../utils')

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from torch.distributions import Categorical, Multinomial

import time
import pdb
import json

from environ    import MPIMapEnv
from outputs    import Output

from seq2seq         import PolicyNetwork
# from seq2seq_attn    import PolicyNetwork



# Agent

class Agent(object):

	def __init__ (self, env, params):

		# Communication graph
		graph = params["Graph"]
		self.P        = graph["P"]
		self.M        = graph["M"]
		self.capacity = torch.tensor(graph["capacity"], dtype=torch.long).detach()

		# Hyperparameters
		hyperparams = params["Hyperparameters"]
		self.gamma       = hyperparams["gamma"]   # Discounted factor
		self.alpha       = hyperparams["alpha"]   # Learning rate
		self.K           = hyperparams["K"]       # Num. samples per batch

		self.config        = params["Config"]
		self.verbose       = self.config["verbose"]     # Verbosity
		self.verbosity_int = self.config["verbosity_interval"]

		# Store loss and episode length evolution
		self.n_episode   = 0
		self.t           = 0

		self.env = env

		# Optimization on GPUs
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		# Parametrized Policy network
		self.policy = PolicyNetwork(params).to(self.device)
		self.optimizer = torch.optim.Adam(self.policy.parameters(),
		    						  	  lr=self.alpha)

		# Information generated in each episode
		self.e_results = {}

		# Generate results using an Output object
		self.output = Output(params=params)


	def __put_results(self, results):

		if self.verbose and (int(results["Episode"]) % self.verbosity_int) == 0:
			s = self.env.reset()
			a, info = self.predict(s)

			results["MaxActions"]  = info["MaxActions"]
			results["PredRewards"] = info["PredRewards"]

		self.output.render(results)
		return


	def __get_return(self, rewards):

		T = len(rewards[0])
		returns = torch.zeros((self.K, T), dtype=torch.float).to(self.device)
		future_ret = 0

		for t in reversed(range(T)):

			future_ret = rewards[:, t] + self.gamma * future_ret
			returns[:, t] = future_ret

		# Discounted rewards
		self.e_results["Discounted_rws"] = returns

		# Constant baseline and normalization
		avg = returns.mean(dim=1)
		std = returns.std(dim=1)

		self.e_results["B_avg"] = avg.mean()
		self.e_results["B_std"] = std.mean()

		# TBD: 23-11-2021. Deactivated. It drives returns to 0 (???)
		# returns = torch.sub(returns, avg.unsqueeze(dim=1))
		# returns = torch.div(returns, std.unsqueeze(dim=1) + np.finfo(float).eps)

		return returns


	# New episode information and status
	def reset(self):

		# Policy network
		self.policy.train()

		self.n_episode += 1
		self.t          = 0


	def generate_trajectory (self, s):

		state_tensor = torch.FloatTensor(s).unsqueeze(0).to(self.device)

		action_probs = self.policy(state_tensor).to(self.device)
		action_probs = action_probs.view(self.P, -1)

		a_dist   = Categorical(logits=action_probs)
		actions  = a_dist.sample([self.K])
		logprobs = a_dist.log_prob(actions)

		self.t += self.P

		# Save info to output
		self.e_results["Logits"]   = action_probs
		self.e_results["Logprobs"] = logprobs
		self.e_results["Entropy"]  = a_dist.entropy()

		return actions


	def save_trajectory(self, s, a, r, s_, info):

		self.e_results["Episode"]  = self.n_episode
		self.e_results["Actions"]  = a
		self.e_results["Rewards"]  = r

		return


	def learn (self):

		# Compute discounted rewards (to Tensor):
		rewards = self.e_results["Rewards"]
		returns = self.__get_return(rewards).to(self.device)

		logprobs = self.e_results["Logprobs"].to(self.device)

		loss = - (returns * logprobs)
		loss = loss.sum(dim=1)
		self.e_results["J_tau"] = loss  # Loss per trajectory
		loss = loss.mean()
		# loss = - torch.dot(discounted_reward, logprob_tensor)

		# Update parameters:
		self.optimizer.zero_grad()
		loss.backward()        # Compute loss gradient
		self.optimizer.step()  # Update parameters

		# Complete information
		self.e_results["J"]        = loss     # Average loss
		self.e_results["T"]        = self.t
		self.e_results["Returns"]  = returns

		self.__put_results(self.e_results)
		return


	def predict (self, s):

		self.policy.eval()

		state_tensor = torch.FloatTensor(s).unsqueeze(0).to(self.device)

		# Policy: next action
		with torch.no_grad():
			action_probs = self.policy(state_tensor).to(self.device)
			action_probs = action_probs.view(self.P, -1)

			max_actions = action_probs.argmax(dim=1).unsqueeze(0)

			s_, r, info = self.env.steps(max_actions)

		info["MaxActions"]  = max_actions[0]
		info["PredRewards"] = r[0]

		return max_actions, info
