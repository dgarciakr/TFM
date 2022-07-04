#!/usr/bin/env python

## MonteCarlo Policy Gradients (REINFORCE)
#
# Implementing MonteCarlo policy gradient algorithm
# (REINFORCE, [Williams, 1992]) to learn a mapping for a
# given communication graph representing an application.

import numpy  as np
import pandas as pd

import time
import json
import os
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from torch.distributions import Categorical, Multinomial

from collections import defaultdict

from graph      import adjacency, binomial_broadcast
from embeddings import Embeddings

DIGITS = 4

class Output(object):

	def __init__ (self, params):

		# Communication graph
		graph      = params["Graph"]
		self.P          = graph["P"]
		self.root       = graph["root"]
		self.M          = graph["M"]
		self.node_names = graph["node_names"]

		# Configuration parameters
		config        = params["Config"]
		self.rw_type       = config["reward_type"]
		self.baseline_type = config["Baseline"]
		self.verbose       = config["verbose"]     # Verbosity
		self.verbosity_int = config["verbosity_interval"]

		# Hyperparameters
		hyperparams = params["Hyperparameters"]
		self.gamma       = hyperparams["gamma"]       # Discounted factor
		self.alpha       = hyperparams["alpha"]       # Learning rate
		self.n_episodes  = hyperparams["n_episodes"]
		self.K           = hyperparams["K"]           # Num. samples

		# Policy Network
		policy    = params["Policy"]
		self.nn_type   = policy["type"]
		self.optimizer = policy["optimizer"]
		self.typecell  = policy["typecell"]

		# GNN
		gnn       = params["GNN"]
		self.gnn_type  = gnn["type"]
		self.dims      = gnn["dimensions"]
		self.n_walks   = gnn["n_walks"]
		self.walk_len  = gnn["walk_length"]

		# Output config.
		output = params["Output"]
		self.output_file = output["output_file"]
		self.graph_file  = output["graph_file"]

		# History values
		self.J_history    = []
		self.Rw_history   = []
		self.D_rw_history = []
		self.R_history    = []
		self.T_history    = []


		# Write header to file
		self.f = open(self.output_file, "w")

		self.f.write("#P: "           + str(self.P)             + "\n")
		self.f.write("#M: "           + str(self.M)             + "\n")
		self.f.write("#alpha: "       + str(self.alpha)         + "\n")
		self.f.write("#gamma: "       + str(self.gamma)         + "\n")
		self.f.write("#n_episodes: "  + str(self.n_episodes)    + "\n")
		self.f.write("#K: "           + str(self.K)             + "\n")
		self.f.write("#Baseline: "    + str(self.baseline_type) + "\n")

		self.f.write("#Node names: "  + str(self.node_names)    + "\n")
		self.f.write("#Reward_type: " + str(self.rw_type)       + "\n")

		self.f.write("#Policy type: " + str(self.nn_type)       + "\n")
		self.f.write("#Optimizer: "   + str(self.optimizer)     + "\n")
		self.f.write("#Cell type: "   + str(self.typecell)      + "\n")

		self.f.write("#GNN type: "    + str(self.gnn_type)      + "\n")
		self.f.write("#Dimensions: "  + str(self.dims)          + "\n")
		self.f.write("#Num. Walk: "   + str(self.n_walks)       + "\n")
		self.f.write("#Walk length: " + str(self.walk_len)      + "\n")

		self.f.write("#StartTime: "   + str(time.time())        + "\n")


		try:
			slurm_job_id   = os.environ['SLURM_JOB_ID']
			slurm_job_id   = "slurm-" + slurm_job_id + ".txt"
			slurm_nodelist = os.environ['SLURM_NODELIST']
		except:
			slurm_job_id   = '0'
			slurm_nodelist = 'local'

		self.f.write("#slurm file: "  + str(slurm_job_id)    + "\n")
		self.f.write("#node list: "   + str(slurm_nodelist)  + "\n")

		self.f.write("# \n")

		self.f.write("#    e     \t     J     \t    time    \t     T     \t    Reward    \t    D_Rw    \t    B_avg    \t    B_std    \t    Return    \t  log_probs)   \t     Actions (K = 0)          \n")
		self.f.write("#--------- \t --------- \t ---------- \t --------- \t ------------ \t ---------- \t ----------- \t ----------- \t ------------ \t ------------- \t ---------------------------- \n")

		return


	def __output_screen (self, e_info):

		# Mean info of all samples along P dimension
		n_episode   = e_info["Episode"]
		J           = e_info["J"].item()
		T           = e_info["T"]
		Rw          = e_info["Rewards"].sum(dim=1).mean().item()
		D_rw        = e_info["Discounted_rws"].sum(dim=1).mean().item()
		R           = e_info["Returns"].sum(dim=1).mean().item()

		B_avg       = e_info["B_avg"].item()
		B_std       = e_info["B_std"].item()

		logprobs    = e_info["Logprobs"].sum(dim=1).mean().item()
		entropy     = e_info["Entropy"].mean().item()

		actions     = e_info["Actions"].type(torch.IntTensor)

		# Save history
		self.J_history.append(J)
		self.Rw_history.append(Rw)
		self.D_rw_history.append(D_rw)
		self.R_history.append(R)
		self.T_history.append(T)

		# Return if nothing to show
		if (not self.verbose) or (n_episode % self.verbosity_int) != 0:
			return


		# Info per sample (k)
		J_tau       = e_info["J_tau"].detach().tolist()
		Rw_k        = e_info["Rewards"].sum(dim=1).tolist()
		D_rw_k      = e_info["Discounted_rws"].sum(dim=1).tolist()
		R_k         = e_info["Returns"].sum(dim=1).tolist()
		logprobs_k  = e_info["Logprobs"].sum(dim=1).detach().tolist()


		# Logits (same for all episodes)
		logits     = e_info["Logits"].detach()

		# Prediction data (no training)
		ractions      = e_info["MaxActions"].squeeze(0).type(torch.IntTensor).tolist()
		prewards      = e_info["PredRewards"].squeeze(0).sum().item()

		# Mean of data along interval and mean of all history values
		start = n_episode - self.verbosity_int
		end   = n_episode
		n     = end - start

		# Output:
		costs = np.array(self.J_history)

		print("\n Episode ", n_episode," of ", self.n_episodes)
		print("--------------------------------------")

		print("===>>>  HISTORY: ")
		print("  Loss (mean/std/max/min): ", np.round(costs[:end].mean(), DIGITS),
		                                     np.round(costs[:end].std(),  DIGITS),
											 np.round(costs[:end].max(),  DIGITS),
											 np.round(costs[:end].min(),  DIGITS))
		print("  Reward:      ", np.round(sum(self.Rw_history[:end])   / n_episode, DIGITS))
		print("  Discted_rw:  ", np.round(sum(self.D_rw_history[:end]) / n_episode, DIGITS))
		print("  Returns (B): ", np.round(sum(self.R_history[:end])    / n_episode, DIGITS))

		print("===>>>  INTERVAL: ")
		print("  Loss (mean/std/max/min): ", np.round(costs[start:end].mean(), DIGITS),
		                                     np.round(costs[start:end].std(),  DIGITS),
											 np.round(costs[start:end].max(),  DIGITS),
											 np.round(costs[start:end].min(),  DIGITS))
		print("  Reward:      ", np.round(sum(self.Rw_history[start:end])   / n, DIGITS))
		print("  Discted_rw:  ", np.round(sum(self.D_rw_history[start:end]) / n, DIGITS))
		print("  Return (B):  ", np.round(sum(self.R_history[start:end])    / n, DIGITS))

		print("===>>>  EPISODE: ")
		print("  Loss:            ", np.round(J, DIGITS))
		print("  Reward:          ", np.round(Rw, DIGITS))
		print("  Discted_rw:      ", np.round(D_rw, DIGITS))
		print("  Baseline (mu,s): ", B_avg, B_std)
		print("  Return (-B):     ", np.round(R, DIGITS))
		print("  Log probs:       ", np.round(logprobs, DIGITS))
		print("  Entropy:         ", np.round(entropy, DIGITS))

		print("  K-Loss:          ", np.round(J_tau, DIGITS))
		print("  K-Rewards:       ", [np.round(x, DIGITS) for x in Rw_k])
		print("  K-Discted rws:   ", [np.round(x, DIGITS) for x in D_rw_k])
		print("  K-Returns (-B):  ", [np.round(x, DIGITS) for x in R_k])
		print("  K-Log probs:     ", [np.round(x, DIGITS) for x in logprobs_k])

		print("  Actions (K):     ")
		print(actions)
		assignment = self.__assigned_nodes(actions[0])
		print("  Assignment (K=0): ", assignment)

		print("  Logits from NN:  ")
		print(np.around(logits.cpu().view(self.P, self.M), DIGITS))

		print("===>>>  PREDICT: ")
		print("  ArgMax Actions:    ", ractions)
		rassignment = self.__assigned_nodes(ractions)
		print("  ArgMax Assignment: ", rassignment)
		print("  Rewards:           ", np.round(prewards, DIGITS))

		print(flush=True)
		return


	def __assigned_nodes (self, a):

		devs = defaultdict(list)

		for p in range(self.P):
			node = self.node_names[a[p]]
			devs[node].append(p)

		return sorted(devs.items())


	def __output_file (self, e_info):

		n_episode   = e_info["Episode"]
		J           = e_info["J"].item()
		T           = e_info["T"]
		Rw          = e_info["Rewards"].sum(dim=1).mean().item()
		D_rw        = e_info["Discounted_rws"].sum(dim=1).mean().item()
		R           = e_info["Returns"].sum(dim=1).mean().item()

		B_avg       = e_info["B_avg"].item()
		B_std       = e_info["B_std"].item()

		logprobs    = e_info["Logprobs"].sum(dim=1).mean().item()

		actions     = e_info["Actions"][0].squeeze(0).type(torch.IntTensor).tolist()

		self.f.write(str(n_episode)    + " # " +
					 str(J)            + " # " +
					 str(time.time())  + " # " +
					 str(T)            + " # " +
					 str(Rw)           + " # " +
					 str(D_rw)         + " # " +
					 str(B_avg)        + " # " +
					 str(B_std)        + " # " +
					 str(R)            + " # " +
					 str(logprobs)     + " # " +
					 str(actions)      + "\n")

		return


	# Main interface operation
	def render(self, e_info):

		self.__output_screen(e_info)
		self.__output_file(e_info)

		n_episode = e_info["Episode"]
		if n_episode == self.n_episodes:
			self.f.flush()
			self.f.close()

		return
