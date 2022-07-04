import numpy as np
import torch
import math
import subprocess
import sys
from collections import defaultdict
sys.path.append('../utils')

from graph  import plot_graph
from embeddings import Embeddings



class MPIMapEnv(object):

	def __init__(self, params):
		super(MPIMapEnv, self).__init__()

		# Dictionary containing data structures in the environment
		# including state, matrices and params. Currently:
		#  - P, M
		#  - embeddings
		#  - Matrices (volume, num. messages and adjacency)
		#  - Reward function (object Reward)
		self.env_state = dict()


		# Communication graph
		graph  = params["Graph"]
		M = graph["M"]
		P = graph["P"]
		self.env_state["M"] = M
		self.env_state["P"] = P

		hyperp = params["Hyperparameters"]
		K = hyperp["K"]
		self.env_state["K"] = K

		# Get embedding from graph features:
		emb = Embeddings(params)
		self.env_state["embeddings"] = emb.get_embeddings()


		# We create three PxP matrices:
		#  1) Volume matrix: total message volume communicated by pairs of procs.
		#  2) Num. msgs matrix: number of messages communicated by pair of procs.
		#  3) Adjacency matrix: simple adjacency matrix

		edges  = graph["comms"]["edges"]
		volume = graph["comms"]["volume"]
		n_msgs = graph["comms"]["n_msgs"]

		# Volume matrix
		volume_matrix = np.zeros((P, P))
		for m, edge in zip(volume, edges):
			volume_matrix[edge[0]][edge[1]] = m
		self.env_state["volume_matrix"] = volume_matrix

		# Num. messages matrix
		msgs_matrix = np.zeros((P, P))
		for n, edge in zip(n_msgs, edges):
			msgs_matrix[edge[0]][edge[1]] = n
		self.env_state["msgs_matrix"] = msgs_matrix

		# Adjacency matrix
		adjacency_matrix = np.zeros((P, P))
		for edge in edges:
			adjacency_matrix[edge[0]][edge[1]] = 1
		self.env_state["adjacency_matrix"] = adjacency_matrix


		# Node capacity:
		#  We assume a number of nodes with a specific capacity (number of
		#   processes that can be executed in each node):
		self.env_state["capacity"] = graph["capacity"]


		# Reward type (TO BE DONE)
		#  Different forms of computing reward signal
		rw_type = params["Config"]["reward_type"]

		if   rw_type == "tLop":
			from tLop.tLop_reward     import Reward

		elif rw_type == "mpi":
			from .mpi.mpi_reward      import Reward

		elif rw_type == "num_msgs":
			from num_msgs.num_msgs_reward     import Reward

		elif rw_type == "volume":
			from volume.volume_reward import Reward

		else:
			raise Exception("The reward function does not exist: ", rw_type)


		# Create Reward object (function)
		self.reward_fxn = Reward()
		self.env_state["reward_fxn"] = self.reward_fxn


		self.verbose = params["Config"]["verbose"]
		if self.verbose:
			print("Adjacency MATRIX: ")
			print(adjacency_matrix)
			print("Volume MATRIX: ")
			print(volume_matrix)
			print("Num. messages MATRIX: ")
			print(msgs_matrix)
			print("Reward computing: ", rw_type)

		return


	def steps (self, actions):

		r, info = self.reward_fxn.get_reward(self.env_state, actions)

		return self.env_state["embeddings"], r, info


	def reset(self):

		self.t     = 0
		self.valid = False

		return self.env_state["embeddings"]
