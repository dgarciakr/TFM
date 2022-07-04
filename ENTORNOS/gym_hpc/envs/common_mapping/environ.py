import numpy as np
import json


class Environ (object):

	def __init__(self, config_file):
		super(Environ, self).__init__()

		# Dictionary containing data structures defining the environment
		# iincluding communication graph for computing reward.
		#  - P, M
		#  - Matrices (volume, num. messages and adjacency)
		#  - Capacity of the nodes in the platform
		#  - Reward function (object Reward)
		params = self.__read_config(config_file)
  
  
		# Creating state dictionary:
		self.env_state = dict()
  
		# 1. Communication graph parameters
		graph  = params["Graph"]
		self.M = graph["M"]
		self.P = graph["P"]
		self.env_state["M"] = self.M
		self.env_state["P"] = self.P

		# 2. Creating three PxP matrices:
		#  a) Volume matrix: total message volume communicated by pairs of procs.
		#  b) Num. msgs matrix: number of messages communicated by pair of procs.
		#  c) Adjacency matrix: simple adjacency matrix

		edges  = graph["comms"]["edges"]
		volume = graph["comms"]["volume"]
		n_msgs = graph["comms"]["n_msgs"]

		# a) Volume matrix
		volume_matrix = np.zeros((self.P, self.P))
		for m, edge in zip(volume, edges):
			volume_matrix[edge[0]][edge[1]] = m
		self.env_state["volume_matrix"] = volume_matrix

		# b) Num. messages matrix
		msgs_matrix = np.zeros((self.P, self.P))
		for n, edge in zip(n_msgs, edges):
			msgs_matrix[edge[0]][edge[1]] = n
		self.env_state["msgs_matrix"] = msgs_matrix

		# c) Adjacency matrix
		adjacency_matrix = np.zeros((self.P, self.P))
		for edge in edges:
			adjacency_matrix[edge[0]][edge[1]] = 1
		self.env_state["adjacency_matrix"] = adjacency_matrix

		# 3. Node capacity:
		#  We assume a number of nodes with a specific capacity (number of
		#   processes that can be executed in each node):
		self.env_state["capacity"] = graph["capacity"]
  
		# 4. Reward type (TO BE COMPLETED)
		#  Different forms of computing reward signal
		rw_type = params["Config"]["reward_type"]

		if   rw_type == "tLop":
			from tLop.tLop_reward     import Reward

		elif rw_type == "mpi":
			from mpi.mpi_reward      import Reward

		elif rw_type == "num_msgs":
			from gym_hpc.envs.common_mapping.num_msgs.num_msgs_reward  import Reward

		elif rw_type == "volume":
			from gym_hpc.envs.common_mapping.volume.volume_reward      import Reward

		else:
			raise Exception("The reward function does not exist: ", rw_type)

		# Create Reward object (function)
		self.reward_fxn = Reward(self.env_state)
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


	def get_reward(self, state):
     
		rw, info = self.reward_fxn.get_reward(state)
  
		return rw, info


	###   PRIVATE Functions

	def __read_config (self, config_file = './graph.json'):
    		
		config = {}

		try:
			with open(config_file, 'r') as js:
				config = json.load(js)

		except EnvironmentError:
			print ('Error: file not found: ', config_file)

		return config

