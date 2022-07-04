import os
import sys
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from config     					import read_config
from node2vec 						import Node2Vec
from sklearn.mixture 				import GaussianMixture
from networkx.drawing.nx_agraph 	import graphviz_layout


# Create a NetworkX graph with a set of automatically generated features, and
# optional features specified as a dictionary in the input .JSON file.
#
# Nodes features:
#  - Number of input edges.
#  - Vector of input ranks.
#  - Number of output edges.
#  - Vector of output ranks.
#  + Optional features specified as a dictionary in .JSON
# Edges features:
#  - Messages total size (volume).
#  - Number of messages (n_msgs).
#  + Optional features specified as a dictionary in .JSON


class Embeddings():

	def __init__(self, params):

		self.verbose = params["Config"]["verbose"]

		# Graph parameters from .JSON
		graph = params["Graph"]
		P       = graph["P"]
		M       = graph["M"]

		# Comms parameters from .JSON
		comms = params["Graph"]["comms"]
		edges           = comms["edges"]
		volume          = comms["volume"]
		n_msgs          = comms["n_msgs"]
		opt_nodes_feats = comms["opt_nodes_feats"]
		opt_edges_feats = comms["opt_edges_feats"]

		# Graph Neural Network hyperparameters:
		gnn = params["GNN"]
		dimensions  = gnn["dimensions"]
		n_walks     = gnn["n_walks"]
		walk_length = gnn["walk_length"]


		# Create an Undirected Graph
		self.G = nx.Graph()

		# Add nodes and edges together with generated features:
		self.G.add_nodes_from(np.arange(P))

		for p in range(P):
			self.G.nodes[p]["n_inputs"]  = 0
			self.G.nodes[p]["inputs"]    = []
			self.G.nodes[p]["n_outputs"] = 0
			self.G.nodes[p]["outputs"]   = []

		for edge, v, n in zip(edges, volume, n_msgs):

			src = edge[0]
			dst = edge[1]

			self.G.nodes[src]["n_outputs"] += 1
			self.G.nodes[src]["outputs"].append(dst)

			self.G.nodes[dst]["n_inputs"] += 1
			self.G.nodes[dst]["inputs"].append(src)

			self.G.add_edge(src, dst)
			self.G.edges[src, dst]["volume"] = v
			self.G.edges[src, dst]["n_msgs"] = n

		# Add optional node features:
		for k,v in opt_nodes_feats.items():
			for p in range(P):
				self.G.nodes[p][k] = v[p]

		# Add optional edge features:
		for k,v in opt_edges_feats.items():
			i = 0
			for edge in edges:
				src = edge[0]
				dst = edge[1]
				self.G.edges[src, dst][k] = v[i]
				i += 1

		# Generate model
		node2vec = Node2Vec (self.G,
							 dimensions  = dimensions,
							 walk_length = walk_length,
							 num_walks   = n_walks,
							 workers     = os.cpu_count(),
							 quiet       = True)

		model = node2vec.fit(window = 10, min_count = 1, batch_words = M)

		# Embeddings:
		self.embeddings = np.copy(model.wv.vectors)

		# Normalize embeddings (2021-10-26 -> TBT)
		m = self.embeddings.mean()
		s = self.embeddings.std()
		self.embeddings = (self.embeddings - m) / s

		if self.verbose:
			print("Graph nodes features: ")
			print(self.G.nodes.data())
			print("Graph edges features: ")
			print(self.G.edges.data())
			print("Graph embeddings: ")
			print(self.embeddings)


	def get_embeddings(self):

		return self.embeddings
