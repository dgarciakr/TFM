#!/usr/bin/env python

## Parametrized Policy network.

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



# Policy Network

# Encoder of the seq2seq policy Network
class Encoder(nn.Module):

	def __init__(self, params, num_inputs, num_hidden):
		super(Encoder, self).__init__()

		self.policy_params = params["Policy"]
		self.typecell = self.policy_params["typecell"]

		self.num_inputs = num_inputs
		self.num_hidden = num_hidden
		self.cell = nn.LSTM(input_size = self.num_inputs, hidden_size = self.num_hidden, batch_first = True)

		print("[seq2seq] Encoder init: ")
		print("#inputs:  ", self.num_inputs)
		print("#hidden:  ", self.num_hidden)
		print(self.cell)

		return

	def forward(self, state):
		output, (hidden, cell) = self.cell(state)
		return  hidden, cell


# Decoder of the seq2seq policy Network
class Decoder(nn.Module):

	def __init__(self,params, num_hidden, num_outputs):

		super(Decoder, self).__init__()

		self.params = params["Graph"]
		self.P = self.params["P"]

		self.num_hidden = num_hidden
		self.num_outputs = num_outputs

		self.cell = nn.LSTM(input_size = self.num_outputs, hidden_size = self.num_hidden, batch_first = True)

		self.fc = nn.Linear(self.num_hidden, self.num_hidden * 2)
		self.fc2 = nn.Linear(self.num_hidden * 2, self.num_outputs)

		self.dropout = nn.Dropout(p=0.4)
		self.relu    = nn.ReLU()
		self.softmax = nn.Softmax(dim = 2)

		print("[seq2seq] Decoder init: ")
		print("#hidden:  ", self.num_hidden)
		print("#outputs: ", self.num_outputs)
		print(self.cell)
		print(self.fc)


	def forward(self, inputs, hidden, cell):
		#Para hacer el embedding es necesario, pasar el tensor de float a long

		# print("[Decoder] state:  ", inputs.size())
		# print("[Decoder] hidden: ", hidden.size())
		# print("[Decoder] cell:   ", cell.size())

		output, (hidden, cell) = self.cell(inputs, (hidden, cell))

		# print("[Decoder] hidden: ", hidden.size())
		# print("[Decoder] output: ", output.size())

		logits = self.fc(output)

		logits = self.dropout(logits)
		logits = self.relu(logits)
		logits = self.fc2(logits)

		# logits = self.softmax(logits)

		# print("[Decoder] logits: ", logits.size())

		return  logits, hidden, cell


# Seq2seq new policiy network
class PolicyNetwork(nn.Module):

	def __init__(self, params):

		super(PolicyNetwork, self).__init__()

		self.params = params["Graph"]
		self.P = self.params["P"]
		self.M = self.params["M"]

		self.policy_params = params["Policy"]

		# Num inputs is set to the dimmesions of the embedding vectors
		self.num_inputs  = params["GNN"]["dimensions"]
		self.num_outputs = self.policy_params["n_outputs"]
		self.num_hidden  = self.policy_params["n_hidden"]

		print("[seq2seq] PolicyNetwork init: ")
		print("#inputs (embeddings dim):  ", self.num_inputs)
		print("#hidden:  ", self.num_hidden)
		print("#outputs: ", self.num_outputs)


		# Optimization on GPUs
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		self.encoder = Encoder( params,
								self.num_inputs,
								self.num_hidden).to(self.device)

		self.decoder = Decoder( params,
								self.num_hidden,
								self.num_outputs).to(self.device)

		# assert encoder.num_hidden == decoder.num_hidden, \
		#     "Hidden dimensions of encoder and decoder must be equal!"



	def forward(self, state):

		# print("[Policy] state:  ", state.size())

		hidden, cell = self.encoder(state)

		# print("[Policy] encoder hidden:  ", hidden.size())
		# print("[Policy] encoder cell:    ", cell.size())

		output = torch.zeros((1, 1, self.num_outputs)).to(self.device)

		outputs = torch.zeros((1, self.P, self.num_outputs)).to(self.device)

		# print("[Policy] start:   ", start.size())
		# print("[Policy] outputs: ", outputs.size())

		for p in range(0, self.P):

			output, hidden, cell = self.decoder(output, hidden, cell)

			# print("[Policy] output:  ", output.size())
			# print("[Policy] hidden:  ", hidden.size())
			# print("[Policy] cell:    ", cell.size())

			# Avoiding to outperform capacity in nodes
			# print(output)
			outputs[:, p] = output[:, 0]

			# print("[Policy] outputs: ", outputs.size())
			# print(" ")

		return outputs
