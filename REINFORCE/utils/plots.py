import numpy  as np
import pandas as pd

import sys

import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

verbose = True   # Show some progress messages
show    = False  # Show plot (True) or save to file (False)


# Data from output file columns to plot in files. Available:
# J: cost function.
# Reward: reward obtained from environment.
# Discounted_RW: discounted reward.
# B_avg: baseline average (if available).
# Return: return after discounted reward and baseline are appied.
# logprob: logarithm of the probabilities of the actions.
plot_data = ["J", "Reward", "logprob"]


def plot_file (file_names, graph_file=None, show=False):

	list_df = []

	for f_name in file_names:

		try:
			df = pd.read_csv(f_name, index_col=0, delimiter="#", skiprows=22, usecols=[0,1,2,3,4,5,6,7,8,9], names=["n_episode", "J", "t", "T", "Reward", "Discounted_RW", "B_avg", "B_std", "Return", "logprob"])
			list_df.append(df)

		except:
			print("Error: output_file can not be open or read: ", f_name)
			print("$ plot <output PNG file> <input TXT file 1 [, input TXT file 2, ...]>")
			continue

	mdf = pd.concat(list_df, keys=file_names, axis=1)

	cols = len(file_names)

	X_AXIS = 100  # Hardcoded: 100 ticks
	start = 0
	end   = mdf.shape[0]
	end   = end - (end % X_AXIS)
	chunk = int(np.floor(end / X_AXIS))

	for num, plt_data in enumerate(plot_data):

		if verbose:
			print("Generating ", plt_data, " data ...")

		idx = pd.IndexSlice
		data = mdf.loc[:, idx[:, plt_data]]

		# Average data in bins
		data = data.to_numpy()
		data = data.reshape((X_AXIS, chunk, cols))
		d_mean = data.mean(axis=1)
		d_std  = data.std(axis=1)

		# Figure
		fig, ax = plt.subplots(figsize=(15,8))
		# plt.axis('on')

		# X_axis = np.linspace(1, X_AXIS, chunk)
		X_axis = np.linspace(1, X_AXIS, X_AXIS)

		ax.set_xlabel('# Episode')
		ax.set_ylabel(plt_data)

		# Ticks and labels in X axis (hardcoded)
		# xlab = np.arange(start+X_AXIS, end+1, X_AXIS)
		xlab = np.linspace(start+chunk, end, X_AXIS, dtype=np.int)
		ax.set_xticks(X_axis)
		ax.set_xticklabels(xlab)
		ax.xaxis.set_major_locator(plt.MaxNLocator(12))

		# Plot shadow standard deviation on plot (TBD: correct?):
		# Indeed, in order to reduce the plot space, I reduce arbitrary the std
		#   REMOVE!!!!
		REDUCTION = 10.0
		for c in range(cols):
			d_fill_dw = d_mean[:, c] - d_std[:, c]
			d_fill_up = d_mean[:, c] + d_std[:, c]
			ax.fill_between(np.arange(0, X_AXIS, 1), d_fill_dw, d_fill_up, alpha=0.1)

		# Plot some helping stuff:
		ax.hlines(0, 0, X_AXIS, colors='r', linestyles='dashed', linewidth=0.5)

		# Plot:
		ax.plot(X_axis, d_mean)
		ax.legend(labels=file_names)
		graph_file_name = graph_file.replace(".png", "_"+str(num)+"_"+plt_data+".png")

		if show:
			plt.show()
		else:
			plt.savefig(graph_file_name, dpi=300)

	return



##########   MAIN   ##########

if len(sys.argv) < 3:
	print("ERROR: plot <outpu PNG file> <input TXT file1> [... <input TXT fileN>")
	sys.exit(1)

output_graph = sys.argv[1]
input_files  = sys.argv[2:]

plot_file (input_files, graph_file=output_graph)
