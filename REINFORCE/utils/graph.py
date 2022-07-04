import numpy  as np
import torch

import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout



def plot_graph(state, title):

    G = nx.from_numpy_matrix(state, create_using = nx.MultiDiGraph())

	# pos = nx.spring_layout(G)
    pos=nx.nx_pydot.graphviz_layout(G, prog='dot')
    nx.draw_networkx(G, pos, with_labels=True,
					 node_color='lightgray', node_size=500, node_shape='o',
					 arrowstyle="-|>", arrowsize=20, width=2)


    plt.title(title)
    plt.axis('off')
    plt.show()
    plt.savefig('graph.png')
    
	
def binomial_broadcast(params):
	order = params["order"]
	nodes = sum([2**k for k in range(order)])

	lista_com = []
	depth = [0]
	msgs = []
	for i in range(int(np.ceil(np.log2(nodes)))):
		n_com = 2**i-1
		for idnode in range(nodes+1): # [0,nodes+1)
			if idnode+2**i <= nodes:
				lista_com.append([idnode, idnode+2**i])
				depth.append(i+1)
				msgs.append(65536) #Tamaño de mensaje
			if n_com == 0: break
			else: n_com-= 1
	return lista_com, depth, msgs


def adjacency (P, comms, msg):

    adj = np.zeros((P, P), dtype=np.int)
    #for i, edge in enumerate(comms["edges"]):
    for i, edge in enumerate(comms):
        src = edge[0]
        dst = edge[1]
        adj[src, dst] = msg[i]
        adj[dst, src] = msg[i]

        # Symmetric

    return adj

