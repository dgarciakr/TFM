# MPIMapping

MPI Mapping algorithm based on Reinforcement Learning (RL).
The objective is to map MPI processes to nodes of a parallel computer 
optimizing the communications.

Target: data-parallel applications and MPI implementations.

Goal: implementing different RL algorithms (REINFORCE, A2C, PPO, etc.) and
evaluate effectiveness, suitability and performance.

## Executing algorithms:

$ python mapping.py <comm_graph>

For example:

$ python mapping.py ./graphs/graph.json


## Algorithms included:

### REINFORCE:
Implementation of the policy as a Recurrent Neural Network (Sequence to Sequence model), which takes the initial state (embedding of the processes in the commmunication graph of the application) and generates a sequence of P actions (node assigned to each process). Each action a_i is a probability distribution (over M) of the process P_i of being mapped to a node M_j.

### REINFORCE (2):
Implementation of the policy as a Recurrent Neural Netwrok (Sequence to Sequence model), which takes the
initial state (a Sequential mapping) and generates a sequence of action. Each action a_i is the probability distribution (over P) of switching the process P_i with a process P_j.
This algorithm design avoids the problem of MORL (Multi-Objective Reinforcement Learning), which requires large more episodes to converge and depends on two objectives: (1) avoid oversubscripting and (2) optimal communication cost.


## Visualizing results:

plot.py in ~/utils generate simple plots for results of execution.

"cd" to the folder where the output text file generated is, then

$ python <utils-folder>/plots.py <graph-file> <grapg-txt-1> [, <graph-txt-2> ...]

For example:
$ cd ../outputs
$ python ../MPIMapping/utils/plots.py graph.png graph.txt

## Libraries:

- Torch
- Numpy


## TODO:

* Environment (env) code:
    - Implement as an OpenAI Gym environment.
    - Complete and stabilize rewards (num_msgs, volume, tLop and mpi).
* Include new agents and algorithms in ~/Agent
    - REINFORCE with switch algorithm.
    - A2C (Advantage Actor Critic).
    - PPO (Proximal Policy Optimization).
* Misc:
    - Plot: running average fix.
    - Seq2Seq Neural Network with Attention.
    - Advanced neural network implementations (Transformers?).


--
Juan A. Rico (jarico@unex.es)
