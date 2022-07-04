# Gym HPC
--------------

## What is?

OpenAI Gym-like environment that exposes environments representing HPC problems to solve using Reinforcement Learning methods and algorithms.
We assume a HPC platform composed of nodes connected by communication channels. Every node is composed of one or more processing units, hence, several processes could be assigned to the same node. 
In general, the objective is to deploy processes, build algorithms or connfigure applications in such a way that the execution cost (both computation and communication costs) was optimal.

## Installation
1. Install [OpenAi Gym](https://github.com/openai/gym)
```bash
pip install gym
```

2. Download and install `gym-hpc`
```bash
git clone https://github.com/hpc-unex/gym-hpc
cd gym-hpc
pip install -e .
```

## Environments description
Start by importing the package and initializing the environment.
Currently, the following environments are available:

- Mapping-v0: environment exposing the current state as a vector of `P` processes. Each cell contains the node to which the process `P[i]` is assigned. As well, state contains a mask vector indicating what nodes are assigned with an enough number of processes to not produce oversubscripting.

```python
state = {
            "Mapping":  spaces.Box(low=-1, high=M-1, shape=(P,), dtype=int),
            "Mask":     spaces.Box(low= 0, high=1,   shape=(M,), dtype=int)
        }
```
with `P` the number of processes and `M` the number of nodes.

At each time step `t`, an action indicates the node assinged to the process`P[t]` in sequence (from process `0` to `P-1`). Hence, it is a single value:
```python
Discrete(M)
```

Reward is in the range `(-100,0)`, and its value depends on the method used to compute it. Please, see documentation for more details.

- Mapping-v1: environment containing the state as a vector of processes assigned to a node in a supercomputer. Initially, processes are assigned to nodes in Sequential mapping (process `p` to node p // (P//M)`, with `P` the number of processes and `M` the number of nodes). 

```python
spaces.Box(low=0, high=M-1, shape=(P,), dtype=int)
```

Actions are a tuple `(P_src, P_dst)` indicating a switch of the computational units assigned to the processes. The episode finishes when an enough switches have been completed (detault to `2 x P`) or source and destination processes are the same.

```python
spaces.MultiDiscrete ([P, P])
```

Reward is in the range `(-100,0)`, and its value depends on the method used to compute it. Please, see documentation for more details.

## Running
Just follow the OpenAI Gym procedure.

```python
import gym
import gym-hpc
env = gym.make('gym_hpc:Mapping-v0', config_file = "./graph.json")

for episode in range(n_episodes):
    obs = env.reset()
    agent.reset()

    done = False

    while not done:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        agent.learn(obs, reward)

    env.render()
. . .  
```

You should provide a .Json file at the moment of creating the environment. In such file the is a description of the communication graph of processes inn the application, number of processes, number of nodes and their capacities, etc. This is a minimal example graph file with `P=6` processes and `M=3` homogeneous nodes:

```json
{
	"Graph": {
		"P":                   6,
		"M":                   3,
		"node_names":          ["M0", "M1", "M2"],
		"capacity":            [2, 2, 2],
		"net":                 "IB",
		"comms": {
			"edges":             [[0, 2], [0, 5], [1, 3], [1, 4], [2, 0], [2, 3], [3, 1], [3, 2], [4, 1], [4, 5], [5, 0], [5, 4]],
			"volume":            [ 64,     256,    64,     256,    64,     256,    64,     256,    256,    64,     256,    64],
			"n_msgs":            [ 1,      8,      1,      8,      1,      8,      1,      8,      8,      1,      8,      1],
		}
	},

	"Config": {
		"reward_type":         "volume",
		"verbose":              false,
		"verbosity_interval":   500
	},

	"Hyperparameters": {
		"n_episodes":           100000,
		"gamma":                0.999,
		"alpha":                0.0001,
	},

}
```

Both according to the number of messages and also to the volume of data communicated, one optimal mapping will be `[0,1,2,2,1,0]`, that is, assignments of processes to nodes will be `M0={0,5}, M1={1,4}, M2={2,3}`.
