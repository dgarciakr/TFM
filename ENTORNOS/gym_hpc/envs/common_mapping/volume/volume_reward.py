import numpy as np
import torch
import sys

sys.path.append('../../utils')

class Reward:

    def __init__(self, env_state):
        
        self.P = env_state["P"]
        self.M = env_state["M"]
        
        # Normalize message matrix
        self.volume_matrix = env_state["volume_matrix"]
        self.volume_matrix = self.volume_matrix / np.max(self.volume_matrix)  
        
        self.capacity = np.array(env_state["capacity"], copy=True)

        return


    # Reward base on the volume of messages communicated between two 
    #  processes (using slow communicating channels) and oversubscripting 
    #  (more processes in a node than computing units).
    # Return a weighted sum of both values.     
    def get_reward(self, state):

        capacity = np.copy(self.capacity) 

        # 1. Overload capacity (oversubscripting)
        #    Vectorized:                      
        capacity = np.bincount(state, minlength=capacity.size)
        capacity = np.subtract(capacity, self.capacity)
        oversubs = np.where(capacity >= 0, capacity, 0).sum()
        """ Computation: 
        capacity = np.copy(self.capacity) 
        for m in range(0, self.M):
            M = (state == m) * 1            
            capacity[m] = max(M.sum() - capacity[m], 0)
            
        oversubs = capacity.sum()
        """        
        # 2. Nummber of inter-communications
        #  TODO: vectorize indeed next computation

        n_inter = 0.0

        for src in range(0, self.P):
            tmp_state = np.zeros(self.P) 
            m = state[src]
            np.putmask(tmp_state, state != m, 1)
            n_inter += np.dot(tmp_state, self.volume_matrix[src])
            
        """ Computation: 
        for src in range(0, self.P):
            for dst in range(0, self.P):
                if self.msgs_matrix[src,dst] != 0: # Different node
                    n_msgs = self.volume_matrix[src,dst]
                    n_inter += (state[src] != state[dst]) * n_msgs        
        """

        # Return: Final reward is the sum of inter-comminications and oversubscripting
        b = 0.5
        #reward = np.sqrt((b * n_inter)**2 + ((1 - b) * oversubs)**2)
        reward = np.sqrt(n_inter**2 + (self.P * (oversubs + 1))**2)

        info = {"Oversubscripting_rw":    oversubs,
                "intercommunication_rw":  n_inter,
                "reward":                -reward,
                "n_inter":                  n_inter,
                "oversubs":                 oversubs,
                "valid_oversubs":         (oversubs == 0) }
        
        return -reward, info

