import numpy as np
import torch
import sys

sys.path.append('../../utils')

class Reward:

    def __init__(self):

        self.length  = 500
        self.episode = 0

        self.window = torch.zeros(self.length)
        self.avg    = 0.0
        self.stddev = 0.0

        return


    def get_reward(self, state, actions):

        vol_inter  = 0

        actions = torch.squeeze(actions, dim=0)
        P = state["P"]
        volume_matrix = state["volume_matrix"]

        rewards = torch.zeros(P, dtype=torch.float)

        # Volume of intercommunication (communications through the network)
        total_inter = 0
        for p in range(0, P):
            M_i = actions[p]
            inter = np.logical_not(actions == M_i) # Processes not in p node
            v_inter = np.multiply(inter, volume_matrix[p])
            total_inter += v_inter.sum()

        r = -total_inter.item()  # r is expected to be a number (not tensor)

        self.window[self.episode % self.length] = r
        self.episode += 1

        self.avg    = self.window.mean()
        self.stddev = self.window.std()
        rw = (r - self.avg)
        if self.stddev != 0:
            rw /= self.stddev

        rewards[P-1] = rw

        valid = True
        info = {"valid": valid, "R": r, "R-b": rw, "Baseline": [self.avg, self.stddev]}

        return rewards, info
