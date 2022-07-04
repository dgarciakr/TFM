import numpy as np
import torch
import sys

sys.path.append('../../utils')

class Reward:

    def __init__(self):
        return


    def get_reward(self, state, actions):

        P = state["P"]
        K = state["K"]
        rows = torch.arange(K).detach()
        msgs_matrix = state["msgs_matrix"]

        msgs_matrix = msgs_matrix / np.max(msgs_matrix)

        n_inter = torch.zeros(K).detach()
        rewards = torch.zeros((K, P), dtype=torch.float)

        # Overload capacity (OVERSUBSCRIPTING)
        oversubs = torch.zeros(K).detach()
        # acum = torch.zeros(K).detach()

        capacity = torch.tensor(state["capacity"]).detach()
        capacity = capacity.repeat(K, 1)

        for p in range(0, P):

            M = actions[:, p]
            capacity[rows, M] -= 1

            c_rows = (capacity[rows, M] < 0) * 1
            oversubs = oversubs + c_rows

            # First oversubscripting is not added -> favors exploration
            #  The subtle issue is that changing one processes is far worst
            #  than the benefit of changing two.
            # rewards[:, p] = oversubs
            # oversubs += (c * acum)

        oversubs = oversubs / (oversubs.max() + np.finfo(float).eps)


        # INTER-COMMUNICATIONS
        # n_inter(Kx1) will contain the number of messages through the network
        #   for processes according to mapping in actions.
        for src in range(0, P):
            for dst in range(0, P):
                if msgs_matrix[src,dst] != 0: # Different node
                    n_msgs = msgs_matrix[src,dst]
                    n_inter += (actions[:, src].cpu() != actions[:,dst].cpu()) * n_msgs

        n_inter = n_inter / (n_inter.max() + np.finfo(float).eps)


        # Final rewards are the sum of inter-comminications and oversubscripting
        b = 0.5
        rewards[:, P-1] = np.sqrt((b * n_inter)**2 + ((1 - b) * oversubs)**2)
        # rewards[:, P-1] = oversubs

        valid = True
        info = {"valid": valid}

        return -rewards, info
