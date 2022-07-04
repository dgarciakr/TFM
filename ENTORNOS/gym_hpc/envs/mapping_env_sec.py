
import gym
import json
from gym import spaces

import numpy as np
from gym_hpc.envs.common_mapping.environ    import Environ

import os

class MappingEnvSec (gym.Env):
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, config_file):
        print('Mapping Env init (sequential): ', config_file)
        
        # Code for supporting reward and others
        self.env = Environ(config_file)
        
        self.M = M = self.env.M
        self.P = P = self.env.P
        
        # 1. Action space
        self.action_space = spaces.Discrete(M)
        
        # 2. Observation space
        state = {
            "Mapping":  spaces.Box(low=-1, high=M-1, shape=(P,), dtype=int),
            "Mask":     spaces.Box(low= 0, high=M,   shape=(M,), dtype=int)
        }
        self.observation_space = spaces.Dict(state)
                      
        # 3. Reward range
        self.reward_range = (-100, 0)
        
        # actual data
        self.capacity = np.array(self.env.env_state["capacity"], copy=True)
        self.obs      = None
        self.mask     = None

        # Starting point        
        self.t = 0        
        self.n_episode = 0
                
        self.verbose = self.env.verbose


        self.reward = None
        """
        with open(config_file, "r") as f:
            data = json.load(f)
            try:
                self.output_f = open(data["Output"]["output_file"], "w")
            except:
                print("ERROR")
        """

        return 
        
    
    def __print_msg(self, *args):
        
        if self.verbose == True:
            for arg in args:
                print(args, end='')
            print(' ')
            
                
    def step (self, action):
        
        # Action is the numbe of node assigned to process P_t
        m = action
        p = self.t
        self.__print_msg('[MappingEnv::step] ', 'P_', p, ' to ', m)
        
        # Update current mapping
        self.obs[p] = m

        # Update current capacity and mask
        self.capacity[m] -= 1
        #self.mask = np.where(self.capacity > 0, 1, 0)
        self.mask[action] -= 1
        
        # Reward is returned only at the end of the episode
        self.t = self.t + 1
        reward = 0.0
        done   = False
        info   = {}
        if self.t == self.P:            
            reward, info = self.env.get_reward(self.obs)
            done = True
        else:
            for i in range(len(self.mask)):
                if self.mask[i] < 0:
                    reward += self.mask[i]
            #self.output_f.write("Episode: " + str(self.n_episode) + "\n" + "State: " + str(self.obs) + "\n" + "Reward: " + str(reward) + "\n" + "Reward values: \n\t- n_inter: " + str(info["n_inter"]) + "\n\t- oversubs: " + str(info["oversubs"])+ "\n")
                            
        return {"Mapping": self.obs, "Mask": self.mask}, reward, done, info
    
    
    def reset (self):
        self.__print_msg('[MappingEnv::reset] ')
        
        # Observation is NO-VALID node (-1) to each process
        self.obs  = -np.ones(self.P, dtype=int)        
        self.mask =  np.full(self.M, self.M)
        
        # Step 0 in the next episode
        self.t = 0        
        self.n_episode += 1

        self.__print_msg("Observation Init: ", self.obs)
        self.__print_msg("Capacity: ", self.capacity)
        self.__print_msg("Mask: ", self.mask)
                        
        return {"Mapping": self.obs, "Mask": self.mask}
    
    
    def render (self, mode='human'):
        # TODO: no se como funciona esto por ahora
        end_ep = False
        if (self.t == self.P) and not (self.n_episode % 1000):
            print("Episode: ", self.n_episode, " State: ", self.obs)
            end_ep = True
            
        return end_ep
    
    
    def close (self):
        # TODO: no se que hace esta fxn por ahora
        self.__print_msg('Mapping Env close')