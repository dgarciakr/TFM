import gym
import json
from gym import spaces

import numpy as np
from gym_hpc.envs.common_mapping.environ    import Environ


class MappingEnvSwitch (gym.Env):
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, config_file):
        
        print('Mapping Env init (Switch): ', config_file)
        # Code for supporting reward and others
        self.env = Environ(config_file)
        
        self.M = M = self.env.M
        self.P = P = self.env.P
        
        # 1. Action space
        self.action_space = spaces.MultiDiscrete ([P, P])
        
        # 2. Observation space
        self.observation_space = spaces.Box(low=0, high=M-1, shape=(P,), dtype=int)

        # 3. Reward range              
        self.reward_range = (-100, 0)
        
        # Actual data
        self.capacity = np.array(self.env.env_state["capacity"], copy=True)        
        self.obs      = None
        
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
        
    
    def __print_msg(self, *args):
        
        if self.verbose == True:
            for arg in args:
                print(args, end='')
            print(' ')
            
            
    def step (self, action):
        
        # Action is composed of a pair of processes to switch
        (p_src, p_dst) = action
        
        self.__print_msg('[MappingEnv::step] ', 'P_', p_src, ' <-> ', p_dst)
        
        # Swap processes
        self.obs[p_src], self.obs[p_dst] = self.obs[p_dst], self.obs[p_src]

        # Episode finishes when both processes are the same or
        #  number of iterations is enough
        self.t = self.t + 1
        done = False
        self.reward, info = self.env.get_reward(self.obs)
        if (p_src == p_dst) or (self.t >= self.P * 2):
            done = True
        #self.output_f.write("\nEpisode: " + str(self.n_episode) + " \nState: " + str(self.obs) + "\nReward: " + str(self.reward))
        
        return self.obs, self.reward, done, info
    
    
    def reset (self):
        self.__print_msg('[MappingEnv::reset] ')

        # Initial observation is the Sequential mapping (heterogeneous support)
        self.obs = np.repeat(np.arange(self.M), self.capacity)
        #self.obs = np.zeros(self.P, dtype=int)
        #self.obs[1] = 1
        #self.obs[3] = 1

        # Step 0 in the next episode
        self.t = 0        
        self.n_episode += 1
        
        return self.obs
    
    
    def render (self, mode='human'):
        # TODO: no se como funciona esto por ahora
        end_ep = False
        if not (self.n_episode % 1000):
            print("Episode: ", self.n_episode, " (", self.t, ")", " State: ", self.obs)
            end_ep = True

        return end_ep
        
    
    def close (self):
        # TODO: no se como funciona esto por ahora
        self.__print_msg('Mapping Env close')