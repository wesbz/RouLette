import numpy as np

import gym
from gym import spaces
from gym.utils import seeding


class RouletteEnv(gym.Env):
    """Roulette environment
    The roulette wheel has 37 spots.
    Here are the possible bets:
    Inside bets:
        - Single: Bet on one number. 37 possible bets (e.g. 0 to 36).
        - Split: Bet on 2 neighbor numbers. 57 possible bets (e.g. 1-4 or 2-3).
        - Street: Bet on an entire 3-numbers line. 12 possible bets (e.g. 7-8-9).
        - Corner: Bet on four neighbor numbers. 22 possible bets (e.g. 1-2-4-5).
        - Six Line: Bet on 6 consecutive numbers. 11 possible bets (e.g. 7-8-9-10-11-12).
        - Trio: Bet on 0-1-2 or 0-2-3.
        - First Four: Bet on 0-1-2-3.
    Outside bets:
        - 1 to 18 or 19 to 36.
        - Red or Black.
        - Even or Odd.
        - Dozen bet (1 to 12, 13 to 24 or 25 to 36).
        - Column bet (e.g. column 1 to 34).
    
    One action is the linear combination of each of these bets.
    One parameter for each possible bet: 154 total parameters.
    
    If you win, the reward is gains+1. If you lose, the reward is -bet-1.
    The last action stops the rollout for a return of 0 (walking away).
    """
    def __init__(self, spots=37):
        self.n = spots
        # self.action_space = spaces.MultiDiscrete([10000 for _ in range(151)])
        self.action_space = spaces.Box(low=0, high=1, shape=(152,))
        # self.action_space = spaces.Dict({
        #     "0":            spaces.Box(low=0, high=np.inf, shape=( 1, )),
        #     "single":       spaces.Box(low=0, high=np.inf, shape=(12,3)),
        #     "line_split":   spaces.Box(low=0, high=np.inf, shape=(12,2)),
        #     "column_split": spaces.Box(low=0, high=np.inf, shape=(11,3)),
        #     "street":       spaces.Box(low=0, high=np.inf, shape=(12, )),
        #     "corner":       spaces.Box(low=0, high=np.inf, shape=(11,2)),
        #     "six_line":     spaces.Box(low=0, high=np.inf, shape=(11, )),
        #     "half":         spaces.Box(low=0, high=np.inf, shape=( 2, )),
        #     "red_black":    spaces.Box(low=0, high=np.inf, shape=( 2, )),
        #     "even_odd":     spaces.Box(low=0, high=np.inf, shape=( 2, )),
        #     "dozen_bet":    spaces.Box(low=0, high=np.inf, shape=( 3, )),
        #     "column_bet":   spaces.Box(low=0, high=np.inf, shape=( 3, )),
        #     # "leave":        spaces.Discrete(2)
        # })
        self.gains = {
            "0":            35,
            "single":       35,
            "line_split":   17,
            "column_split": 17,
            "street":       11,
            "corner":        8,
            "six_line":      5,
            "half":          1,
            "red_black":     1,
            "even_odd":      1,
            "dozen_bet":     2,
            "column_bet":    2
        }
        self.red = set([1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36])
        self.observation_space = spaces.Discrete(spots+1)
        self.history = []
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # action = np.round(1000*action)
        assert self.action_space.contains(action)
        action_dict = {
            "0":            action[:1],
            "single":       action[1:37].reshape((12,3)),
            "line_split":   action[37:61].reshape((12,2)),
            "column_split": action[61:94].reshape((11,3)),
            "street":       action[94:106],
            "corner":       action[106:128].reshape((11,2)),
            "six_line":     action[128:139],
            "half":         action[139:141],
            "red_black":    action[141:143],
            "even_odd":     action[143:145],
            "dozen_bet":    action[145:148],
            "column_bet":   action[148:-1],
            "leave":        action[-1:]
        }
        
        if action_dict["leave"] == 1:
            return 38, -10, True, {"history": self.history+["leave"]}
        reward = 0
        # if action["leave"] == 1:
        #     # Leave the table
        #     # observation, reward, done, info
        #     return 0, reward, True, {"history": self.history}

        # N.B. np.random.randint draws from [A, B) while random.randint draws from [A,B]
        val = self.np_random.randint(0, self.n)
        # print(val)
        self.history.append(val)
        
        if val == 0:
            reward += action_dict["0"][0]*(self.gains["0"])  
        
        else:
            val_arr = np.zeros((12, 3))
            val_arr[(val-1)//3, (val-1)%3] = 1
            
            reward += np.sum(action_dict["single"]*self.gains["single"]*val_arr)
            
            reward += np.sum(action_dict["line_split"]*self.gains["line_split"]*val_arr[:,1:])
            reward += np.sum(action_dict["line_split"]*self.gains["line_split"]*val_arr[:,:-1])
            
            reward += np.sum(action_dict["column_split"]*self.gains["column_split"]*val_arr[1:,:])
            reward += np.sum(action_dict["column_split"]*self.gains["column_split"]*val_arr[:-1,:])
            
            reward += np.sum(np.tile(action_dict["street"], (3, 1)).T*self.gains["street"]*val_arr)
            
            reward += np.sum(action_dict["corner"]*self.gains["corner"]*val_arr[1:,1:])
            reward += np.sum(action_dict["corner"]*self.gains["corner"]*val_arr[1:,:-1])
            reward += np.sum(action_dict["corner"]*self.gains["corner"]*val_arr[:-1,1:])
            reward += np.sum(action_dict["corner"]*self.gains["corner"]*val_arr[:-1,:-1])
            
            reward += np.sum(np.tile(action_dict["six_line"], (3, 1)).T*self.gains["six_line"]*val_arr[1:,:])
            reward += np.sum(np.tile(action_dict["six_line"], (3, 1)).T*self.gains["six_line"]*val_arr[:-1,:])
            
            eye2 = np.eye(2)
            eye3 = np.eye(3)
            
            reward += np.sum(eye2[int(val > 18)]*action_dict["half"]*self.gains["half"])
            
            reward += np.sum(eye2[int(val not in self.red)]*action_dict["red_black"]*self.gains["red_black"])
            
            reward += np.sum(eye2[val % 2]*action_dict["even_odd"]*self.gains["even_odd"])
            
            reward += np.sum(eye3[(val-1)//12]*action_dict["dozen_bet"]*self.gains["dozen_bet"])
            
            reward += np.sum(eye3[(val-1)%3]*action_dict["column_bet"]*self.gains["column_bet"])
        
        for name, bets in action_dict.items():
            reward -= np.sum(bets)
        
        # observation, reward, done, info
        return val, reward, False, {"history": self.history}

    def reset(self):
        self.history = []
        self.seed()
        return 0