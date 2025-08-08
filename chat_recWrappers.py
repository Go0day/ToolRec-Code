import json
import pickle
import os
import gym
import numpy as np
import re
import string
from collections import Counter
from utils import *



class reActWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.user_idx = ''

    def reset(self, seed=None, return_info=False, options=None, idx=None):
        self.env.reset(seed=seed, return_info=return_info, options=options, uid=idx)
        try:
            self.env.step('')
        except:
            pass
        self.env.reset(seed=seed, return_info=return_info, options=options, uid=idx)
        self.user_idx = idx
        observation = f"{user_profile[self.user_idx]}"
        info = self._get_info()
        return (observation, info) if return_info else observation

    def _get_info(self):
        return {
        "user_id": self.user_idx,
        # "steps": self.steps, 
        # "answer": self.answer,
        # "question": user_profile[self.user_idx]
        }

    def get_reward(self, info):
        '''
        whether recommended item num is 10.
        '''
        if info['answer'] is not None:
            # label = self.data[self.data_idx][1]
            retrived_items = info['answer']
            pred = [item.split(', ')[0] for item in info['answer'].strip().split('\n')]
            if len(pred) == 10:
                return 1
        return 0

    def step(self, action):
        # first step obs does not have question. 
        obs, _, done, info = self.env.step(action)
        reward = self.get_reward(info)
        if done:
            obs = f"Episode finished, reward = {reward}\n"
            info.update({"user_idx": self.user_idx, 'reward': reward})
            # info.update({'em': reward, 'reward': reward, 'f1': reward})
        return obs, reward, done, info
    
    def __len__(self):
        return len(uid_iid)



class LoggingWrapper(gym.Wrapper):
    def __init__(self, env, folder="trajs", file_id=None):
        super().__init__(env)
        self.trajs = []
        self.traj = {"observations": [], "actions": []}
        self.folder = folder
        self.file_id = np.random.randint(0, 10000000) if file_id is None else file_id
        self.file_path = f"{self.folder}/{self.file_id}.json"
        os.makedirs("trajs", exist_ok=True)

    def __len__(self):
        return len(self.env.data)
  

    def reset(self, seed=None, return_info=False, options=None, userID=None):
        output = self.env.reset(seed=seed, return_info=return_info, options=options, idx=userID)
        observation = output[0] if return_info else output
        self.traj = {"observations": [observation], "actions": []}
        return output

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.traj["observations"].append(obs)
        self.traj["actions"].append(action)
        if done:
            self.traj.update(info)
        return obs, reward, done, info

    def update_record(self):
        if len(self.traj) > 0:
            self.trajs.append(self.traj)
            self.traj = {"observations": [], "actions": []}
  
    def write(self):
        self.update_record()
        with open(self.file_path, "w") as f:
            json.dump(self.trajs, f)
            print(f"Saved trajs to trajs/{self.file_id}.json")
    
    def close(self):
        self.write()
