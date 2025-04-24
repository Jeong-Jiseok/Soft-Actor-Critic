import os
import pickle
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.index = 0
        self.size = 0

        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)


    def add(self, state, action, reward, next_state, done):
        self.state[self.index] = state
        self.action[self.index] = action
        self.reward[self.index] = reward
        self.next_state[self.index] = next_state
        self.done[self.index] = float(done)

        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)


    def sample(self, batch_size):
        indexes = np.random.randint(0, self.size, size=batch_size)
        
        return (torch.tensor(self.state[indexes]), 
                torch.tensor(self.action[indexes]), 
                torch.tensor(self.reward[indexes]), 
                torch.tensor(self.next_state[indexes]), 
                torch.tensor(self.done[indexes]),
                )
    

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {"state": self.state, 
                "action": self.action, 
                "reward": self.reward, 
                "next_state": self.next_state, 
                "done": self.done, 
                "index": self.index, 
                "size": self.size, 
                "capacity": self.capacity
                }
        
        with open(path, "wb") as f:
            pickle.dump(data, f)


    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.state = data["state"]
        self.action = data["action"]
        self.reward = data["reward"]
        self.next_state = data["next_state"]
        self.done = data["done"]
        self.index = data["index"]
        self.size = data["size"]
        self.capacity = data["capacity"]


    def __len__(self):
        return self.size