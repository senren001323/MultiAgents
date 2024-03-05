import torch
import numpy as np
import collections
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

class ReplayBuffer:
    """ReplyBuffer

    Buffer: (state, action, reward, next_state, done)
    """
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Return touples of states, actions, next_states, rewards, dones 
        """
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return state, action, reward, next_state, done

    def size(self):
        return len(self.buffer)


def stack_array(x):
    arrange = [[sub_x[i] for sub_x in x]
                  for i in range(len(x[0]))]
    return [
        torch.FloatTensor(np.vstack(a)).to(device)
        for a in arrange
    ]

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))