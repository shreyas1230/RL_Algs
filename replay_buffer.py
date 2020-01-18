from collections import namedtuple
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):

    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample_random(self, batch_size):
        return random.sample(self.memory, batch_size)

    def sample_recent(self, batch_size):
        if self.position < batch_size:
            return self.memory[-(batch_size-self.position):] + self.memory[:self.position]
        else:
            return self.memory[self.position-batch_size:self.position]

    def __len__(self):
        return len(self.memory)
