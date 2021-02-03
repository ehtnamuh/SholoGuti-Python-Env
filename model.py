import datetime
import random
import time
from itertools import count

import numpy as np
import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, lr, lamda, seed=123):
        super(BaseModel, self).__init__()
        self.lr = lr
        self.lamda = lamda  # trace-decay parameter
        self.start_episode = 0

        self.eligibility_traces = None
        self.optimizer = None

        torch.manual_seed(seed)
        random.seed(seed)

    def update_weights(self, p, p_next):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def init_weights(self):
        raise NotImplementedError

    def init_eligibility_traces(self):
        self.eligibility_traces = [torch.zeros(weights.shape, requires_grad=False) for weights in
                                   list(self.parameters())]

    def checkpoint(self, checkpoint_path, step, name_experiment):
        path = checkpoint_path + "/{}_{}_{}.tar".format(name_experiment,
                                                        datetime.datetime.now().strftime('%Y%m%d_%H%M_%S_%f'), step + 1)
        torch.save({'step': step + 1, 'model_state_dict': self.state_dict(),
                    'eligibility': self.eligibility_traces if self.eligibility_traces else []}, path)
        print("\nCheckpoint saved: {}".format(path))

    def load(self, checkpoint_path, optimizer=None, eligibility_traces=None):
        checkpoint = torch.load(checkpoint_path)
        self.start_episode = checkpoint['step']

        self.load_state_dict(checkpoint['model_state_dict'])

        if eligibility_traces is not None:
            self.eligibility_traces = checkpoint['eligibility']

        if optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

    def train_agent(self, env, n_episodes, save_path=None, eligibility=False, save_step=0, name_experiment=''):
        raise NotImplementedError
