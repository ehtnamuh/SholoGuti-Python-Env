from typing import Dict, List

import numpy as np
import torch
from model import NN
import experience
import SholoGutiUnityWrapper


class Trainer:
  # Hyperparameters
  input_size = 37
  output_size = 1
  encoding_size = 64
  # num_layers = 2

  learning_rate = 0.001
  batch_size = 64
  num_epochs = 1

  #eperience Buffer
  buffer: experience.Buffer = []

  # Create a Mapping from AgentId to Trajectories. This will help us create
  # trajectories for each Agents
  dict_trajectories_from_agent: Dict[int, experience.Trajectory] = {}
  # Create a Mapping from AgentId to the last observation of the Agent
  dict_last_obs_from_agent: Dict[int, np.ndarray] = {}
  # Create a Mapping from AgentId to the last observation of the Agent
  dict_last_action_from_agent: Dict[int, np.ndarray] = {}
  # Create a Mapping from AgentId to cumulative reward (Only for reporting)
  dict_cumulative_reward_from_agent: Dict[int, float] = {}
  # Create a list to store the cumulative rewards obtained so far
  cumulative_rewards: List[float] = []


  # Set device
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # set network
  model = NN(input_size=input_size).to(device)

  loss_function = torch.nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  def generate_trajectories(self):
    raise NotImplemented

  def train_network(self):
    raise NotImplemented