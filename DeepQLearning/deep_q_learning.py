import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


import gym
import matplotlib
import matplotlib.pyplot as plt
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim

from model import DQN
from replay_memory import ReplayMemory
from utils import select_action, optimize_model, plot_durations



env = gym.make('CartPole-v1')
plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




BATCH_SIZE = 128 # the number of transitions sampled from the replay buffer
GAMMA = 0.99 # the discount factor as mentioned in the previous section
EPS_START = 0.9 # the starting value of epsilon
EPS_END = 0.05 # the final value of epsilon
EPS_DECAY = 1000 # controls the rate of exponential decay of epsilon, higher means a slower decay
TAU = 0.005 # the update rate of the target network
LR = 1e-4 # the learning rate of the AdamW optimizer
SHOW_EVERY = 150

if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 50


# Get number of actions from gym action space
n_actions = env.action_space.n
state = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
criterion = nn.SmoothL1Loss()
memory = ReplayMemory(10000)


steps_done = 0
episode_durations = []


for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    for t in count():
        action = select_action(state, policy_net, EPS_END, EPS_START, EPS_DECAY,
                                 steps_done, env.action_space, device) 
        steps_done +=1

        observation, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        if done:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)


        if i_episode % SHOW_EVERY == 0:
            env.render()

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model(policy_net, target_net, BATCH_SIZE, GAMMA, memory, criterion, optimizer, device)

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            plot_durations(episode_durations)
            break

print('Complete')
plot_durations(episode_durations, show_result=True)
plt.ioff()
plt.show()