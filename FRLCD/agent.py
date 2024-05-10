import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

# 配置日志格式和级别
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
# Define the Deep Q-Network
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Define the Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(reward, dtype=np.float32), np.array(next_state), np.array(
            done, dtype=np.uint8)

    def __len__(self):
        return len(self.buffer)


# Define the Agent
class Agent:

    def __init__(self, state_dim, action_dim, capacity, lr):
        self.device = torch.device("cpu")
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        self.buffer = ReplayBuffer(capacity)
        self.q_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.learn_step_counter = 0
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.clients_states = {}  # 存储不同客户端的状态
        self.clients_fit_cid = []  # 本轮参与训练的客户端id列表
        self.actions = {}  # 存储不同客户端的动作
        self.global_accuracy = 0
        self.EPSILON = 0.1  # greedy policy
        self.GAMMA = 0.9  # Discount factor
        self.buffer_minsize = 32

    def act(self, state):
        if random.random() > self.EPSILON:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_value = self.q_net(state)
            action = q_value.max(1)[1].item()
        else:
            action = random.randrange(0, self.action_dim)
        return action

    def update(self, batch_size, gamma):
        state, action, reward, next_state, done = self.buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        q_values = self.q_net(state).gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_net(next_state).max(1)[0]
        expected_q_values = reward + gamma * next_q_values * (1 - done)

        loss = F.mse_loss(q_values, expected_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def train(self, reward):  # 保存状态元组到回放缓冲区 并 进行训练
        for cid in self.clients_fit_cid:  # 保存状态元组到回放缓冲区
            if len(self.clients_states[cid]) >= 2:
                self.buffer.push(self.clients_states[cid][-2], self.actions[cid][-2], reward,
                                 self.clients_states[cid][-1], 0)
                logging.info(f"保存状态序列:{(self.clients_states[cid][-2], self.actions[cid][-2], reward, self.clients_states[cid][-1], 0)}")
        self.clients_fit_cid = []  # 更新下一轮的状态序列
        if self.buffer.__len__() >= self.buffer_minsize:
            self.update(32, self.GAMMA)
