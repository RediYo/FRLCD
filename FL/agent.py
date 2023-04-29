import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque


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
class ReplayBuffer():
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        self.buffer = ReplayBuffer(capacity)
        self.q_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_value = self.q_net(state)
            action = q_value.max(1)[1].item()
        else:
            action = random.randrange(self.action_dim)
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

# agent = Agent(state_dim, action_dim, capacity, lr)
# for episode in range(num_episodes):
#     state = env.reset()
#     epsilon = max(epsilon_min, epsilon_decay**episode)
#     done = False
#     while not done:
#         action = agent.act(state, epsilon)
#         next_state, reward, done, _ = env.step(action)
#         agent.buffer.push(state, action, reward, next_state, done)
#         state = next_state
#         if len(agent.buffer) > batch_size:
#             agent.update(batch_size, gamma)
#     if episode % target_update == 0:
#         agent.update_target()
