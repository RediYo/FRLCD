# Author: Robert Guthrie
import collections
import warnings
from collections import OrderedDict

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym import spaces

# LSTM任务为Pytorch教程中单词词性预测

MEAN_REWARD_BOUND = 15
BATCH_SIZE = 32
REPLAY_SIZE = 2000
LEARNING_RATE = 1e-4
REPLAY_START_SIZE = 2000
TARGET_REPLACE_ITER = 100  # 目标网络更新频率
EPSILON = 0.1  # greedy policy
GAMMA = 0.99  # Discount factor
Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])
warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Define Agent model, MLP
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        # self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(5, 128),  # input 5dims state
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 10),  # output 10dims action
        )

    def forward(self, x):
        # x = self.flatten(x)
        x = x.to(torch.float32)
        logits = self.linear_relu_stack(x)  # 在深度学习中，logits就是最终的全连接层的输出，而非其本意
        return logits


# 经验池
class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions, dtype=np.intp), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)


class Agent:
    def __init__(self, exp_buffer):
        # self.env = env
        # self.state = torch.from_numpy(np.asarray((0, 0, 0, 0, 0))).float()
        self.pre_state = (0, 0, 0, 0, 0)
        self.state = (0, 0, 0, 0, 0)
        self.exp_buffer = exp_buffer
        self.pre_action = 5
        self.action = 5
        self.pre_loss = 0
        self.loss = 0
        self.pre_reward = 0
        self.reward = 0
        self.learn_step_counter = 0  # for target updating
        self.action_space = spaces.Discrete(10, start=1)  # 动作空间{0，1，2，3，4，5，6，7，8，9，10}
        self._reset()

    def _reset(self):
        self.state = (0, 0, 0, 0, 0)
        self.state = torch.from_numpy(np.asarray([0, 0, 0, 0, 0])).float()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=EPSILON, device="cpu"):

        # todo 利用权重选择模型选择动作
        if np.random.random() < epsilon:
            action = self.action_space.sample()  # 随机选择权重
        else:
            state_a = np.array(self.state, copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            print(q_vals_v)
            _, act_i = torch.max(q_vals_v, 0)
            action = act_i.item() + 1

        print("动作（选择权值）：", action)

        return action


# ----------------------- LSTM -----------------------------
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


training_data = [
    # Tags are: DET - determiner; NN - noun; V - verb
    # For example, the word "The" is a determiner
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
testing_data = [
    # Tags are: DET - determiner; NN - noun; V - verb
    # For example, the word "The" is a determiner
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
word_to_ix = {}
# For each words-list (sentence) and tags-list in each tuple of training_data
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:  # word has not been assigned an index yet
            word_to_ix[word] = len(word_to_ix)  # Assign each word with a unique index
print(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}  # Assign each tag with a unique index

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        print(f"lstm_out{lstm_out}")
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    print(f"pre_test: {tag_scores}")


def train(model, training_data, epochs):
    """Train the network on the training set."""
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    total_loss = 0
    for epoch in range(epochs):  # again, normally you would NOT do 300 epochs, it is toy raw_data
        for sentence, tags in training_data:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tag_to_ix)

            # Step 3. Run our forward pass.
            tag_scores = model(sentence_in)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(tag_scores, targets)
            total_loss += loss
            # print("loss:", loss)
            loss.backward()
            optimizer.step()

    return total_loss


# See what the scores are after training
def test(model, testing_data):
    """Validate the network on the entire test set."""
    loss_function = nn.NLLLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for sentence, tags in testing_data:  # 两条测试数据
            sentence_in = prepare_sequence(sentence, word_to_ix)  # 输入句子的索引值
            targets = prepare_sequence(tags, tag_to_ix)  # 目标tags的索引值
            tag_scores = model(sentence_in)
            loss += loss_function(tag_scores, targets)
            _, predicted = torch.max(tag_scores, 1)
            total += 1
            pred_y = predicted.numpy()
            label_y = targets.numpy()
            if (pred_y == label_y).all:
                correct += 1

            # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
            # for word i. The predicted tag is the maximum scoring tag.
            # Here, we can see the predicted sequence below is 0 1 2 0 1
            # since 0 is index of the maximum value of row 1,
            # 1 is the index of maximum value of row 2, etc.
            # Which is DET NOUN VERB DET NOUN, the correct sequence!
            print(f"test: {tag_scores}")
    accuracy = correct / total
    return loss, accuracy


net = DQN().to(DEVICE)
tgt_net = DQN().to(DEVICE)
buffer = ExperienceBuffer(REPLAY_SIZE)
agent = Agent(buffer)


class Client(fl.client.NumPyClient):

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        print("123")
        self.set_parameters(parameters)
        # todo 训练密接时间推断模型
        episodes = 300
        total_loss = train(model, training_data, episodes)
        # todo 更新权重选择模型参数
        # agent_parameters_str = config["parameters"]
        # # 转换回numpy.array
        # agent_parameters = agent_parameters_str.strip('[')
        # agent_parameters = agent_parameters.strip(']')
        # agent_parameters = agent_parameters.split("(?<=\\)),{1}")
        # agent_parameters_dict_str = config["parameters"]
        # print(agent_parameters_dict_str)
        # agent_parameters_dict = eval(agent_parameters_dict_str)
        # print(agent_parameters_dict)
        # agent_parameters = [val.cpu().numpy() for _, val in agent_parameters_dict.items()]
        # params_dict = zip(net.state_dict().keys(), parameters)
        # state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        # 载入保存的模型参数
        net.load_state_dict(torch.load("../server/state_dict_model.pth"))
        # 不启用 BatchNormalization 和 Dropout
        net.eval()
        print(f"agent_net: {net}")
        action = agent.play_step(net)
        print(f"client action {action}")
        # todo 重放缓冲区保存状态转换元组，当前的状态（探测损失【客户端】，全局模型精度【服务端】，本地数据量大小【客户端】，
        #  全局数据量大小【服务端】，本地训练轮次【客户端】）
        agent.action = action
        agent.pre_reward = config["reward"]
        global_accuracy = config["reward"] + 0.5
        global_train_data_size = 0
        agent.state = (float(total_loss), global_accuracy, len(training_data), global_train_data_size, episodes)
        print(f"client state {agent.state}")

        # todo 训练完成后将参数、权重选择值、当前状态上传到中心服务器云端，training_data可以作为下一次的数据量大小
        return self.get_parameters(config={"w": action}), len(training_data), {"w": action, "state": str(agent.state)}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(model, testing_data)
        # todo 重放缓冲区保存状态转换元组，当前的状态（探测损失【客户端】，全局模型精度【服务端】，本地数据量大小【客户端】，
        #  全局数据量大小【服务端】，本地训练轮次【客户端】），在此时获取到全局数据量大小
        global_train_data_size = config["global_train_data_size"]
        state = agent.state
        agent.state = (state[0], state[1], state[2], global_train_data_size, state[4])
        exp = Experience(agent.pre_state, agent.pre_action, agent.pre_reward, 0, agent.state)
        agent.exp_buffer.append(exp)
        agent.pre_action = agent.action
        agent.pre_state = agent.state

        return float(loss), len(testing_data), {"accuracy": float(accuracy)}


fl.client.start_numpy_client(server_address="localhost:8080", client=Client())
