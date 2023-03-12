import collections
from logging import WARNING

import torch
import torch.optim as optim
from flwr.common.logger import log
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch.nn.functional as F
from flwr.common import Metrics
from typing import Callable, Union

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from gym import spaces
from torch import nn

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""

clients_states = {}
rewards = [0.0]
total_data_size = 0
Experience = collections.namedtuple('Experience', field_names=['state', 'reward', 'new_state'])


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
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


# 服务端进行联合DQN更新
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
            nn.Linear(128, 10)  # output 10dims action
        )

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)  # 在深度学习中，logits就是最终的全连接层的输出，而非其本意
        return logits


class Agent:
    def __init__(self):
        # self.env = env
        # self.state = torch.from_numpy(np.asarray((0, 0, 0, 0, 0))).float()
        self.state = (0, 0, 0, 0, 0)
        self.learn_step_counter = 0  # for target updating
        self.action_space = spaces.Discrete(10, start=1)  # 动作空间{0，1，2，3，4，5，6，7，8，9，10}
        self._reset()

    def _reset(self):
        self.state = (0, 0, 0, 0, 0)
        self.state = torch.from_numpy(np.asarray([0, 0, 0, 0, 0])).float()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=None, device=None):
        action = self.action_space.sample()
        # print("动作（选择权值）：", action)
        return action


# batch 为选择训练的所有边缘用户终端的状态和下一状态，reward 为全局奖励
def calc_loss(batch, reward, net, tgt_net, device):
    states, next_states = batch
    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    reward_v = torch.tensor(reward).to(device)

    state_action_values = net.forward(states_v).sum()  # 通过对评估网络输入状态x，前向传播获得动作的Q值，所有边缘用户终端的状态值函数之和

    next_state_values = tgt_net(next_states_v).max(1)[0].sum()  # 在所有动作对应的Q值中找到最大Q值，得到所有边缘用户终端的最大Q值和
    next_state_values = next_state_values.detach()

    expected_state_action_values = reward_v + next_state_values * GAMMA

    return nn.MSELoss()(state_action_values, expected_state_action_values)


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


BATCH_SIZE = 32
REPLAY_SIZE = 2000
LEARNING_RATE = 1e-4
REPLAY_START_SIZE = 2000
TARGET_REPLACE_ITER = 100  # 目标网络更新频率
EPSILON = 0.1  # greedy policy
GAMMA = 0.99  # Discount factor
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = DQN().to(DEVICE)
tgt_net = DQN().to(DEVICE)
buffer = ExperienceBuffer(REPLAY_SIZE)
agent = Agent()


def agent_train(episodes, batch, reward):

    # print(f"agent网络：{net}")
    # log(1, f"agent网络：{net}")

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    if agent.learn_step_counter % TARGET_REPLACE_ITER == 0:  # 一开始触发，然后每100步触发
        tgt_net.load_state_dict(net.state_dict())  # 将评估网络的参数赋给目标网络
    agent.learn_step_counter += 1  # 学习步数自加1

    optimizer.zero_grad()
    loss_t = calc_loss(batch, reward, net, tgt_net, device=DEVICE)
    loss_t.backward()
    optimizer.step()


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


class FedCustom(fl.server.strategy.Strategy):
    """Configurable FedAvg strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
            self,
            *,
            fraction_fit: float = 1.0,
            fraction_evaluate: float = 1.0,
            min_fit_clients: int = 2,
            min_evaluate_clients: int = 2,
            min_available_clients: int = 2,
            evaluate_fn: Optional[
                Callable[
                    [int, NDArrays, Dict[str, Scalar]],
                    Optional[Tuple[float, Dict[str, Scalar]]],
                ]
            ] = None,
            on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            accept_failures: bool = True,
            initial_parameters: Optional[Parameters] = None,
            fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
            evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:
        """Federated Averaging strategy.

        Implementation based on https://arxiv.org/abs/1602.05629

        Parameters
        ----------
        fraction_fit : float, optional
            Fraction of clients used during training. Defaults to 0.1.
        fraction_evaluate : float, optional
            Fraction of clients used during validation. Defaults to 0.1.
        min_fit_clients : int, optional
            Minimum number of clients used during training. Defaults to 2.
        min_evaluate_clients : int, optional
            Minimum number of clients used during validation. Defaults to 2.
        min_available_clients : int, optional
            Minimum number of total clients in the system. Defaults to 2.
        evaluate_fn : Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]]
            ]
        ]
            Optional function used for validation. Defaults to None.
        on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure training. Defaults to None.
        on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure validation. Defaults to None.
        accept_failures : bool, optional
            Whether or not accept rounds containing failures. Defaults to True.
        initial_parameters : Parameters, optional
            Initial global model parameters.
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        """
        super().__init__()

        if (
                min_fit_clients > min_available_clients
                or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn

    def __repr__(self) -> str:
        rep = f"FedAvg(accept_failures={self.accept_failures})"
        return rep

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available
        clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def initialize_parameters(
            self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def evaluate(
            self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        # todo 分发权重选择模型（合并到参数 parameters）
        # config["parameters"] = [val.cpu().numpy() for _, val in net.state_dict().items()]
        # config["parameters"] = repr(net.state_dict())
        # parameters.tensors.extend([val.cpu().numpy() for _, val in net.state_dict().items()])
        # 建立路径并保存
        torch.save(net.state_dict(), "state_dict_model.pth")

        # 保存:可以是pth文件或者pt文件
        # print(parameters)
        # todo 返回全局奖励，用于边缘用户终端重放缓冲区保存状态转换元组
        config["reward"] = rewards[-1]
        print(f"奖励{config['reward']}")
        fit_ins = FitIns(parameters, config)
        # print(fit_ins)
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        config["global_train_data_size"] = total_data_size
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.metrics["w"])  # 根据边缘终端选择的权重进行加权
            for _, fit_res in results
        ]
        # todo 中心服务云端存储边缘用户终端的状态
        for client, fit_res in results:
            if clients_states.get(client.cid) is None:  # 没有存储过该客户端的状态值就创建一个新列表
                clients_states[client.cid] = []
            clients_states[client.cid].append(eval(fit_res.metrics["state"]))

        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        # todo 返回全局数据大小，为了重放缓冲区保存状态转换元组
        total_data_size = 0
        for _, res in results:
            total_data_size += res.num_examples

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        # todo 服务端进行联合DQN更新
        # if len(buffer) > REPLAY_START_SIZE:
        if len(clients_states) >= 2:
            batch = []
            total_accuracy = 0.0
            for client, res in results:  # 客户端状态批（用来训练联合Q函数）
                if len(clients_states[client.cid]) >= 2:
                    batch.append((clients_states[client.cid][-2], clients_states[client.cid][-1]))
            total_accuracy += res.metrics["accuracy"]
            reward = total_accuracy / len(results) - 0.5  # 全局奖励
            rewards.append(reward)
            if len(batch)==2:
                agent_train(episodes=1, batch=batch, reward=rewards[-2])

        return loss_aggregated, metrics_aggregated


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["w"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

# strategy = fl.server.strategy.FedAvg()

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=2),
    strategy=FedCustom(),
)
