from logging import WARNING
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

from agent import Agent
from aggregate import aggregate, weighted_loss_avg, weighted_accuracy_avg
from flwr.server.strategy import Strategy

import torch
import torch.optim as optim
from flwr.common.logger import log
from typing import Dict, List, Optional, Tuple
import logging

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""

# 配置日志格式和级别
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)


# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FedAvg(Strategy):
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
            agent: Agent,
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
        self.agent = agent
        self.centralized_reward = 0.0  # 中心化评估奖励
        self.fed_reward = 0.0  # 联邦评估奖励
        self.probe_max = 0
        self.local_sizes = []
        self.local_num_epochs = []

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

        self.agent.global_accuracy = metrics["accuracy"]
        # 中心化评估方法奖励
        self.centralized_reward = metrics["accuracy"] - 0.5
        if server_round >= 3:
            self.agent.train(self.centralized_reward)
        logging.info(f"centralized_reward:{self.centralized_reward}")

        return loss, metrics

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)

        # 分发权重选择模型，方式为保存模型
        # agent q_net为客户端共享
        # 建立路径并保存
        torch.save(self.agent.q_net.state_dict(), "state_dict_model.pth")

        fit_ins = FitIns(parameters, config)

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
        # config["global_train_data_size"] = total_data_size
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
            (parameters_to_ndarrays(fit_res.parameters), fit_res.metrics["w"])
            for _, fit_res in results
        ]

        w_s = []  # 所有客户端参与权重列表
        for client, fit_res in results:
            if self.agent.actions.get(client.cid) is None:  # 没有存储过该客户端的状态值就创建一个新列表
                self.agent.actions[client.cid] = []
            self.agent.actions[client.cid].append(fit_res.metrics["w"])
            self.agent.clients_fit_cid.append(client.cid)  # 保存本轮训练cid
            w_s.append(fit_res.metrics["w"])

        # 根据权值聚合参数
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        # 中心服务云端更新并存储边缘用户终端的状态
        # 状态值处理：状态向量每个值进行区块划分以减小状态空间
        for client, fit_res in results:
            if self.agent.clients_states.get(client.cid) is None:  # 没有存储过该客户端的状态值就创建一个新列表
                self.agent.clients_states[client.cid] = []
            state = fit_res.metrics["state"]
            # state = (state[0], avg_accuracy, state[2], state[3])
            if server_round <= 3:
                # 探测损失自动调整 probe_loss
                if state[0] > self.probe_max:
                    self.probe_max = state[0]
                self.local_sizes.append(state[2])
                self.local_num_epochs.append(state[3])
            else:
                state = state_division(state, self.probe_max, self.local_sizes, self.local_num_epochs)
                self.agent.clients_states[client.cid].append(state)
            # print(f"clients_states:{self.agent.clients_states}")

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
        # Aggregate accuracy
        avg_accuracy = weighted_accuracy_avg(
            results.__len__(),
            [evaluate_res.metrics["accuracy"] for _, evaluate_res in results]
        )
        self.agent.global_accuracy = avg_accuracy

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        # 联邦评估方法 服务端进行联合DQN更新，增加重放缓冲区
        # 在此时获取到全局数据量大小和全局精度信息，state填入全局数据量大小和精度，服务器保存状态元组到回放缓冲区
        self.fed_reward = avg_accuracy - 0.5
        if server_round >= 3:
            self.agent.train(self.fed_reward)
        logging.info(f"fed_reward:{self.fed_reward}")

        return loss_aggregated, metrics_aggregated


def state_division(state, probe_max, local_sizes, local_num_epochs):  # 状态自动化分区方法

    # 状态(探测损失, 全局模型精度, 本地数据量大小, 本地训练轮次）

    # 探测损失 最大值二分法
    probe_loss_index = max_segment_index(probe_max, state[0])

    global_accuracy_index = int(state[1] * 20) + 1

    local_size_index = median_segment_index(local_sizes, state[2])

    num_epochs_index = median_segment_index(local_num_epochs, state[3])

    return probe_loss_index, global_accuracy_index, local_size_index, num_epochs_index


def max_segment_index(up, x):
    lst = [0]
    i = 1
    while (i <= up):
        lst.append(i)
        i = i * 2
    lst[-1] = up
    i = 1
    while (i < len(lst)):
        if (x <= lst[i]):  # 左开右闭区间
            return i
        i += 1
    return i  # 索引值从1开始


def median_segment_index(lst, x):
    # 对lst进行排序
    lst = sorted(lst)
    # print(lst)
    if (x > lst[-1]):
        return 0
    i = 1
    while (len(lst) >= 2):
        median_index = len(lst) // 2 - 1  # 如果是偶数则取中间前一位，奇数就是中间位置数
        if (x > lst[median_index]):
            return i  # 返回值索引 从小到大，但是区间范围从大到小
        i += 1
        lst = lst[:median_index + 1]

    return -1
