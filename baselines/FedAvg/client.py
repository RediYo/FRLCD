import random
from collections import OrderedDict
from typing import Callable, Dict, Tuple

import flwr as fl
import torch
from flwr.common.typing import NDArrays, Scalar
from torch.utils.data import DataLoader

import model
from dataset_loader import load_datasets


class FlowerClient(fl.client.NumPyClient):
    """Standard Flower client for CNN training."""

    def __init__(
            self,
            net: torch.nn.Module,
            trainloader: DataLoader,
            valloader: DataLoader,
            device: torch.device,
            epochs_list: list,
            learning_rate: float,
    ):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        # self.device = device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.epochs_list = iter(epochs_list)
        self.learning_rate = learning_rate

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Returns the parameters of the current net."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Changes the parameters of the model using the given ones."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(
            self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Implements distributed fit function for a given client."""
        self.set_parameters(parameters)
        try:
            num_epochs = next(self.epochs_list)
            model.train(
                self.net,
                self.trainloader,
                self.device,
                epochs=num_epochs,
                learning_rate=self.learning_rate,
            )
        except StopIteration:
            print("迭代器已经到达末尾")

        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(
            self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """Implements distributed evaluation for a given client."""
        self.set_parameters(parameters)
        loss, accuracy = model.test(self.net, self.valloader, self.device)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def gen_client_fn(
        device: torch.device,
        num_clients: int,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
) -> Callable[[str], FlowerClient]:
    """Generates the client function that creates the Flower Clients.

    Parameters
    ----------
    device : torch.device
        The device on which the the client will train on and test on.
    iid : bool
        The way to partition the data for each client, i.e. whether the data
        should be independent and identically distributed between the clients
        or if the data should first be sorted by labels and distributed by chunks
        to each client (used to test the convergence in a worst case scenario)
    balance : bool
        Whether the dataset should contain an equal number of samples in each class,
        by default True
    num_clients : int
        The number of clients present in the setup
    num_epochs : int
        The number of local epochs each client should run the training for before
        sending it to the server.
    batch_size : int
        The size of the local batches each client trains on.
    learning_rate : float
        The learning rate for the SGD  optimizer of clients.
    partition: int

    Returns
    -------
    Tuple[Callable[[str], FlowerClient], DataLoader]
        A tuple containing the client function that creates Flower Clients and
        the DataLoader that will be used for testing
    """
    trainloaders, valloaders = load_datasets(num_clients=num_clients, batch_size=batch_size)

    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client representing a single organization."""

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load model
        INPUT_DIM = 2
        OUTPUT_DIMs = [1, 1, 1]  # 三个任务 # (0, 2.5)(2.5, 5.5)(5.5, 10)
        HIDDEN_DIM = 32

        net = model.AttentionLSTM(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIMs, 2, 1, 0).to(device)

        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]

        # 根据不同的客户端id设置随机数种子
        random.seed(int(cid))
        # 生成1000个不重复的一定范围内的随机整数
        epochs_list = []
        for i in range(1000):
            epochs_list.append(random.randint(num_epochs, 30))  # num_epochs为最小

        # Create a  single Flower client representing a single organization
        return FlowerClient(
            net, trainloader, valloader, device, epochs_list, learning_rate
        )

    return client_fn
