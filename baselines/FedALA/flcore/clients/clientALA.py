import copy
import random
from typing import Tuple

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
# from utils.data_utils import read_client_data
from utils.ALA import ALA

# from utils.data_utils import read_client_data_partition


class clientALA(object):
    def __init__(self, args, id, train_samples, test_samples, train_loader, test_loader):
        self.model = copy.deepcopy(args.model)
        self.dataset = args.dataset
        self.device = args.device
        self.id = id

        self.num_classes = 10
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = 32
        self.learning_rate = args.local_learning_rate
        self.local_steps = args.local_steps

        self.loss = nn.L1Loss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        self.eta = args.eta
        self.rand_percent = args.rand_percent
        self.layer_idx = args.layer_idx
        # self.train_loader = read_client_data_partition(self.dataset, self.id, is_train=True)
        # self.test_loader = read_client_data_partition(self.dataset, self.id, is_train=False)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.ALA = ALA(self.id, self.loss, self.train_loader, self.batch_size,
                       self.rand_percent, self.layer_idx, self.eta, self.device)

        # 测试精度控制
        self.lim_diff = [0.1, 0.2]

    def train(self):
        trainloader = self.train_loader
        self.model.train()

        for step in range(self.local_steps):
            for i, (x, y) in enumerate(trainloader):
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                # print(f"loss:{loss}")
                loss.backward()
                self.optimizer.step()

    def local_initialization(self, received_global_model):
        self.ALA.adaptive_local_aggregation(received_global_model, self.model)

    # def load_train_data(self, batch_size=None):
    #     if batch_size == None:
    #         batch_size = self.batch_size
    #     train_data = read_client_data_partition(self.dataset, self.id, is_train=True)
    #     return train_data

    # def load_test_data(self, batch_size=None):
    #     if batch_size == None:
    #         batch_size = self.batch_size
    #     test_data = read_client_data_partition(self.dataset, self.id, is_train=False)
    #     return test_data

    def test_metrics(self, model=None):
        testloader = self.test_loader
        if model == None:
            model = self.model
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        model.eval()
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        if len(testloader.dataset) == 0:
            raise ValueError("Testloader can't be 0, exiting...")
        loss /= len(testloader.dataset)
        accuracy = correct / total
        return loss, accuracy

    def train_metrics(self, model=None):
        trainloader = self.train_loader
        if model == None:
            model = self.model
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        model.eval()
        with torch.no_grad():
            for images, labels in trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        if len(trainloader.dataset) == 0:
            raise ValueError("Testloader can't be 0, exiting...")
        loss /= len(trainloader.dataset)
        accuracy = correct / total
        return loss, accuracy

# --------------------- BLE ----------------

    def train(
        self,
        # net: nn.Module,
        # trainloader: DataLoader,
        # device: torch.device,
        epochs: int,
        # learning_rate: float,
    ) -> None:
        """Train the network on the training set.

        Parameters
        ----------
        net : nn.Module
            The neural network to train.
        trainloader : DataLoader
            The DataLoader containing the data to train the network on.
        device : torch.device
            The device on which the model should be trained, either 'cpu' or 'cuda'.
        epochs : int
            The number of epochs the model should be trained for.
        learning_rate : float
            The learning rate for the SGD optimizer.
        """
        net = self.model
        trainloader = self.train_loader
        criterion = torch.nn.L1Loss()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
        net.train()
        total_loss = 0
        for epoch in range(1, epochs + 1):
            loss_epoch = 0
            for features, labels in trainloader:
                net.zero_grad()
                outputs = net(features)
                # 将所有张量在第1个维度上拼接起来
                outputs_concatenated = torch.cat(outputs, dim=1)
                loss = [criterion(output, label) for output, label in zip(outputs_concatenated, labels)]
                losses = sum(loss)
                # logging.info(f"Epoch {epoch}, loss: {loss}")
                loss_epoch += losses
                losses.backward()
                optimizer.step()
            total_loss += loss_epoch

    def test(
            self,
            # net: nn.Module,
            # testloader: DataLoader,
            # device: torch.device
    ) -> Tuple[float, float]:
        """Evaluate the network on the entire test set.

        Parameters
        ----------
        net : nn.Module
            The neural network to test.
        testloader : DataLoader
            The DataLoader containing the data to test the network on.
        device : torch.device
            The device on which the model should be tested, either 'cpu' or 'cuda'.

        Returns
        -------
        Tuple[float, float]
            The loss and the accuracy of the input model on the given data.
        """
        net = self.model
        testloader = self.test_loader
        criterion = torch.nn.L1Loss()
        net.eval()
        lim_diff = self.lim_diff
        correct, total, losses, total_loss = 0, 0, 0.0, 0.0
        # writer = SummaryWriter('./data_log/test')
        with torch.no_grad():
            for features, labels in testloader:
                # features = torch.tensor(data_items[0])
                # labels = [torch.tensor([x]) for x in data_items[1]]
                outputs = net(features)
                outputs = torch.cat(outputs, dim=1)
                # print(f"outputs:{outputs}")
                # print(f"labels:{labels}")
                losses = [criterion(output, label) for output, label in zip(outputs, labels)]
                total_loss += sum(losses)
                predicteds = outputs
                total += 1
                pred_y = [tensor.numpy() for tensor in predicteds]
                label_y = [tensor.numpy() for tensor in labels]
                diff = [abs(x - y) for x, y in zip(pred_y, label_y)]
                # print(f"diff:{diff}")
                if all(d <= l for d, l in zip(diff[0], lim_diff)):  # 输出概率差值全部小于lim_diff则认为是预测正确，因为是批处理所以修改为diff[0]
                    correct += 1

        accuracy = correct / total
        avg_loss = total_loss / total
        # print(f"accuracy: {accuracy}")

        return avg_loss, accuracy

    def train_metrics(
            self,
            # net: nn.Module,
            # testloader: DataLoader,
            # device: torch.device
    ) -> Tuple[float, float]:
        """Evaluate the network on the entire test set.

        Parameters
        ----------
        net : nn.Module
            The neural network to test.
        testloader : DataLoader
            The DataLoader containing the data to test the network on.
        device : torch.device
            The device on which the model should be tested, either 'cpu' or 'cuda'.

        Returns
        -------
        Tuple[float, float]
            The loss and the accuracy of the input model on the given data.
        """
        net = self.model
        testloader = self.train_loader
        criterion = torch.nn.L1Loss()
        net.eval()
        lim_diff = self.lim_diff
        correct, total, losses, total_loss = 0, 0, 0.0, 0.0
        # writer = SummaryWriter('./data_log/test')
        with torch.no_grad():
            for features, labels in testloader:
                # features = torch.tensor(data_items[0])
                # labels = [torch.tensor([x]) for x in data_items[1]]
                outputs = net(features)
                outputs = torch.cat(outputs, dim=1)
                # print(f"outputs:{outputs}")
                # print(f"labels:{labels}")
                losses = [criterion(output, label) for output, label in zip(outputs, labels)]
                total_loss += sum(losses)
                predicteds = outputs
                total += 1
                pred_y = [tensor.numpy() for tensor in predicteds]
                label_y = [tensor.numpy() for tensor in labels]
                diff = [abs(x - y) for x, y in zip(pred_y, label_y)]
                # print(f"diff:{diff}")
                if all(d <= l for d, l in zip(diff[0], lim_diff)):  # 输出概率差值全部小于lim_diff则认为是预测正确，因为是批处理所以修改为diff[0]
                    correct += 1

        accuracy = correct / total
        avg_loss = total_loss / total
        # print(f"accuracy: {accuracy}")

        return avg_loss, accuracy
