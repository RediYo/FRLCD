import copy
import random

import numpy as np
import torch
import time
from flcore.clients.clientALA import *
# from utils.data_utils import read_client_data
from threading import Thread

# from utils.ble_dataset_loader import load_datasets
from utils.h0h1 import create_dataset_loader

class FedALA(object):
    def __init__(self, args, times):
        self.device = args.device
        self.dataset = args.dataset
        self.global_rounds = args.global_rounds
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.join_clients = int(self.num_clients * self.join_ratio)

        self.clients = []
        self.selected_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_train_loss = []

        # 每轮的 精确度
        self.acc_epoches = []

        self.times = times
        self.eval_gap = args.eval_gap

        self.set_clients(args, clientALA)  # 初始化客户端

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()
                # 纪录轮次与相应的test accuracy
                self.acc_epoches.append((i, self.rs_test_acc[-1]))

            threads = []
            for client in self.selected_clients:
                epochs = random.randint(3, 32)
                threads.append(Thread(target=client.train(epochs)))
            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            [t.start() for t in threads]
            [t.join() for t in threads]

            self.receive_models()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-' * 50, self.Budget[-1])



        print("\nBest global accuracy.")
        print(f"max_test_acc:{max(self.rs_test_acc)}")
        print(f"avg_time:{sum(self.Budget[1:]) / len(self.Budget[1:])}")
        print(self.acc_epoches)

    def set_clients(self, args, clientObj):
        # for i in range(self.num_clients):
        #     train_data = read_client_data(self.dataset, i, is_train=True)
        #     test_data = read_client_data(self.dataset, i, is_train=False)
        #     client = clientObj(args,
        #                        id=i,
        #                        train_samples=len(train_data),
        #                        test_samples=len(test_data))

        # BLE-Move
        # trainloaders, valloaders = load_datasets(self.num_clients)
        # H0H1
        trainloaders, valloaders = create_dataset_loader(self.num_clients)

        for i in range(self.num_clients):
            client = clientObj(args,
                               id=i,
                               train_samples=len(trainloaders[i]),
                               test_samples=len(valloaders[i]),
                               train_loader=trainloaders[i],
                               test_loader=valloaders[i])
            self.clients.append(client)

    def select_clients(self):
        if self.random_join_ratio:
            join_clients = np.random.choice(range(self.join_clients, self.num_clients + 1), 1, replace=False)[0]
        else:
            join_clients = self.join_clients
        selected_clients = list(np.random.choice(self.clients, join_clients, replace=False))

        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            client.local_initialization(self.global_model)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_train_samples = 0
        for client in self.selected_clients:
            active_train_samples += client.train_samples

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []
        for client in self.selected_clients:
            self.uploaded_weights.append(client.train_samples / active_train_samples)
            self.uploaded_ids.append(client.id)
            self.uploaded_models.append(client.model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data = torch.zeros_like(param.data)

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def test_metrics(self):
        loss, accuracy = 0.0, 0.0
        for c in self.clients:
            # loss, accuracy = c.test_metrics()
            loss, accuracy = c.test()
            break

        return loss, accuracy

    def train_metrics(self):
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            print(f'Client {c.id}: Train loss: {cl * 1.0 / ns}')
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    def evaluate(self, acc=None, loss=None):
        # stats = self.test_metrics()
        # stats_train = self.train_metrics()
        #
        # test_acc = sum(stats[2])*1.0 / sum(stats[1])
        # test_auc = sum(stats[3])*1.0 / sum(stats[1])
        # train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        # accs = [a / n for a, n in zip(stats[2], stats[1])]
        # aucs = [a / n for a, n in zip(stats[3], stats[1])]
        #
        # if acc == None:
        #     self.rs_test_acc.append(test_acc)
        # else:
        #     acc.append(test_acc)
        #
        # if loss == None:
        #     self.rs_train_loss.append(train_loss)
        # else:
        #     loss.append(train_loss)
        loss, accuracy = self.test_metrics()
        if acc is None:
            self.rs_test_acc.append(accuracy)
        else:
            acc.append(accuracy)

        print(f"loss:{loss}, accuracy:{accuracy}")
