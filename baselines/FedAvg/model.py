from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.q_transform = nn.Linear(hidden_dim, hidden_dim)
        self.k_transform = nn.Linear(hidden_dim, hidden_dim)
        self.v_transform = nn.Linear(hidden_dim, hidden_dim)
        self.out_transform = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        q = self.q_transform(x)
        k = self.k_transform(x)
        v = self.v_transform(x)
        q = q.view(q.size(0), self.num_heads, -1, q.size(-1) // self.num_heads)
        k = k.view(k.size(0), self.num_heads, -1, k.size(-1) // self.num_heads)
        v = v.view(v.size(0), self.num_heads, -1, v.size(-1) // self.num_heads)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / ((x.size(-1) // self.num_heads) ** 0.5)
        soft_attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(soft_attn_weights, v).transpose(1, 2).contiguous().view(x.size(0), -1, x.size(-1))
        out = self.out_transform(attn_output)
        return out


class AttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dims, num_heads, num_layers, dropout):
        super(AttentionLSTM, self).__init__()
        self.attention = SelfAttention(input_dim, num_heads)
        self.lstm = nn.LSTM(input_size=input_dim*2, hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        # 定义多个全连接层，分别用于不同的任务
        self.fcs = nn.ModuleList([nn.Linear(hidden_dim, output_dim) for output_dim in output_dims])
        self.softmax = nn.Softmax(dim=1)  # 添加softmax层

    def forward(self, x):
        attention_output = self.attention(x)
        lstm_input = torch.cat([x, attention_output], dim=-1)  # 将原始输入和注意力机制的输出进行拼接
        lstm_output, _ = self.lstm(lstm_input)
        output = lstm_output[:, -1, :]
        # 分别传入多个全连接层中
        outputs = [fc(output) for fc in self.fcs]

        # 每个任务的输出之和为1，即所有任务的输出之和为n
        all_outputs = torch.cat(outputs, dim=1)
        all_outputs_sum = torch.sum(all_outputs, dim=1, keepdim=True)
        outputs = [output / all_outputs_sum for output in outputs]

        return outputs


def train(
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
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
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    net.train()
    total_loss = 0
    # writer = SummaryWriter('./data_log/train')
    for epoch in range(1, epochs + 1):
        loss_epoch = 0
        for features, labels in trainloader:
            # features = torch.tensor(data_items[0]).to(device)
            # labels = [torch.tensor([x]).to(device) for x in data_items[1]]
            # print(f"features:{features}")
            # print(f"labels:{labels}")
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
        # writer.add_scalar("loss_epoch", loss_epoch, epoch)
        total_loss += loss_epoch


def test(
    net: nn.Module, testloader: DataLoader, device: torch.device
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
    criterion = torch.nn.L1Loss()
    net.eval()
    lim_diff = [0.1, 0.2, 0.3]
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
            # if pred_y.all == label_y.all:
            #     correct += 1
            # print(f"pred_y: {pred_y}")
            # print(f"label_y: {label_y}")
            # print(f"diff: {diff}")
            # print(f"correct: {correct}")

    accuracy = correct / total
    avg_loss = total_loss / total
    # print(f"accuracy: {accuracy}")

    return avg_loss, accuracy
