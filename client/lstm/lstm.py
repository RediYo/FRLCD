import csv
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path

# todo 数据预处理 使用走廊或者大厅前20分钟数据作为训练集，后10分钟作为验证集
sorted_path = "../sorted_data"
sp = Path(sorted_path)
data = []
# 直接遍历出文件绝对路径
for file_name in sp.rglob('*.csv'):
    with open(file_name) as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        tag = next(csv_reader)  # 读取第一行作为标签
        temp = 0
        tempData = []
        for row in csv_reader:  # 将csv 文件中的数据保存到data中
            # 转换成时间数组
            timeArray = time.strptime(row[5], "%Y/%m/%d %H:%M:%S")
            # 转换成时间戳
            timestamp = time.mktime(timeArray)
            if temp == 0:
                interval = 5  # 5s
            else:
                interval = timestamp - temp
            temp = timestamp
            t = (interval, float(row[8]))  # 选择某几列加入到data数组中
            tempData.append(t)
        dataTuple = (
        tempData, [float(tag[0]) / 10, float(tag[1]) / 10, float(tag[2]) / 10, float(tag[3]) / 10, float(tag[4]) / 10])
        data.append(dataTuple)
print(data)

training_data = data
testing_data = data

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
INPUT_DIM = 2
OUTPUT_DIM = 5
HIDDEN_DIM = 6


class LSTMTagger(nn.Module):

    def __init__(self, input_dim, hidden_dim, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, input):
        lstm_out, _ = self.lstm(torch.tensor(input, dtype=torch.float32))
        print(f"lstm_out {lstm_out[-1]}")
        tag_space = self.hidden2tag(lstm_out[-1])
        print(f"tag_space {tag_space}")
        tag_scores = F.softmax(tag_space, dim=0)
        print(f"tag_scores {tag_scores}")
        return tag_scores


model = LSTMTagger(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)


# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
# with torch.no_grad():
#     inputs = training_data[0][0]
#     tag_scores = model(inputs)
#     print(f"pre_test: {tag_scores}")


def train(model, training_data, epochs):
    """Train the network on the training set."""
    loss_function = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    total_loss = 0
    for epoch in range(epochs):
        for data_items in training_data:
            features = data_items[0]
            tags = data_items[1]
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Run our forward pass.
            tag_scores = model(features)

            # Step 3. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(tag_scores, torch.tensor(tags, dtype=torch.float32))
            total_loss += loss
            # print("loss:", loss)
            loss.backward()
            optimizer.step()
    return total_loss


def test(model, testing_data):
    """Validate the network on the entire test set."""
    loss_function = nn.MSELoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for data_items in testing_data:
            features = data_items[0]
            tags = data_items[1]
            tag_scores = model(features)
            loss += loss_function(tag_scores, torch.tensor(tags, dtype=torch.float32))
            predicted = tag_scores
            total += 1
            pred_y = predicted.numpy()
            label_y = np.array(tags)
            diff = abs(np.array(pred_y - label_y))
            if np.max(diff) <= 0.05:  # 输出概率差值小于0.05则认为是预测正确
                correct += 1
            # if pred_y.all == label_y.all:
            #     correct += 1
            print(f"pred_y: {pred_y}")
            print(f"label_y: {label_y}")
            print(f"diff: {diff}")
            print(f"correct: {correct}")
    accuracy = correct / total
    print(f"accuracy: {accuracy}")
    return loss, accuracy


model.load_state_dict(torch.load("state_dict_model.pth"))
total_loss = train(model, training_data, 30)
torch.save(model.state_dict(), "state_dict_model.pth")
loss, accuracy = test(model, testing_data)
