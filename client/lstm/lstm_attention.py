import csv
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

# # 关闭WIFI组走廊
# dataset_train_not_wifi_A = "../dataset/关闭WIFI组/走廊"
# # 关闭WIFI组大厅
# dataset_train_not_wifi_B = "../dataset/关闭WIFI组/大厅"
# # 开启WiFi组走廊
# dataset_train_wifi_A = "../dataset/开启WIFI组/走廊"
# # 开启WiFi组大厅
# dataset_train_wifi_B = "../dataset/开启WIFI组/大厅"


# 关闭WIFI组走廊
dataset_train_not_wifi_A = "../dataset/关闭WIFI组/走廊/is_full_data"
# 关闭WIFI组大厅
dataset_train_not_wifi_B = "../dataset/关闭WIFI组/大厅/is_full_data"
# 开启WiFi组走廊
dataset_train_wifi_A = "../dataset/开启WIFI组/走廊/is_full_data"
# 开启WiFi组大厅
dataset_train_wifi_B = "../dataset/开启WIFI组/大厅/is_full_data"

# 关闭WIFI组走廊
dataset_train_not_wifi_A_not_full = "../dataset/关闭WIFI组/走廊/not_full_data"
# 关闭WIFI组大厅
dataset_train_not_wifi_B_not_full = "../dataset/关闭WIFI组/大厅/not_full_data"
# 开启WiFi组走廊
dataset_train_wifi_A_not_full = "../dataset/开启WIFI组/走廊/not_full_data"
# 开启WiFi组大厅
dataset_train_wifi_B_not_full = "../dataset/开启WIFI组/大厅/not_full_data"

data_path_train = [dataset_train_not_wifi_A, dataset_train_not_wifi_B, dataset_train_wifi_A, dataset_train_wifi_B]
data_path_test = [dataset_train_not_wifi_A_not_full, dataset_train_not_wifi_B_not_full, dataset_train_wifi_A_not_full,
                  dataset_train_wifi_B_not_full]

# temp = data_path_train
# data_path_train = data_path_test
# data_path_test = temp


# 遍历目录下所有的 csv 文件
# 遍历 https://blog.csdn.net/li1123576747/article/details/111307414
def generate_data(data_path_strs):
    data = []
    for data_path in data_path_strs:
        train_p = Path(data_path + "")
        # tag_p = Path(dataset_train_tag_not_wifi_A_0_10)
        for file in train_p.glob('*.csv'):
            with open(file) as csvfile:
                csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件

                header = next(csv_reader)  # 读取第一行作为表头
                temp = 0
                tempData = []
                timeLong = 0
                for row in csv_reader:  # 将csv 文件中的数据保存到data中
                    # 转换成时间数组
                    timeArray = time.strptime(row[2], "%Y/%m/%d %H:%M:%S")
                    # 转换成时间戳
                    timestamp = time.mktime(timeArray)
                    if temp == 0:
                        interval = 5  # 5s
                    else:
                        interval = timestamp - temp
                    temp = timestamp
                    timeLong += interval
                    t = (interval, float(row[6]))  # 选择某几列加入到data数组中
                    tempData.append(t)
                # 1.如果一次训练数据太少则不参与训练，因为很可能缺失了数据，蓝牙信号未接收到
                # 2.如果相隔时间太短则也表示数据缺失严重不参与训练
                # if csv_reader.line_num < 10 or timeLong < 400:
                #     continue
                # 利用tag文件数据生成标签，如果设置new_tag则将距离区间范围扩大例如 （0,2.5)(2.5,5.5)(5.5,10)
                # strs = (file.name.replace(".csv", "")).split("_")
                # filename = strs[0] + "_" + strs[1]
                # print(f"filename:{filename}")
                # tag_file = Path(data_path + "/tag/" + filename + "_tag.csv")
                # print(f"tag_file: {tag_file}")
                # if not tag_file.exists():
                #     filename = strs[1] + "_" + strs[0]
                #     tag_file = Path(data_path + "/tag/" + filename + "_tag.csv")
                tag_file = Path(str(Path(data_path).parent) + "/tag/" + file.name)
                with open(tag_file) as tag_file_in:
                    csv_tag_reader = csv.reader(tag_file_in)  # 使用csv.reader读取csv
                    tag = next(csv_tag_reader)  # 读取第一行标签
                    print(f"tag: {tag}")
                tag = [int(x) for x in tag]
                new_tag = []  # （0,2.5)(2.5,5.5)(5.5,10)
                new_tag.append(tag[0] + tag[1] + tag[2])
                new_tag.append(tag[3] + tag[4] + tag[5])
                new_tag.append(tag[6] + tag[7] + tag[8] + tag[9])

                # dataTuple = (
                #     tempData,
                #     [float(tag[0]) / 10, float(tag[1]) / 10, float(tag[2]) / 10, float(tag[3]) / 10, float(tag[4]) / 10,
                #      float(tag[5]) / 10, float(tag[6]) / 10, float(tag[7]) / 10, float(tag[8]) / 10, float(tag[9]) / 10])
                print(f"sum(new_tag):{sum(new_tag)}")
                dataTuple = (
                    tempData,
                    # [new_tag[0] / sum(new_tag), new_tag[1] / sum(new_tag), new_tag[2] / sum(new_tag)])
                    [new_tag[0], new_tag[1], new_tag[2]])
                data.append(dataTuple)
    print(data)
    return data


train_data = generate_data(data_path_train)
test_data = generate_data(data_path_test)

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
INPUT_DIM = 2
OUTPUT_DIM = 3
HIDDEN_DIM = 6


class LSTMTagger_Attention(nn.Module):

    def __init__(self, input_dim, hidden_dim, tagset_size):
        super(LSTMTagger_Attention, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.w_omega = nn.Parameter(torch.Tensor(HIDDEN_DIM, HIDDEN_DIM * 2))
        self.u_omega = nn.Parameter(torch.Tensor(HIDDEN_DIM * 2, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def attention_net(self, lstm_output, h_t):
        # lstm_output.shape=(seq_len, hidden_size)
        # lstm_output.shape=(seq_len,6)  h_t(1,6)
        # Attention过程
        u = torch.tanh(torch.mm(lstm_output, self.w_omega))
        # u.shape=(seq_len, HIDDEN_DIM * 2)
        att = torch.mm(u, self.u_omega)
        # att.shape=(seq_len, 1)
        print(f"att.shape {att.shape}")
        att_score = F.softmax(att, dim=1)
        # att_score.shape=(seq_len, 1)
        scored = lstm_output * att_score
        # scored.shape=(seq_len, HIDDEN_DIM)
        print(f"scored.shape {scored.shape}")
        # Attention过程结束

        attn_out = torch.sum(scored, dim=0)  # 加权求和
        # feat.shape=(HIDDEN_DIM)
        print(f"feat.shape {attn_out.shape}")
        return attn_out

    def forward(self, input):
        lstm_output, (h_t, c_t) = self.lstm(torch.tensor(input, dtype=torch.float32))
        print(f"lstm_out{lstm_output}")
        print(f"h_t {h_t}")
        # 对lstm_output做attention
        attn_out = self.attention_net(lstm_output, h_t)
        tag_space = self.hidden2tag(attn_out)
        print(f"tag_space {tag_space}")
        tag_scores = F.softmax(tag_space, dim=0)
        print(f"tag_scores {tag_scores}")
        return tag_scores


model = LSTMTagger_Attention(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)


# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
# with torch.no_grad():
#     inputs = train_data[0][0]
#     tag_scores = model(inputs)
#     print(f"pre_test: {tag_scores}")
# Tensorboard https://zhuanlan.zhihu.com/p/103630393

def train(model, train_data, epochs):
    """Train the network on the training set."""
    loss_function = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    total_loss = 0
    writer = SummaryWriter('./data_log/train')
    for epoch in range(epochs):
        loss_epoch = 0
        for data_items in train_data:
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
            loss_epoch += loss
            # print("loss:", loss)
            loss.backward()
            optimizer.step()
        writer.add_scalar("loss", loss_epoch, epoch)
        print(f"loss_epoch:{loss_epoch}")
        total_loss += loss_epoch
    return total_loss


def test(model, test_data):
    """Validate the network on the entire test set."""
    loss_function = nn.MSELoss()
    correct, total, loss = 0, 0, 0.0
    writer = SummaryWriter('./data_log/test')
    with torch.no_grad():
        for data_items in test_data:
            features = data_items[0]
            tags = data_items[1]
            tag_scores = model(features)
            loss += loss_function(tag_scores, torch.tensor(tags, dtype=torch.float32))
            predicted = tag_scores
            total += 1
            pred_y = predicted.numpy()
            label_y = np.array(tags)
            diff = abs(np.array(pred_y - label_y))
            if np.max(diff) <= 1:  # 输出概率差值全部小于1分钟则认为是预测正确
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
total_loss = train(model, train_data, 30)
print(f"total_loss:{total_loss}")
torch.save(model.state_dict(), "state_dict_model.pth")
loss, accuracy = test(model, test_data)
