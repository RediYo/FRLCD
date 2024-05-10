import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalModel(nn.Module):
    def __init__(self, feature_extractor, head):
        super(LocalModel, self).__init__()

        self.feature_extractor = feature_extractor
        self.head = head
        
    def forward(self, x, feat=False):
        out = self.feature_extractor(x)
        if feat:
            return out
        else:
            out = self.head(out)
            return out


class FedAvgCNN(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024, dim1=512):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features,
                        32,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                        64,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, dim1), 
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(dim1, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out


class fastText(nn.Module):
    def __init__(self, hidden_dim, padding_idx=0, vocab_size=98635, num_classes=10):
        super(fastText, self).__init__()
        
        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        
        # Hidden Layer
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output Layer
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        text, text_lengths = x

        embedded_sent = self.embedding(text)
        h = self.fc1(embedded_sent.mean(1))
        z = self.fc(h)
        out = z

        return out

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

# 密接检测模型
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