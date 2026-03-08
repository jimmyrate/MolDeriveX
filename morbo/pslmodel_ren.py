import torch
import torch.nn as nn
import math
import torch.nn.functional as F



def thermometer(v, n_bins=50, vmin=0, vmax=1, device='cuda:0'):
    bins = torch.linspace(vmin, vmax, n_bins, device=device)
    gap = bins[1] - bins[0]
    return (v[..., None] - bins.reshape((1,) * v.ndim + (-1,))).clamp(0, gap.item()) / gap

# class ParetoSetModel256(torch.nn.Module):
#     def __init__(self, n_dim, n_obj):
#         super(ParetoSetModel256, self).__init__()
#         self.n_dim = n_dim
#         self.n_obj = n_obj

#         # self.fc1 = nn.Linear(self.n_obj, 512)
#         # self.fc2 = nn.Linear(512, 512)
#         # self.fc3 = nn.Linear(512, self.n_dim)
#         self.fc1 = nn.Linear(self.n_obj, 128)
#         self.fc2 = nn.Linear(128, 512)
#         self.fc3 = nn.Linear(512, 512)
#         self.fc4 = nn.Linear(512, self.n_dim)

#     def forward(self, feature_vector):
#         x = torch.tanh(self.fc1(feature_vector))
#         x = torch.tanh(self.fc2(x))
#         x = torch.tanh(self.fc3(x))
#         x = self.fc4(x)
#         x = torch.tanh(x)
#         return x.to(torch.float64)
class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.input_dim = input_dim

        self.query = nn.Linear(input_dim, input_dim, bias=False)
        self.key = nn.Linear(input_dim, input_dim, bias=False)
        self.value = nn.Linear(input_dim, input_dim, bias=False)
        self.fc = nn.Linear(input_dim, input_dim, bias=False)

        self.dropout = nn.Dropout(0.1)

        # 使用 xavier 初始化
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        Q = self.query(x).view(x.size(0), -1, self.num_heads, self.head_dim)
        K = self.key(x).view(x.size(0), -1, self.num_heads, self.head_dim)
        V = self.value(x).view(x.size(0), -1, self.num_heads, self.head_dim)

        QK = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        QK = F.softmax(QK, dim=-1)
        QK = self.dropout(QK)
        V = torch.matmul(QK, V).view(x.size(0), -1, self.input_dim).squeeze()
        V = self.fc(V)

        return V

class CrossAttention(nn.Module):
    def __init__(self, input_dim, num_heads, device='cpu'):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.input_dim = input_dim
        self.device = device

        self.query = nn.Linear(input_dim, input_dim, bias=False)
        self.key = nn.Linear(input_dim, input_dim, bias=False)
        self.value = nn.Linear(input_dim, input_dim, bias=False)
        self.fc = nn.Linear(input_dim, input_dim)

        self.dropout = nn.Dropout(0.1)
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, query, key, value):
        batch_size = query.size(0)
        Q = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = F.softmax(attention_scores, dim=-1)
        attention = self.dropout(attention)
        out = torch.matmul(attention, V).transpose(1, 2).contiguous()
        out = out.view(batch_size, -1)  # 重新形状为 [batch_size, input_dim]
        return self.fc(out)


class ParetoSetModel256(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=256, output_dim=128, num_heads=4,device='cuda:0'):
        super(ParetoSetModel256, self).__init__()
        self.encoded_dim = 128  # Adjust this based on your encoding bins
        self.device = device

        self.layer1 = nn.Linear(self.encoded_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.1)
        self.attention = CrossAttention(hidden_dim, num_heads, device=device)
        # self.attention = MultiHeadAttention(hidden_dim, num_heads)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.dropout2 = nn.Dropout(0.1)
        self.layer4 = nn.Linear(hidden_dim, hidden_dim)

        self.layer3 = nn.Linear(self.encoded_dim, output_dim)  # Adjusted to accept encoded_dim
        self.bn3 = nn.BatchNorm1d(output_dim)
        self.bn4 = nn.BatchNorm1d(output_dim)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.Sigmoid = nn.Sigmoid()
        # nn.init.xavier_uniform_(self.layer1.weight)
        # nn.init.xavier_uniform_(self.layer2.weight)
        # nn.init.xavier_uniform_(self.layer3.weight)

        nn.init.kaiming_normal_(self.layer1.weight, mode='fan_out', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.layer2.weight, mode='fan_out', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.layer3.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x):

        x_encoded = thermometer(x, n_bins=self.encoded_dim, device=self.device)  # 编码后维度为 [100, 2, 50]
        preference1 = x_encoded[:, 0, :]  # 第一个偏好, 维度为 [100, 50]
        preference2 = x_encoded[:, 1, :]  # 第二个偏好, 维度为 [100, 50]

        preference1_res = self.bn3(self.layer3(preference1))
        preference2_res = self.bn3(self.layer3(preference2))


        x1 = self.leaky_relu(self.bn1(self.layer1(preference1)))
        x1 = self.dropout1(x1)

        # 处理第二个偏好
        x2 = self.leaky_relu(self.bn1(self.layer1(preference2)))
        x2 = self.dropout1(x2)

        # 应用 Cross-Attention
        x1 = self.attention(x1, x2, x2)  # Preference1 对 Preference2 使用 Cross-Attention
        x2 = self.attention(x2, x1, x1)  # Preference2 对 Preference1 使用 Cross-Attention

        x1 = self.dropout2(x1)
        x1 = self.bn1(x1)
        x1 = self.leaky_relu(self.layer4(x1))
        x1 = self.layer2(x1)
        x1 = self.bn2(x1)
        x1 = self.leaky_relu(x1)
        # x1 = torch.cat((x1, preference1_res), dim=1)
        x1 += preference1_res


        # x1 = self.bn4(x1)


        x2 = self.dropout2(x2)
        x2 = self.bn1(x2)
        x2 = self.leaky_relu(self.layer4(x2))
        x2 = self.layer2(x2)
        x2 = self.bn2(x2)
        x2 = self.leaky_relu(x2)
        x2 += preference2_res  # Adding the residual connection

        # x2 = self.bn4(x2)
        # x2 = torch.cat((x2, preference2_res), dim=1)
        # 合并两个处理后的偏好
        x = x1 + x2  # Concatenate outputs
        x = self.Sigmoid(x)

        return x.to(torch.float64)


class ParetoSetModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=256, output_dim=128, num_heads=4):
        super(ParetoSetModel, self).__init__()

        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.1)
        self.attention = MultiHeadAttention(hidden_dim, num_heads)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.dropout2 = nn.Dropout(0.1)
        self.layer4 = nn.Linear(hidden_dim, hidden_dim)

        self.layer3 = nn.Linear(input_dim, output_dim)  # additional layer for the residual connection
        self.bn3 = nn.BatchNorm1d(output_dim)  # BatchNorm for the residual connection
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        # 使用 xavier 初始化
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.xavier_uniform_(self.layer3.weight)

    def forward(self, x):
        preference = self.bn3(self.layer3(x))

        x = self.leaky_relu(self.bn1(self.layer1(x)))
        x = self.dropout1(x)
        x = self.attention(x)
        x = self.dropout2(x)
        x = self.leaky_relu(self.layer4(x))
        x = self.bn2(self.layer2(x))
        x = torch.cat((x, preference), dim=1)

        # x = torch.sigmoid(x)
        return x.to(torch.float64)