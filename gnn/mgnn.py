# 多图神经网络
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class MGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MGNN, self).__init__()
        # 第一个GCN层
        self.conv1 = GCNConv(input_dim, hidden_dim)
        # 第二个GCN层
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        # 第三个GCN层
        self.conv3 = GCNConv(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        # GCN层1
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)

        # GCN层2
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)

        # GCN层3
        x = self.conv3(x, edge_index)

        return x
