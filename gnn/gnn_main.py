import pandas as pd
import numpy as np
import torch
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data
import torch.optim as optim
from mgnn import MGNN
from gat import GAT
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# 读取数据
papers = pd.read_csv('../../papers.csv.gz', compression='gzip')
feats = pd.read_csv('../../feats.csv.gz', compression='gzip', header=None).values.astype(np.float32)
edges = pd.read_csv('../../edges.csv.gz', compression='gzip', header=None).values.T.astype(np.int32)  # 转置，citer, citee
print("read done")

# 获取标签
labels = papers['category']
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# 构建邻接矩阵
num_nodes = feats.shape[0]
adj = sp.coo_matrix((np.ones(len(edges[0])), (edges[0], edges[1])), shape=(num_nodes, num_nodes))

# 添加自环
adj = adj + sp.eye(num_nodes)

# 归一化邻接矩阵
degree = np.array(adj.sum(1)).flatten()
degree_inv = 1.0 / degree
degree_inv[np.isinf(degree_inv)] = 0.0
degree_inv = sp.diags(degree_inv)

# 计算归一化邻接矩阵
adj_normalized = degree_inv @ adj

# 数据准备
# 创建图数据
edge_index = torch.tensor(edges, dtype=torch.long)  # 论文引用关系（边）

# 特征矩阵
features = torch.tensor(feats, dtype=torch.float32)

# 标签
labels = torch.tensor(labels_encoded, dtype=torch.long)

# 构建 PyG 数据
data = Data(x=features, edge_index=edge_index, y=labels)

# 训练集、验证集、测试集划分
train_mask = papers['year'] <= 2017
val_mask = papers['year'] == 2018
test_mask = papers['year'] >= 2019

data.train_mask = torch.tensor(train_mask.values, dtype=torch.bool)
data.val_mask = torch.tensor(val_mask.values, dtype=torch.bool)
data.test_mask = torch.tensor(test_mask.values, dtype=torch.bool)


# 初始化模型
input_dim = feats.shape[1]
hidden_dim = 128
num_classes = len(np.unique(labels_encoded))

model = GAT(input_dim, hidden_dim, num_classes)

# 选择优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()


# 训练过程
def train(model, data, optimizer, criterion, device):
    print("Training on", device)
    model.train()
    optimizer.zero_grad()
    out = model(data.x.to(device), data.edge_index.to(device))
    loss = criterion(out[data.train_mask], data.y[data.train_mask].to(device))
    loss.backward()
    optimizer.step()
    return loss.item()


def vali(model, data, device):
    model.eval()
    out = model(data.x.to(device), data.edge_index.to(device))
    _, pred = out.max(dim=1)
    correct = (pred[data.val_mask] == data.y[data.val_mask].to(device)).sum()
    acc = correct / data.val_mask.sum().item()
    return acc


# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 使用 DataParallel 实现多GPU训练
model = torch.nn.DataParallel(model)
model.to(device)
data.to(device)

torch.cuda.empty_cache()  # 清理缓存

# 训练和测试
epochs = 100
for epoch in range(epochs):
    loss = train(model, data, optimizer, criterion, device)
    acc = vali(model, data, device)

    print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss: .4f}, Test Accuracy: {acc: .4f}')


def predict(model, data, device):
    print("Predicting on", device)
    model.eval()
    out = model(data.x.to(device), data.edge_index.to(device))
    _, pred = out.max(dim=1)
    return pred.cpu().numpy()


# 获取测试集预测结果
test_predictions = predict(model, data, device)
np.savetxt('test_predictions_gnn.csv', test_predictions, delimiter=',')

