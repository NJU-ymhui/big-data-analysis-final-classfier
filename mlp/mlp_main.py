import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import scipy.sparse as sp
from mlp import MLP
from util import *


# 加载数据
papers = pd.read_csv('../../papers.csv.gz', compression='gzip')
feats = pd.read_csv('../../feats.csv.gz', compression='gzip', header=None).values.astype(np.float32)
edges = pd.read_csv(f'../../edges.csv.gz', compression='gzip', header=None).values.T.astype(np.int32)
print("read done")

# 提取训练集、验证集、测试集
train_mask = papers['year'] <= 2017
val_mask = papers['year'] == 2018
test_mask = papers['year'] >= 2019

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

# 使用邻接矩阵增强特征
feats_augmented = adj_normalized @ feats

# 更新训练、验证和测试集
X_train = feats_augmented[train_mask]
X_val = feats_augmented[val_mask]
X_test = feats_augmented[test_mask]


class PapersDataset(Dataset):
    def __init__(self, features, labels=None):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]


y_train = labels_encoded[train_mask]
y_val = labels_encoded[val_mask]

# 构建数据集和数据加载器
batch_size = 32
train_dataset = PapersDataset(X_train, y_train)
val_dataset = PapersDataset(X_val, y_val)
test_dataset = PapersDataset(X_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 检查类别数量
num_classes = len(np.unique(y_train))

# 初始化模型
input_dim = X_train.shape[1]
hidden_dim = 1024  # 可调节
model = MLP(input_dim, hidden_dim, num_classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 学习率调度：使用学习率调度器


# 训练和测试 MLP
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=20, device=device)
test_predictions = predict(model, test_loader, device=device)
# 使用 label_encoder 将预测结果从数字标签映射回原始类别标签
predicted_categories = label_encoder.inverse_transform(test_predictions)

# 3. 将预测结果写入原始的 papers 数据框
# 将预测的类别添加到 papers 的 `category` 列中，注意只对测试集的部分进行操作
papers.loc[test_mask, 'category'] = predicted_categories

# 4. 保存新的 DataFrame 到新的 CSV 文件
papers.to_csv('papers_with_predictions.csv', index=False, encoding='utf8mb4')

