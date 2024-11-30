import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# 加载数据
papers = pd.read_csv('../papers.csv.gz', compression='gzip')
feats = pd.read_csv('../feats.csv.gz', compression='gzip', header=None).values.astype(np.float32)
print("read done")

# 分离数据集
train = papers[papers['year'] <= 2017]
val = papers[papers['year'] == 2018]
test = papers[papers['year'] >= 2019]

# 特征和标签
X_train = feats[train.index]
y_train = train['category']
X_val = feats[val.index]
y_val = val['category']
X_test = feats[test.index]

# 特征归一化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# 模型训练
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# 模型验证
y_pred_val = clf.predict(X_val)
val_accuracy = accuracy_score(y_val, y_pred_val)
print(f'Validation Accuracy: {val_accuracy}')

# 预测测试集
y_pred_test = clf.predict(X_test)

# 将预测结果与测试集合并
test['predicted_category'] = y_pred_test
test['category'] = 'unknown'  # 标记测试集的category为'unknown'

# 保存预测结果
test.to_csv('predicted_test_set.csv', index=False)

# TODO 用神经网络优化提高准确率
