# -*- coding: utf-8 -*
"""
ID3算法
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# 1. 生成虚拟数据集
data = {
    'Weather': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 'Sunny', 
                'Sunny', 'Rainy', 'Sunny', 'Overcast', 'Overcast', 'Rainy'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 
                    'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 
                 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Windy': ['False', 'True', 'False', 'False', 'False', 'True', 'True', 'False', 
              'False', 'False', 'True', 'True', 'False', 'True'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 
                   'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}
df = pd.DataFrame(data)

# 2. 数据预处理
# 将分类变量转换为数值
le = LabelEncoder()
for column in df.columns:
    df[column] = le.fit_transform(df[column])

# 特征和目标变量
X = df[['Weather', 'Temperature', 'Humidity', 'Windy']]
y = df['PlayTennis']

# 3. 构建 ID3 决策树
model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
model.fit(X, y)

# 4. 输出信息增益分析
features = ['Weather', 'Temperature', 'Humidity', 'Windy']
info_gain = []
H_S = -np.sum((np.bincount(y) / len(y)) * np.log2(np.bincount(y) / len(y)))  # 数据集熵
for i, feature in enumerate(features):
    unique_values = np.unique(X.iloc[:, i])
    H_S_A = 0
    for value in unique_values:
        subset = y[X.iloc[:, i] == value]
        p = len(subset) / len(y)
        H_S_A += p * (-np.sum((np.bincount(subset) / len(subset)) * np.log2(np.bincount(subset) / len(subset))))
    info_gain.append(H_S - H_S_A)

# 可视化信息增益
plt.figure(figsize=(8, 6))
plt.bar(features, info_gain, color=['red', 'blue', 'green', 'purple'], alpha=0.8)
plt.title("Information Gain for Each Feature", fontsize=16)
plt.xlabel("Features", fontsize=14)
plt.ylabel("Information Gain", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 5. 决策树可视化
plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=features, class_names=['No', 'Yes'], filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree Visualization", fontsize=16)
plt.show()

# 6. 测试与优化
# 模拟新数据
test_data = pd.DataFrame({
    'Weather': ['Sunny', 'Rainy', 'Overcast'],
    'Temperature': ['Hot', 'Mild', 'Cool'],
    'Humidity': ['Normal', 'High', 'High'],
    'Windy': ['False', 'True', 'False']
})

# 数据转换
for column in test_data.columns:
    test_data[column] = le.fit_transform(test_data[column])

# 预测
predictions = model.predict(test_data)
print("Predictions for test data:", predictions)

# 优化策略（可选）：调整树的深度，选择特征重要性
optimized_model = DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=42)
optimized_model.fit(X, y)
optimized_predictions = optimized_model.predict(test_data)
print("Optimized Predictions:", optimized_predictions)

# 输出准确率（仅供模拟）
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
print("Model Accuracy on Training Data:", accuracy)
