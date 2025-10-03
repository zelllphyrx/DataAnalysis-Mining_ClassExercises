import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, Binarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# 加载波士顿数据集
dataset = pd.read_csv('DataSet/boston.csv')
print(dataset.head())

# 选择等宽法离散化LSTAT特征
plt.figure(figsize=(12, 5))

# 原始数据分布
plt.subplot(1, 2, 1)
dataset['LSTAT'].hist(bins=20)
plt.title('Original LSTAT Distribution')

# 等宽离散化（分为4类）
dataset['LSTAT_cut'] = pd.cut(dataset['LSTAT'], bins=4, labels=['low', 'medium', 'high', 'very_high'])

# 离散化结果可视化
plt.subplot(1, 2, 2)
dataset['LSTAT_cut'].value_counts().sort_index().plot(kind='bar')
plt.title('Discretized LSTAT Distribution')
plt.tight_layout()
plt.show()

# 数据规范化
# 划分数据集
X = dataset.iloc[:, :-2].values  # 排除最后两列（目标列和离散化列）
y = dataset.iloc[:, -2].values

# 未规范化的数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=25)

#相关系数矩阵，即给出了任意两个变量之间的相关系数
corr = dataset.corr()
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corr, cmap='RdBu', annot=True, fmt=".2f")
plt.xticks(range(len(corr.columns)), corr.columns)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.show()

# 标准化处理
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 模型定义
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

models = {
    'Linear': LinearRegression(),
    'Poly2': Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', LinearRegression())]),
    'Ridge': Ridge(alpha=3.8),
    'Lasso': Lasso(alpha=0.012, max_iter=3000),
    'SVR': SVR(kernel='rbf'),
    'DecisionTree': DecisionTreeRegressor(random_state=0),
    'RandomForest': RandomForestRegressor(n_estimators=500, random_state=0)
}

# 模型训练与评估
results = []

def evaluate_models(X_train, X_test, prefix):
    for name, model in models.items():
        # 训练模型
        model.fit(X_train, y_train)

        # 预测与评估
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results.append({
            'Model': f"{prefix}{name}",
            'RMSE': rmse,
            'R2': r2
        })


# 未规范化的数据评估
evaluate_models(X_train, X_test, 'Raw_')

# 规范化的数据评估
evaluate_models(X_train_scaled, X_test_scaled, 'Scaled_')

# 结果可视化
result_df = pd.DataFrame(results)

plt.figure(figsize=(18, 6))

# RMSE对比
plt.subplot(1, 2, 1)
sns.barplot(x='Model', y='RMSE', data=result_df)
plt.title('RMSE Comparison')
plt.xticks(rotation=45)

# R2对比
plt.subplot(1, 2, 2)
sns.barplot(x='Model', y='R2', data=result_df)
plt.title('R2 Score Comparison')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()