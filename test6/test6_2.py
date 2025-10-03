import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# 1. 数据加载和预处理
# 读取数据文件
wine_data = pd.read_csv('E:\DA&DM\DataSet\wine\wine.data', header=None)

# 设置列名
feature_names = ['类别', '酒精', '苹果酸', '灰', '灰的碱性', '镁', '总酚',
                 '类黄酮', '非黄烷类酚类', '花青素', '颜色强度', '色调',
                 'od280/od315稀释葡萄酒', '脯氨酸']

wine_data.columns = feature_names

# 分离特征和目标变量
X = wine_data.iloc[:, 1:]  # 所有特征列
y = wine_data.iloc[:, 0]   # 类别列

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 构建决策树模型
clf = tree.DecisionTreeClassifier(criterion="entropy", random_state=42, max_depth=3)  # 限制深度便于可视化
clf = clf.fit(X_train, y_train)

# 4. 模型评估
# 在测试集上进行预测
y_pred = clf.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"准确率: {accuracy:.4f}")

# 计算精确率和召回率
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
print(f"精确率: {precision:.4f}")
print(f"召回率: {recall:.4f}")

# 打印分类报告
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=["琴酒", "雪莉", "贝尔摩德"]))

# 打印混淆矩阵
print("混淆矩阵:")
print(confusion_matrix(y_test, y_pred))

# 5. 使用plot_tree可视化决策树
plt.figure(figsize=(20, 10))

# 中文特征名称
feature_name_cn = ['酒精', '苹果酸', '灰', '灰的碱性', '镁', '总酚',
                  '类黄酮', '非黄烷类酚类', '花青素', '颜色强度', '色调',
                  'od280/od315稀释葡萄酒', '脯氨酸']

# 类别名称
class_names = ["琴酒", "雪莉", "贝尔摩德"]

# 绘制决策树
tree.plot_tree(clf,
               feature_names=feature_name_cn,
               class_names=class_names,
               filled=True,
               rounded=True,
               fontsize=10)

plt.title("葡萄酒分类决策树")
plt.savefig('wine_decision_tree_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. 特征重要性分析
# 获取特征重要性
feature_importances = clf.feature_importances_

# 创建特征重要性DataFrame
features_df = pd.DataFrame({
    '特征': feature_name_cn,
    '重要性': feature_importances
}).sort_values('重要性', ascending=False)

print("\n特征重要性排序:")
print(features_df)

import re
# 打开 dot_data.dot，修改 fontname="支持的中文字体"
f = open("./Tree.dot", "r+", encoding="utf-8")
open('./Tree_utf8.dot', 'w', encoding="utf-8").write(re.sub(r'fontname=helvetica', 'fontname="Microsoft YaHei"', f.read()))
f.close()

# 可视化特征重要性
plt.figure(figsize=(10, 6))
plt.barh(features_df['特征'], features_df['重要性'])
plt.xlabel('特征重要性')
plt.title('决策树特征重要性')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
plt.show()