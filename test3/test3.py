import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 加载数据并预处理
train_data = pd.read_csv('DataSet/train.csv',header=0)
test_data = pd.read_csv('DataSet/test.csv',header=0)
gender_submission = pd.read_csv('DataSet/gender_submission.csv',header=0)
# # 查看训练集前几行数据
# train_data.head()
# 查看数据集基本信息
# train_data.info()
# # 查看数值列的统计摘要
# train_data.describe()
# # 查看不同特征值的个数
# train_data['Survived'].value_counts()

# 上船地点对预测结果不重要，将Embarked用众数填充
train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode())
# 对于标称属性舱号Cabin赋一个代表缺失的值‘U0’，其缺失可能代表并没有船舱
train_data.Cabin[train_data.Cabin.isnull()]='U0'
# 对Age字段进行填补
# train_data['Age'] = train_data['Age'].fillna(0) # 0填补或常数填补【适用于连续变量】,0可以替换为任意常数
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean()) # 均值填补【适用于连续变量，最好没有极端值】
# train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median()) # 中位数填补【适用于连续变量，可以有极端值】
# train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mode()) # 众数填补【使用与连续变量和离散变量，可以有极端值】
# 处理异常值
train_data = train_data[train_data['Fare'] < 400]
# 删除不需要的特征
train_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
# 特征选择
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
target = 'Survived'
X = train_data[features]
y = train_data[target]

# 数据分析和可视化探索不同特征与生还率之间的关系
# 性别与生还率的关系
survived_sex = train_data.groupby('Sex')['Survived'].mean()
plt.bar(survived_sex.index, survived_sex.values)
plt.xlabel('Sex')
plt.ylabel('Survival Rate')
plt.title('Survival Rate by Sex')
plt.show()
# 船票等级与生还率的关系
survived_pclass = train_data.groupby('Pclass')['Survived'].mean()
plt.bar(survived_pclass.index, survived_pclass.values)
plt.xlabel('Pclass')
plt.ylabel('Survival Rate')
plt.title('Survival Rate by Pclass')
plt.show()

# 机器学习模型预测乘客是否生还
# 划分训练集和测试集
X = train_data.drop('Survived', axis=1)
y = train_data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 训练模型
model = LogisticRegression()
# 预测并评估模型
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)