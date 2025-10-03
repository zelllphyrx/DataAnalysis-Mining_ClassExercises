import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# 加载数据并预处理
train_data = pd.read_csv('DataSet/train.csv')
test_data = pd.read_csv('DataSet/test.csv')
gender_submission = pd.read_csv('DataSet/gender_submission.csv')
# # 查看训练集前几行数据
# train_data.head()
# 查看数据集基本信息
train_data.info()
# # 查看数值列的统计摘要
# train_data.describe()
# # 查看不同特征值的个数
# train_data['Survived'].value_counts()

# 合并测试集和真实标签（用于生存预测验证）
test_data = test_data.merge(gender_submission, on='PassengerId')


# 统一预处理函数
def preprocess_data(df):
    df = df.copy()
    # 填充缺失值
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Age'] = df['Age'].fillna(df['Age'].mean())  # 默认用均值填充
    # # 上船地点对预测结果不重要，将Embarked用众数填充
    # train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode())
    # # 对于标称属性舱号Cabin赋一个代表缺失的值‘U0’，其缺失可能代表并没有船舱
    # train_data.Cabin[train_data.Cabin.isnull()] = 'U0'
    # # 对Age字段进行填补
    # # train_data['Age'] = train_data['Age'].fillna(0) # 0填补或常数填补【适用于连续变量】,0可以替换为任意常数
    # train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())  # 均值填补【适用于连续变量，最好没有极端值】
    # # train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median()) # 中位数填补【适用于连续变量，可以有极端值】
    # # train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mode()) # 众数填补【使用与连续变量和离散变量，可以有极端值】
    # 处理异常值
    df = df[df['Fare'] < 400]
    # 删除无关特征
    df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
    # 编码分类变量
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    return df


# 预处理训练集和测试集
train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

# # 数据分析和可视化探索不同特征与生还率之间的关系
# # 性别与生还率的关系
# survived_sex = train_data.groupby('Sex')['Survived'].mean()
# plt.bar(survived_sex.index, survived_sex.values)
# plt.xlabel('Sex')
# plt.ylabel('Survival Rate')
# plt.title('Survival Rate by Sex')
# plt.show()
# # 船票等级与生还率的关系
# survived_pclass = train_data.groupby('Pclass')['Survived'].mean()
# plt.bar(survived_pclass.index, survived_pclass.values)
# plt.xlabel('Pclass')
# plt.ylabel('Survival Rate')
# plt.title('Survival Rate by Pclass')
# plt.show()
#
# # 机器学习模型预测乘客是否生还
# # 划分训练集和测试集
# X = train_data.drop('Survived', axis=1)
# y = train_data['Survived']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# # 训练模型
# model = LogisticRegression()
# # 预测并评估模型
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print('Accuracy:', accuracy)

# 验证方案A：用填充后的数据预测生存率
def validate_by_survival_prediction(train_df, test_df):
    # 划分特征和目标
    X_train = train_df.drop('Survived', axis=1)
    y_train = train_df['Survived']
    X_test = test_df.drop('Survived', axis=1)
    y_test = test_df['Survived']

    # 训练逻辑回归模型
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 预测并计算准确率
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"生存预测验证准确率: {accuracy:.4f}")


# 验证方案B：人为扣除数据后验证填充效果
def validate_by_manual_masking(df, target_col='Age', mask_ratio=0.05):
    # 随机隐藏部分数据作为真实值
    np.random.seed(42)
    mask = np.random.rand(df.shape[0]) < mask_ratio
    hidden_values = df.loc[mask, target_col].copy()
    df_masked = df.copy()
    df_masked.loc[mask, target_col] = np.nan

    # 用三种方法填充
    fill_methods = {
        'Mean': df_masked[target_col].mean(),
        'KNN': KNNImputer(n_neighbors=5).fit_transform(df_masked[[target_col]]).flatten(),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    # 计算每种方法的MSE
    results = {}
    for method, filler in fill_methods.items():
        if method == 'RandomForest':
            # 训练随机森林模型
            X_train = df_masked.dropna().drop(target_col, axis=1)
            y_train = df_masked.dropna()[target_col]
            filler.fit(X_train, y_train)
            # 预测缺失值
            X_test = df_masked[df_masked[target_col].isna()].drop(target_col, axis=1)
            filled_values = filler.predict(X_test)
            df_filled = df_masked.copy()
            df_filled.loc[df_masked[target_col].isna(), target_col] = filled_values
        else:
            df_filled = df_masked.copy()
            df_filled[target_col] = filler if method == 'Mean' else filler

        mse = mean_squared_error(hidden_values, df_filled.loc[mask, target_col])
        results[method] = mse

    # 输出结果
    print("\n人为扣除数据验证结果（MSE）:")
    for method, mse in results.items():
        print(f"{method}填充: {mse:.2f}")

# 执行验证
print("验证方案A：用填充后的数据预测生存率")
validate_by_survival_prediction(train_data, test_data)

print("\n验证方案B：人为扣除Age字段5%数据验证填充效果")
validate_by_manual_masking(train_data)
