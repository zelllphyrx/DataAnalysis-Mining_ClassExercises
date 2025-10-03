import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

class NaiveBayesClassifier(object):
    def __init__(self):
        #self.label_prob表示每种标签在数据中出现的概率
        self.label_prob = {}
        #self.condition_prob表示每种类别确定的条件下各个特征出现的概率
        self.condition_prob = {}

    def fit(self, feature, label):
        '''
        对模型进行训练，需要将各种概率分别保存在self.label_prob和self.condition_prob中
        :param feature: 训练数据集所有特征组成的ndarray
        :param label:训练数据集中所有标签组成的ndarray
        '''
        row_num = len(feature)
        col_num = len(feature[0])
        unique_label_count = len(set(label))
        for c in label:
            if c in self.label_prob:
                self.label_prob[c] += 1
            else:
                self.label_prob[c] = 1

        for key in self.label_prob.keys():
            # 计算每种类别在数据集中出现的概率，拉普拉斯平滑
            self.label_prob[key] += 1
            self.label_prob[key] /= (unique_label_count+row_num)

            # 构建self.condition_prob中的key
            self.condition_prob[key] = {}
            for i in range(col_num):
                self.condition_prob[key][i] = {}
                for k in np.unique(feature[:, i], axis=0):
                    self.condition_prob[key][i][k] = 1


        for i in range(len(feature)):
            for j in range(len(feature[i])):
                if feature[i][j] in self.condition_prob[label[i]]:
                    self.condition_prob[label[i]][j][feature[i][j]] += 1

        for label_key in self.condition_prob.keys():
            for k in self.condition_prob[label_key].keys():
                #拉普拉斯平滑
                total = len(self.condition_prob[label_key].keys())
                for v in self.condition_prob[label_key][k].values():
                    total += v
                for kk in self.condition_prob[label_key][k].keys():
                    # 计算每种类别确定的条件下各个特征出现的概率
                    self.condition_prob[label_key][k][kk] /= total

    def predict(self, feature):
        '''
        对数据进行预测，返回预测结果
        :param feature:测试数据集所有特征组成的ndarray
        :return:
        '''
        result = []
        # 对每条测试数据都进行预测
        for i, f in enumerate(feature):
            # 可能的类别的概率
            prob = np.zeros(len(self.label_prob.keys()))
            ii = 0
            for label, label_prob in self.label_prob.items():
                # 计算概率
                prob[ii] = label_prob
                for j in range(len(feature[0])):
                    prob[ii] *= self.condition_prob[label][j][f[j]]
                ii += 1
            # 取概率最大的类别作为结果
            result.append(list(self.label_prob.keys())[np.argmax(prob)])
        return np.array(result)


train_data = pd.read_csv('DataSet/data_car/train.csv')
test_data = pd.read_csv('DataSet/data_car/test.csv')
train_data = np.array(train_data)
test_data = np.array(test_data)
# [,]: 逗号前代表多少行的数据，逗号后代表提取指定连续列的数据
feature = train_data[:, 0:6]
label = train_data[:, 6]
cs = NaiveBayesClassifier()
cs .fit(feature,label)
feature2 = test_data[:,0:6]
label2 = test_data[:, 6]
result = cs . predict(feature2)

crr = 0
for i in range(len(result)):
    if result[i] == label2[i]:
        crr += 1
cl = crr/len(label2)
print(cs.label_prob)
print(cs.condition_prob)
print("正确率",cl)
print(classification_report(label2, result))