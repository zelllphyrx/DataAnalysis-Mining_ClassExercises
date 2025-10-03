from numpy import *
import numpy as np
from math import log
import operator
import pandas as pd
from sklearn.metrics import classification_report

def dataload():
    train_data = pd.read_csv('DataSet/data_car/train.csv')
    test_data = pd.read_csv('DataSet/data_car/test.csv')
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    x_test = test_data[:, 0:6]
    y_test = test_data[:, 6]
    return train_data,x_test, y_test

# H(p)  -sum (pi * log2(pi))
def calcShannonEnt(dataSet):  # 计算香农熵
    numEntries = len(dataSet)
    labelCounts = {}
    for featureVec in dataSet:  # the the number of unique elements and their occurance
        currentLabel = featureVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)  # log base 2
    return shannonEnt

def splitDataSet(dataSet, axis, value):  # 按照给定特征值划分数据集
    retDataSet = []
    for featureVec in dataSet:
        if featureVec[axis] == value:
            reducedFeatVec = featureVec[:axis]  # chop out axis used for splitting
            reducedFeatVec.extend(featureVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):  # 选择香农熵最高的特征值
    numFeatures = len(dataSet[0]) - 1  # the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0;
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value) #按照给定特征划分数据集
            prob = len(subDataSet) / float(len(dataSet))# 极大似然估计概率
            newEntropy += prob * calcShannonEnt(subDataSet)#∑pH(Y|X=xi) 条件熵的计算
        infoGain = baseEntropy - newEntropy  # calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):  # compare this to the best gain so far
            bestInfoGain = infoGain  # if better than current best, set to best
            bestFeature = i
    return bestFeature  # returns an integer

#返回出现次数最多的分类名称
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]  # 递归边界，当只有1种分类时停止返回
    if len(dataSet[0]) == 1: # 特征向量中所有的特征都已经用于划分
        return majorityCnt(classList) # 此时如果分类不唯一，则选取最有可能的类别

    bestFeat = chooseBestFeatureToSplit(dataSet) # 通过熵增益选取最优的特征，返回最优特征的编号
    bestFeatLabel = labels[bestFeat] # 获取最优特征
    myTree = {bestFeatLabel: {}} #将其作为当前子树的父节点
    del (labels[bestFeat]) # 从特征向量中删除该已用于划分的特征
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues) # 提取该特征所有能够的取值，构成集合(集合取值具有唯一性)
    for value in uniqueVals:
        # 生成父节点的子节点
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), labels[:])
        # 使用labels[:],需要新开辟内存放置删除某些特征后的特征向量
        # 如果直接使用labels进行传参，是公共用的同一内存空间，在生成树时无法递归返回，回溯时无法恢复状态
    return myTree


def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr) # 获取当前节点特征在特征向量中的下标
    key = testVec[featIndex] # 获取测试的特征向量中，该特征的具体值key
    if(key not in secondDict):
        valueOfFeat = -1
    else:
        valueOfFeat = secondDict[key] #进入第几个子树
    if isinstance(valueOfFeat, dict): #还没到达叶子节点，当前节点类型还是个字典
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel

def pred(inputTree, featlabels, xdata, ydata):
    m = shape(ydata)[0]
    sum = 0
    for i in range(m):
        res = classify(inputTree, featlabels, xdata[i])
        if(res == ydata[i]):
            sum = sum + 1

    return sum/m


if __name__ == '__main__':
    mydata, test_x, test_y = dataload()
    mydata = np.ndarray.tolist(mydata)
    labels = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
    mytree = createTree(mydata, labels.copy())

    # 预测测试集结果
    y_pred = []
    for x in test_x:
        y_pred.append(classify(mytree, labels, x))

    # 打印准确率和分类报告
    print("正确率为：", pred(mytree, labels, test_x, test_y))
    from sklearn.metrics import classification_report

    print(classification_report(test_y, y_pred))