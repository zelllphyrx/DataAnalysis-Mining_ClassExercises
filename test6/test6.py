# import pandas as pd
# from sklearn import tree
# from sklearn.datasets import load_wine
# from sklearn.model_selection import train_test_split
# Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine.data,wine.target,test_size=0.3)
# clf = tree.DecisionTreeClassifier(criterion="entropy")
# clf = clf.fit(Xtrain, Ytrain)
# score = clf.score(Xtest, Ytest) #返回预测的准确度accuracy
#
# feature_name = ['酒精','苹果酸','灰','灰的碱性','镁','总酚','类黄酮','非黄烷类酚类','花青素','颜色强度','色调','od280/od315稀释葡萄酒','脯氨酸']
# import graphviz
# dot_data = tree.export_graphviz(clf, out_file=".\Tree.dot"
#  ,feature_names = feature_name
#  ,class_names=["琴酒","雪莉","贝尔摩德"]
#  ,filled=True
#  ,rounded=True
#  )
#
# import re
# # 打开 dot_data.dot，修改 fontname="支持的中文字体"
# f = open("./Tree.dot", "r+", encoding="utf-8")
# open('./Tree_utf8.dot', 'w', encoding="utf-8").write(re.sub(r'fontname=helvetica', 'fontname="Microsoft YaHei"', f.read()))
# f.close()
#
# --------------------------------------------------------------
#
# from sklearn.tree import DecisionTreeRegressor
# data = np.array([[1, 1], [2, 3], [3, 3], [4, 6], [5, 6]])
# clf = DecisionTreeRegressor().fit(data[:, 0].reshape(-1, 1), data[:, 1])
# # 同样可以借助tree.plot_tree进行结果的可视化呈现
# plt.figure(figsize=(6, 2), dpi=150)
# tree.plot_tree(clf)

import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
data = np.array([[1, 1], [2, 3], [3, 3], [4, 6], [5, 6]])
clf = DecisionTreeRegressor().fit(data[:, 0].reshape(-1, 1), data[:, 1])
# 同样可以借助tree.plot_tree进行结果的可视化呈现
plt.figure(figsize=(6, 2), dpi=150)
tree.plot_tree(clf)