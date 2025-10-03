import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, spearmanr

class DataAnalyzer:
    def __init__(self, file_path, file_type):
        self.file_path = file_path
        self.file_type = file_type
        self.data = None

    def load_data(self):
        # 加载数据
        if self.file_type == 'csv':
            # 读取csv文件，将第一列作为索引列
            self.data = pd.read_csv(self.file_path, index_col=0)
        elif self.file_type == 'xls':
            # 读取xls文件，将第一列作为索引列
            self.data = pd.read_excel(self.file_path, index_col=0)
        elif self.file_type == 'txt':
            # 读取txt文件（以空格或制表符分隔），将第一列作为索引列
            self.data = pd.read_csv(self.file_path, sep='\s+', header=0, index_col=0)
        else:
            raise ValueError("不支持的文件类型。请使用‘csv’、‘xls’或‘txt’。")

    # 计算同类间的余弦相似度
    def calculate_cosine_similarity(self, class_labels):
        cosine_similarities = {}
        for class_label in np.unique(class_labels):
            class_data = self.data[class_labels == class_label].iloc[:, :-1]  # 排除标签列
            samples = class_data[:5]  # 取每类前5个样本
            similarities = cosine_similarity(samples)
            cosine_similarities[class_label] = similarities
        return cosine_similarities

    # 计算不同类间的余弦相似度
    def calculate_inter_class_cosine_similarity(self, class_labels):
        inter_class_similarities = {}
        unique_classes = np.unique(class_labels)
        for i in range(len(unique_classes)):
            for j in range(i + 1, len(unique_classes)):
                class1_data = self.data[class_labels == unique_classes[i]].iloc[:1, :-1]  # 取第一个类的1个样本
                class2_data = self.data[class_labels == unique_classes[j]].iloc[:5, :-1]  # 取第二个类的5个样本
                similarities = cosine_similarity(class1_data, class2_data)
                inter_class_similarities[f'{unique_classes[i]}_vs_{unique_classes[j]}'] = similarities
        return inter_class_similarities

    # 计算Pearson相关系数
    def calculate_pearson_correlation(self):
        return self.data.iloc[:, :-1].corr(method='pearson')

    # 计算Spearman相关系数
    def calculate_spearman_correlation(self):
        return self.data.iloc[:, :-1].corr(method='spearman')

    # 计算两两属性之间的相关系数
    def calculate_pairwise_correlation(self):
        return self.data.iloc[:, :-1].corr()


if __name__ == "__main__":
    # 初始化DataAnalyzer对象
    file_path = 'DataSet/iris.csv'
    file_type = 'csv'
    analyzer = DataAnalyzer(file_path, file_type)

    # 加载数据
    analyzer.load_data()

    # 获取标签列
    class_labels = analyzer.data.iloc[:, -1]

    # 计算同类间的余弦相似度
    cosine_similarities = analyzer.calculate_cosine_similarity(class_labels)
    print("同类间的余弦相似度：")
    for class_label, similarities in cosine_similarities.items():
        print(f"类别{class_label}:")
        print(pd.DataFrame(similarities, columns=[f"样本{i+1}" for i in range(5)], index=[f"样本{i+1}" for i in range(5)]))
        print()

    # 计算不同类间的余弦相似度
    inter_class_similarities = analyzer.calculate_inter_class_cosine_similarity(class_labels)
    print("不同类间的余弦相似度：")
    for comparison, similarities in inter_class_similarities.items():
        print(f"{comparison}:")
        print(pd.DataFrame(similarities, columns=[f"样本{i+1}" for i in range(5)], index=["样本1"]))
        print()

    # 计算Pearson相关系数
    pearson_corr = analyzer.calculate_pearson_correlation()
    print("Pearson相关系数：")
    print(pearson_corr)

    # 计算Spearman相关系数
    spearman_corr = analyzer.calculate_spearman_correlation()
    print("Spearman相关系数：")
    print(spearman_corr)

    # 计算两两属性之间的相关系数
    pairwise_corr = analyzer.calculate_pairwise_correlation()
    print("两两属性间的逐对相关性：")
    print(pairwise_corr)