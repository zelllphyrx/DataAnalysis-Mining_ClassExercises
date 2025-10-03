import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class DataLoader:
    def __init__(self, file_path, file_type='csv', normalize=False):
        """
        :param file_path: 文件路径
        :param file_type: 文件类型，支持 'csv', 'xls', 'txt', 'data'
        :param normalize: 是否对数据进行归一化
        """
        self.file_path = file_path
        self.file_type = file_type
        self.data = None
        self.features = None
        self.labels = None

        # 检查文件路径和权限
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"文件 {self.file_path} 不存在")
        if not os.access(self.file_path, os.R_OK):
            raise PermissionError(f"没有读取文件 {self.file_path} 的权限")

        self.load_data()
        if normalize:
            self.normalize_data()

    def load_data(self):
        # 加载数据
        if self.file_type == 'csv':
            # 读取csv文件
            self.data = pd.read_csv(self.file_path, index_col=0)
        elif self.file_type == 'xls':
            # 读取xls文件
            self.data = pd.read_excel(self.file_path)
        elif self.file_type == 'txt':
            # 读取txt文件（以空格或制表符分隔）
            self.data = pd.read_csv(self.file_path, sep='\s+', header=0)
        elif self.file_type == 'data':
            # 读取.data文件
            with open(self.file_path, 'r') as file:
                lines = file.readlines()
            # 解析数据
            data = []
            for line in lines:
                line = line.strip()
                if line:  # 忽略空行
                    parts = line.split(',')  # 以逗号分隔
                    if len(parts) > 1:  # 确保有数据
                        data.append(parts)
            # 转换为 DataFrame
            if data:
                self.data = pd.DataFrame(data, columns=[
                    'Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width', 'Species'
                ])
                # 将特征列转换为浮点数
                self.data.iloc[:, :-1] = self.data.iloc[:, :-1].astype(float)
            else:
                raise ValueError("No valid data found in the .data file.")
        else:
            raise ValueError("Unsupported file type. Use 'csv', 'xls', 'txt', or 'data'.")

        # 假设最后一列是标签，其余列是特征
        self.features = self.data.iloc[:, :-1]
        self.labels = self.data.iloc[:, -1]

    def calculate_central_tendency(self):
        # 计算中心位置度量：均值、中位数、众数
        mean = self.features.mean()
        median = self.features.median()
        mode = self.features.mode().iloc[0]  # 取第一行作为众数
        return mean, median, mode

    def calculate_dispersion(self):
        # 计算散布度量：极差、方差、标准差
        range_val = self.features.max() - self.features.min()
        variance = self.features.var()
        std_dev = self.features.std()
        return range_val, variance, std_dev

    def plot_histogram(self):
        # 绘制直方图（按类别区分颜色）
        n_features = self.features.shape[1]
        fig, axes = plt.subplots(n_features, 1, figsize=(10, 6 * n_features))
        if n_features == 1:
            axes = [axes]
        for i, col in enumerate(self.features.columns):
            for label in self.labels.unique():
                axes[i].hist(
                    self.features[self.labels == label][col],
                    bins=20,
                    alpha=0.5,
                    label=label
                )
            axes[i].set_title(f'Histogram of {col}')
            axes[i].legend()
        plt.tight_layout()
        plt.show()

    def plot_pareto(self):
        # 绘制帕累托图
        feature_sums = self.features.sum().sort_values(ascending=False)
        feature_sums.plot(kind='bar', figsize=(10, 6))
        plt.title('Pareto Chart of Features')
        plt.xlabel('Features')
        plt.ylabel('Sum')
        plt.show()

    def plot_scatter_matrix(self):
        # 绘制多维数据的散点图矩阵（按类别区分颜色），并添加图例
        # 创建散点图矩阵
        scatter_matrix = pd.plotting.scatter_matrix(
            self.features,
            figsize=(10, 10),
            c=pd.factorize(self.labels)[0],  # 按类别着色
            marker='o',
            hist_kwds={'bins': 50},
            s=50,
            alpha=0.8
        )

        # 添加图例
        unique_labels = self.labels.unique()
        num_labels = len(unique_labels)

        # 创建图例
        for i, label in enumerate(unique_labels):
            # 将类别索引归一化到 [0, 1] 范围内
            color = plt.cm.viridis(i / (num_labels - 1))  # 归一化并映射到颜色
            plt.scatter([], [], c=[color], label=label)  # 使用颜色列表

        # 调整图例位置和大小
        plt.legend(
            title='Species',
            bbox_to_anchor=(1.05, 1),  # 将图例放在图表右侧
            loc='upper left',
            borderaxespad=0.,
            fontsize='small'  # 设置图例字体大小
        )

        plt.suptitle('Scatter Matrix of Features')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # 加载数据
    file_path = 'DataSet/iris.csv'
    file_type = 'csv'  # 'csv', 'xls', 'txt', 'data' 选一

    try:
        data_loader = DataLoader(file_path, file_type)

        # 计算中心位置度量和散布度量
        mean, median, mode = data_loader.calculate_central_tendency()
        range_val, variance, std_dev = data_loader.calculate_dispersion()

        print("中心位置度量:")
        print(f"均值:\n{mean}")
        print(f"中位数:\n{median}")
        print(f"众数:\n{mode}")

        print("\n散布度量:")
        print(f"极差:\n{range_val}")
        print(f"方差:\n{variance}")
        print(f"标准差:\n{std_dev}")

        # 数据可视化
        data_loader.plot_histogram()  # 直方图
        data_loader.plot_pareto()     # 帕累托图
        data_loader.plot_scatter_matrix()  # 散点图矩阵
    except Exception as e:
        print(f"发生错误: {e}")