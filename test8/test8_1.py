import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.metrics import silhouette_score

# 1. 数据加载
df = pd.read_csv("DataSet/car_price.csv")

# 2. 特征处理
df['doornumber'] = df['doornumber'].map({'two': 2, 'four': 4})

# 3. 数据建模
df1 = df[['symboling', 'wheelbase', 'horsepower', 'price', 'doornumber']]
cols = ['carbody', 'drivewheel']

# 4. 数据转换
model_ohe = OneHotEncoder(sparse=False)
ohe_matrix = model_ohe.fit_transform(df[cols])
model_scaler = MinMaxScaler()
data_scaled = model_scaler.fit_transform(df1)
X = np.hstack((data_scaled, ohe_matrix))

# 5. 聚类分析
model_kmean = KMeans(n_clusters=5)
cluster_labels_ = model_kmean.fit_predict(X)
cluster_labels = pd.DataFrame(cluster_labels_, columns=['clusters'])
merge_data = pd.concat((df, cluster_labels), axis=1)
merge_data.to_csv('merge_data.csv')

# 6. 竞品分析
price_bins = [0, 7000, 7500, 8000, 8500, 9000, 310000]
price_labels = ['7000以下', '7000-7500', '7500-8000', '8000-8500', '8500-9000', '9000以上']
df['price_range'] = pd.cut(df['price'], bins=price_bins, labels=price_labels)

