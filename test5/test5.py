import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# 设置pandas输出表格的属性
pd.options.display.max_colwidth = 100
pd.options.display.width = None
# pd.set_option('display.width',None) #设置数据展示宽度，展开读取数据时默认的省略号部分

# 读取原始数据
movies = pd.read_csv(r'E:\DA&DM\DataSet\ml-latest-small\movies.csv')
# print(movies.head(10))

# 第一步将原始数据转化为标准格式
movies_standard = movies.drop('genres', 1).join(movies.genres.str.get_dummies())
print(movies_standard.head(10))
# 一共包含9742部电影，一共有20种不同的电影类型（有2列是ID和电影名）
print(movies_standard.shape)  # (9742, 22)

# 利用mlxtend提供的apriori算法函数得到频繁项集，其中设置最小支持度为0.05
movies_standard.set_index(['movieId', 'title'], inplace=True)
frequent_item_sets = apriori(movies_standard, min_support=0.05, use_colnames=True)
print(frequent_item_sets)

# # 计算规则，并设置提升度阈值为1.25（返回各个指标的数值）
# rules = association_rules(frequent_item_sets, metric='lift', min_threshold=1.25)
# print(rules)

# # 对lift降序排序，查看lift较大的是哪些规则
# rules_sort = rules.sort_values(by=['lift'], ascending=False)
# print(rules_sort)

# 计算规则，并设置置信度阈值为0.3（返回各个指标的数值）
rules = association_rules(frequent_item_sets, metric='confidence', min_threshold=0.3)
print(rules)

# 对confidence降序排序，查看confidence较大的是哪些规则
rules_sort = rules.sort_values(by=['confidence'], ascending=False)
print(rules_sort)
