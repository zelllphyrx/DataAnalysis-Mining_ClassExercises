import math
import time
import pandas as pd

def load_movie_data(filepath):
    """
    加载电影数据并转换为交易格式
    :param filepath: 文件路径
    :return: 交易数据集
    """
    # 读取CSV文件
    movies = pd.read_csv(filepath)

    # 将genres列拆分为多个二元特征
    genres_dummies = movies['genres'].str.get_dummies(sep='|')

    # 转换为交易列表格式
    transactions = []
    for _, row in genres_dummies.iterrows():
        transaction = [genre for genre, present in row.items() if present == 1]
        transactions.append(transaction)

    return transactions


def get_item_set(data):
    '''
    获取项的字典
    :param data: 数据集
    :return: 项的字典
    '''
    item_set = set()
    for d in data:
        item_set = item_set | set(d)
    return item_set


def apriori(item_set, data, min_support=0.50):
    '''
    获取频繁项集
    :param item_set: 项的字典
    :param data: 数据集
    :param min_support: 最小支持度
    :return: 非频繁项集，频繁项集，频繁项集支持度
    '''
    infrequent_list = []
    frequent_list = []
    frequent_support_list = []

    for n in range(1, len(item_set) + 1):
        c = []
        supports = []
        if len(frequent_list) == 0:
            for item in item_set:
                items = {item}
                support = calc_support(data, items)
                if support >= min_support:
                    c.append(items)
                    supports.append(support)
                else:
                    infrequent_list.append(items)
        else:
            for last_items in frequent_list[-1]:
                # items = last_items.copy()
                # items.add(item)
                # # 如果items的子集没有非频繁项集才计算支持度
                # if is_infrequent(infrequent_list, items) is False:
                #   support = calc_support(data, items)
                #   # 如果支持度大于等于最小支持度就为频繁项集
                #   if support >= min_support:
                #       c.append(items)
                #       supports.append(support)
                #   else:
                #       infrequent_list.append(items)
                for item in item_set:
                    # 保证组合顺序一致，避免重复
                    new_items = set(last_items)
                    new_items.add(item)
                    if len(new_items) == n and not is_infrequent(infrequent_list, new_items):
                        support = calc_support(data, new_items)
                        if support >= min_support:
                            if new_items not in c:
                                c.append(new_items)
                                supports.append(support)
                        else:
                            infrequent_list.append(new_items)

        frequent_list.append(c)
        frequent_support_list.append(supports)

        supports_rounded = [f"{s:.3f}" for s in supports] #保留三位小数
        print(f"{n}-项集: {c} , 支持度分别为: {supports_rounded}")

    return infrequent_list, frequent_list, frequent_support_list


def is_infrequent(infrequent_list, items):
    '''
    判断是否属于非频繁项集的超集
    :param infrequent_list: 非频繁项集列表
    :param items: 项集
    :return: 是否属于非频繁项集的超集
    '''
    for infrequent in infrequent_list:
        if infrequent.issubset(items):
            return True
    return False

def calc_support(data, items):
    '''
    计算 support
    :param data: 数据集
    :param items: 项集
    :return: 计算好的支持度
    '''
    cnt = 0
    for d in data:
        if items.issubset(d):
            cnt += 1
    return cnt / len(data)


def generate_rules(frequent_list, data, min_confidence=0.5):
    '''
    根据频繁项集和最小置信度生成规则
    :param frequent_list: 存储频繁项集的列表
    :param data: 数据集
    :param min_confidence: 最小置信度
    :return: 规则
    '''
    rule_key_set = set()
    rules = []
    for frequent in frequent_list:
        for items in frequent:
            if len(items) > 1:
                for n in range(1, math.ceil(len(items) / 2) + 1):
                    front_set_list = get_all_combine(list(items), n)
                    for front_set in front_set_list:
                        back_set = items - front_set
                        confidence = calc_confidence(front_set, items, data)
                        if confidence >= min_confidence:
                            support = calc_support(data, items)
                            lift = confidence / calc_support(data, back_set)
                            rule = (front_set, back_set, support, confidence, lift)
                            key = f'{front_set} ==> {back_set}'
                            if key not in rule_key_set:
                                rule_key_set.add(key)
                                rules.append(rule)
                                print(
                                    f"规则{len(rules)}: {key}, 支持度: {support:.3f}, 置信度: {confidence:.3f}, 提升度: {lift:.3f}")
    return rules


def get_all_combine(data_set, length):
    '''
    在指定数据集种获取指定长度的所有组合
    :param data_set: 数据集
    :param length: 指定的长度
    :return: 所有符合约束的组合
    '''

    def dfs(cur_index, cur_arr):
        if cur_index < len(data_set):
            cur_arr.append(data_set[cur_index])
            if len(cur_arr) == length:
                combine_list.append(set(cur_arr))
            else:
                for index in range(cur_index + 1, len(data_set)):
                    dfs(index, cur_arr.copy())

    combine_list = []
    for start_index in range(len(data_set)):
        dfs(start_index, [])
    return combine_list


def calc_confidence(front_set, total_set, data):
    '''
    计算规则 X==>Y 的置信度
    :param front_set: X
    :param total_set: X ∪ Y
    :param data: 数据集
    :return: 返回规则 X==>Y 的置信度
    '''
    front_cnt = 0
    total_cnt = 0
    for d in data:
        if front_set.issubset(d):
            front_cnt += 1
        if total_set.issubset(d):
            total_cnt += 1
    return total_cnt / front_cnt


if __name__ == '__main__':
    # 记录开始时间
    s = time.time()

    # 加载电影数据
    filepath = r'E:\DA&DM\DataSet\ml-latest-small\movies.csv'
    print(f"加载数据集: {filepath}")
    data = load_movie_data(filepath)
    print(f"项集条数: {len(data)}")
    # 获取项的字典
    item_set = get_item_set(data)
    print("电影类型总数:", len(item_set))
    print("类型分别为:", sorted(list(item_set))[:])

    # 根据 Apriori算法 获取 n-频繁项集
    print("\n生成频繁项集:")
    min_support = 0.05  # 由于电影类型较多，降低支持度阈值
    infrequent_list, frequent_list, frequent_support_list = apriori(item_set, data, min_support=min_support)

    # 生成规则
    print("\n生成关联规则:")
    min_confidence = 0.5  # 置信度阈值
    rule_set = generate_rules(frequent_list, data, min_confidence=min_confidence)

    # 输出总用时
    print("\n总用时:", round(time.time() - s, 2), "s")