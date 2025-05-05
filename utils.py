import tushare as ts
import numpy as np


def get_client():
    pro = ts.pro_api('7d5be2cd9be532a8ad51c542c499c59c275db649a98f7e159b2d68a7')
    return pro


def check_macd_positive(numners):
    last_neg_index = -1
    for i, num in enumerate(numners):
        if num < 0:
            last_neg_index = i
        # 没有负数，或者最后一个负数在末尾时返回False
        if last_neg_index == -1 or last_neg_index == len(numners) -1:
            return False

        # 提取转正后的子序列
        sublist = numners[last_neg_index+1:]
        # print(sublist)

        # 检查第一个元素是否为正
        if sublist[0] < 0:
            return False

        # 趋势向上
        if sublist[-1] < np.mean(sublist):
            return False
        return True


def is_increaseing_trend_up(sequence):
    if sequence[-1] < np.mean(sequence):
        return False

    if len(sequence) < 2:
        return False

    x = np.arange(len(sequence))
    y = np.array(sequence)

    slope = np.polyfit(x, y, 1)[0]
    return slope > 0

# ---macd test
# numbers_list = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
# result = check_macd_positive(numbers_list)
# print(result)


# ---boll test
data_list = [1, 1, 2, 1, 3, 2, 1]
print(is_increaseing_trend_up(data_list))
