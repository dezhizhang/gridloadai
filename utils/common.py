import numpy as np
import pandas as pd


def data_preprocessing():
    """数据处理函数"""
    # 1. 加载数据
    data = pd.read_csv('../data/train.csv')

    # 2. 时间格式化转化为:'%Y-%m-%d %H:%M:%S'
    data['time'] = pd.to_datetime(data['time']).dt.strftime('%Y-%m-%d %H:%M:%S')

    # 4. 按照时间升序排序
    data.sort_values("time", ascending=True, inplace=True)

    # 5. 对数据进行去重处理
    data.drop_duplicates(inplace=True)

    return data


if __name__ == '__main__':
    data_preprocessing()
