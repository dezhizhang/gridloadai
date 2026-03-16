import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from utils.log import Logger
from utils.common import data_preprocessing
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class PowerLoadModel:
    def __init__(self):
        """"初始化信息"""
        # 1. 拼接日志文件名
        logfile_name = "train" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        # 2. 创建日志对像
        self.logfile = Logger("../", logfile_name).get_logger()

        # 3. 添加日志信息
        self.logfile.info("开始创建电力负荷模型类对像了")

        # 4. 获取数据源
        self.data_source = data_preprocessing()


def ana_data(data):
    """查看数据分布情况"""
    # 1. 防止修改原数据拷贝一次
    ana_data = data.copy()

    # 2. 绘制图表分而情况
    fig = plt.figure(figsize=(20, 40))
    ax1 = fig.add_subplot(411)
    ax1.hist(ana_data['power_load'], bins=100)
    ax1.set_title('负荷整体分布情况')
    ax1.set_xlabel('负荷')
    # plt.show()

    # 2. 新增一列充当小时
    ana_data["hour"] = ana_data['time'].str[11:13]
    hour_load_mean = ana_data.groupby(['hour'], as_index=False)['power_load'].mean()

    # 3. 绘制图表
    ax2 = fig.add_subplot(412)
    ax2.plot(hour_load_mean['hour'],hour_load_mean['power_load'])
    ax2.set_title('各个小时平均负荷趋势')
    ax2.set_xlabel('小时')
    plt.show()




if __name__ == '__main__':
    pd = PowerLoadModel()

    ana_data(pd.data_source)
