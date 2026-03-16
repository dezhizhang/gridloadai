import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from utils.log import Logger
from utils.common import data_preprocessing
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error,mean_absolute_error
import joblib


plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class PowerLoadModel:
    def __init__(self):
        """"初始化信息"""
        # 1. 拼接日志文件名
        logfile_name = "train" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        # 2. 创建日志对像
        self.logfile = Logger("../",logfile_name).get_logger()

        # 3. 添加日志信息
        self.logfile.info("开始创建电力负荷模型类对像了")

        # 4. 获取数据源
        self.data_source = data_preprocessing()




if __name__ == '__main__':
    PowerLoadModel()