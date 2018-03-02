#! -*- coding: utf-8 -*-
"""
generate dataset for training
"""
from __future__ import print_function
import csv
import numpy as np

# 设定训练数据的系数，然后机器学习把改参数训练出来
W = np.array((0.34, -1.21))
# B参数
B = 0.42
# 噪声系数
NOISE_WEIGHT = 1.0

# 设计一个带有噪声的线性模型, 可自定调整噪声系数
def model(x):
    return np.sum(W * x) + B + np.random.randn() * NOISE_WEIGHT

# 产生训练数据, 数据存储格式为 x0, x1, y
# size: 产生size条训练数据
def gen_dataset(size):
    with open('train.csv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        for _ in range(size):
            # 随机产出输入数据, 范围为0~10
            train_x = 10 * np.random.random_sample(2)
            # 获取模型的结果
            train_y = model(train_x)
            # 写入csv文件
            csvwriter.writerow(list(train_x) + [train_y,])

if __name__ == '__main__':
    print('generate dataset for trainning')
    gen_dataset(10000)
    print('end')