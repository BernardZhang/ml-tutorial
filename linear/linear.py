"""
linear alogrithm to get weight of trainning data
"""
import csv
import numpy as np


DIMENSION = 3 # 线性模型的维度
learning_rate = 0.01 # 学习率

# 设计模型
def model(w, x):
    normal_x = np.append(np.array((1)), x)
    return np.sum(w * normal_x)

# 训练模型，求得最优解w
def train():
    # csv文件读取
    with open('./train.csv', 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        # 一般采用高斯随机初始化参数
        w = np.random.randn(DIMENSION)
        # 读取数据，并训练
        for row in csvreader:
            # 数据预处理
            *x, y = map(float, row)
            x = np.array(x)
            # 模型的输出
            predict_y = model(w, x)
            # 梯度下降法更新w
            w += -learning_rate * (predict_y - y) * np.append(np.array((1)), x)
            print(w)
if __name__ == '__main__':
    train()