# Setup plotting
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')

# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',titleweight='bold', titlesize=18, titlepad=10)

#%%
#数据集‘Red Wine Quality’含有关于大约1600支葡萄牙红酒的理化数据，包括盲测的红酒品质评分
import pandas as pd

red_wine = pd.read_csv('C:/Users/Jade Tong/Desktop/KAGGLE/Intro to Deep Learning/datasets/red-wine.csv')
print(red_wine.shape)  #(1599, 12) 1599条记录，11个特征，目标为quality
red_wine.head()

#%%   1)    定义模型的input_shape 哪几个是自变量
input_shape = [11]  #除了目标，其它都定义为自变量

#%%   2)    定义一个线性模型
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

model = Sequential([Input(shape=(11,)), 
                    Dense(units=1)])

#%%   3)    加权值
w, b = model.weights
print("Weights\n{}\n\nBias\n{}".format(w.numpy(), b.numpy()))

#%%
import tensorflow as tf
import matplotlib.pyplot as plt

model = Sequential([Input(shape=(1,)),
                    Dense(units=1)])

x = tf.linspace(-1.0, 1.0, 100)
y = model.predict(x)

plt.figure(dpi=100)
plt.plot(x, y, 'k')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel("Input: x")
plt.ylabel("Target y")
w, b = model.weights # you could also use model.get_weights() here
plt.title("Weight: {:0.2f}\nBias: {:0.2f}".format(w[0][0], b[0]))
plt.show()













