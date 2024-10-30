import tensorflow as tf
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)


#%%  读取数据
import pandas as pd
concrete = pd.read_csv('C:/Users/Jade Tong/Desktop/KAGGLE/Intro to Deep Learning/datasets/concrete.csv')
concrete.head()

#%%     1) Input Shape
'''数据集中的目标因变量为'CompressiveStrength'，其它特征都系inputs'''
input_shape = [8]

#%%     2) Define a Model with Hidden Layers
'''建立三个hidden layers，每一个有512个units和ReLu激活函数；最终输出为1个unit且没有激活'''
from keras import Sequential 
from keras.layers import Dense, Input 
model = Sequential([Input(shape=(8,)),
                    Dense(512, activation='relu'),
                    Dense(512,activation='relu'),
                    Dense(512,activation='relu'),
                    Dense(1)])

#%%     3)  Activation Layers
'''如果需要在Dense layer和修正函数中间插入其它layer，那也可以将两者分开写，例如
    
model = Sequential([
        layers.Dense(32, input_shape=[8]),
        layers.Activation('relu'), 
        layers.Dense(32),
        layers.Activation('relu'), 
        layers.Dense(1),])
'''


















