# 随机梯度下降  Stochastic Gradient Descent
'''
前两节课中学会了用多层堆叠密集层来建立神经网络，模型建立起初所有的权重weights都是随机的，因为这个神经网络还没开始学习。
在这节课中，我们要学习如何训练神经网络和神经网络是怎样学习的。
除了准备训练集外，还需要两样东西：
1. 损失函数，用来测量模型的预测准确度
2. 优化器，用来调整参数
'''

#%% The Loss Function
'''
loss function tells a network what problem to solve.
损失函数测量目标真值和模型估计值两个点之间的距离；
不同的需求用不同的损失函数，对于线性回归模型，常见的损失函数是平均绝对误差MAE即avg(|y_true-y_pred|)
除MAE之外，其它回归模型的损失函数也可能是均方误差MSE或者Huber loss
在模型训练期间，模型会根据损失函数的指引来自动找到使误差最小的那组权重
'''

#%% The Optimize
'''
所有的优化算法都属于随机梯度下降，它们是逐步去训练模型的迭代算法
1. 从训练集中抽取一些数据sample，拟合模型得出估计值
2. 测量真值和估计值之间的误差loss
3. 调整权重以缩小误差 
4. 重复上述操作直到误差不能再降低

每一个迭代的sample叫做'minibatch'，完成整个训练集叫做周期'epoch'，
如果Epoch数量太少，网络有可能发生欠拟合（即对于定型数据的学习不够充分）；
如果Epoch数量太多，则有可能发生过拟合（即网络对定型数据中的“噪声”而非信号拟合）
'''

#%% Learning Rate and Batch Size
'''
A smaller learning rate means the network needs to see more minibatches before its weights converge to their best values.
学习率和minibatch的size are the two parameters that have the largest effect on how the SGD training proceeds.
'''
'''
在定义了模型之后，可以用.compile()语句加一个损失函数和优化器
   model.compile(optimizer="adam",loss="mae") #adam是SGD里面自动调节参数的算法
'''

#%%  例子 红酒质量
import pandas as pd
from IPython.display import display

red_wine = pd.read_csv('C:/Users/Jade Tong/Desktop/KAGGLE/Intro to Deep Learning/datasets/red-wine.csv')

# Create training and validation splits
df_train = red_wine.sample(frac=0.7, random_state=0)
df_valid = red_wine.drop(df_train.index)
display(df_train.head(4))

# Scale to [0, 1]
max_ = df_train.max(axis=0)
min_ = df_train.min(axis=0)
df_train = (df_train - min_) / (max_ - min_)
df_valid = (df_valid - min_) / (max_ - min_)

# Split features and target
X_train = df_train.drop('quality', axis=1)
X_valid = df_valid.drop('quality', axis=1)
y_train = df_train['quality']
y_valid = df_valid['quality']

#%% 这个神经网络有多少input
print(X_train.shape)  #(1119, 11) 
#11个inputs

#%% 在前一节课上，建立了一个三层，每层有512个神经元的模型
from keras import Sequential 
from keras.layers import Dense, Input 
model = Sequential([Input(shape=(11,)),
                    Dense(512, activation='relu'),
                    Dense(512,activation='relu'),
                    Dense(512,activation='relu'),
                    Dense(1)])

#%%
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=[11]),
    layers.Dense(512, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(1),
])

#%%
model.compile(optimizer='adam',loss='mae')

#%% start training
#设置batch size为256，重复整个周期epoch 10次
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,
    epochs=10,)

#%% plot the loss
import pandas as pd
# convert the training history to a dataframe
history_df = pd.DataFrame(history.history)
# use Pandas native plot method
history_df['loss'].plot();















