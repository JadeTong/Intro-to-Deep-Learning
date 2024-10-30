# setup plotting 
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')


#%%用到的数据集是‘fuel’，目标是根据给定的特征如机动车的引擎种类、生产年份来预测燃油经济
import pandas as pd
fuel = pd.read_csv('C:/Users/Jade Tong/Desktop/KAGGLE/Intro to Deep Learning/datasets/fuel.csv')
X = fuel.copy()
##目标变量是‘FE’
y = X.pop('FE')

#%% 预处理 用程序处理 将分类变量onehot成数值变量
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.model_selection import train_test_split

preprocessor = make_column_transformer(
    (StandardScaler(),
     make_column_selector(dtype_include=np.number)),
    (OneHotEncoder(sparse_output=False),
     make_column_selector(dtype_include=object)),)

X = preprocessor.fit_transform(X)
y = np.log(y) # log transform target instead of standardizing

input_shape = [X.shape[1]]
print("Input shape: {}".format(input_shape))
#Input shape: [50]

#%% 定义神经网络
from keras import Sequential
from keras.layers import Dense, Input

model = Sequential([
    Input(shape=input_shape),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),    
    Dense(64, activation='relu'),
    Dense(1)])

#%% 1)   添加损失函数MAE和优化器adam 
model.compile(optimizer='adam',loss='mae')

#%% 2)   训练模型 设定周期200次，batch size为128，input为X，target为y
history = model.fit(
    X,y,
    batch_size=128, 
    epochs=200
    ) 

# 将各周期的误差曲线画出来
history_df = pd.DataFrame(history.history)
# Start the plot at epoch 5. You can change this to get a different view.
history_df.loc[5:, ['loss']].plot()

#%% 3)   评价模型------‘如果模型训练周期更多，那是否误差会越来越小？’
'''
不一定，要看训练期间loss的移动路径，如果the learning curve已经趋向水平，那再加多周期也不会再优化；
如果损失曲线看上去还在下降，那就可以加多一些epoch
'''




















