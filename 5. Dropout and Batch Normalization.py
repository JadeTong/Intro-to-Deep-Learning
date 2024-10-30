'''
深度学习不止有密集层，还有其它类型的layers，
有些是跟dense layers一样是定义神经元之间的联系，
也有些是起预处理和转化作用的。

在这节课，我们将学习两种特殊的layers，它们本身不包含任何神经元，但是可以对模型有一些功能性的改善。
'''

#%%   Dropout
'''
dropout层可帮助改善过拟合，过拟合是因为模型过度学习到训练集中带有欺骗性的pattern，
而模型之所以会学习到这些pattern是依靠一种特殊的权重组合，
而正因为这个组合的特殊性，they tend to be fragile，移除一个，整个conspiracy就会破解

而这就是dropout背后的思想，
To break up these conspiracies, we randomly drop out some fraction 
of a layer's input units every step of training, 
使得神经网络更难地学习训练集中的spurious pattern.
Instead, it has to search for broad, general patterns, 
whose weight patterns tend to be more robust.
'''

#%%  Adding Dropout
'''
在需要应用的隐藏层前添加'Dropout(rate=  )来定义暂时关闭的神经元占比
Sequntial([
    Input(),
    ......,
    Dropout(rate=  ),
    Dense(),
    ......])

'''

#%%  Batch Normalization
'''
"batch normalization", which can help correct training that is slow or unstable.
A batch normalization layer looks at each batch as it comes in, 
first normalizing the batch with its own mean and standard deviation, 
and then also putting the data on a new scale with two trainable rescaling parameters.

通常是添加在优化步骤的，有时用在预测上；
带batchnorm的模型通常只需要更少的周期epoch就可以完成训练。
'''
#%%  Adding Batch Normalization
'''
batch normalization可以在模型的任何期间添加，可加在密集层后
Sequential([
    .......
    Dense(16, activation='relu'),
    BatchnNormalization(),
    .......])

也可加在layer和它的激活函数中间
Sequential([
    .......
    Dense(16),
    BatchnNormalization(),
    Activation('relu')
    .......])

如果作为第一层layer，它将作用为一个适配预处理器，类似sklearn的'StandardScaler'
'''

#%%  例子 在红酒数据上操作一下
import pandas as pd
red_wine = pd.read_csv('C:/Users/Jade Tong/Desktop/KAGGLE/Intro to Deep Learning/datasets/red-wine.csv')

df_train = red_wine.sample(frac=0.7, random_state=0)
df_valid = red_wine.drop(df_train.index)

X_train = df_train.copy()
y_train = X_train.pop('quality')

X_valid = df_valid.copy()
y_valid = X_valid.pop('quality')









