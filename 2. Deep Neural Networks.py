#主要思想是模块化，building up a complex network from simpler functional units

#%% Layers
'神经网络通常将神经元组织成多个‘层’ layers，当我们将有common set of inputs的线性units组合起来，就得到一个密集层dense layer'
'可以将神经网络的各层视为在做相对简单的transformation；通过各层layers，神经网络将inputs经过越来越复杂地转化'

'In Keras, a layer can be, essentially, any kind of data transformation.'

#%%  Activation Function 激活函数
'没有激活函数，神经网络只能学习线性关系'
'In order to fit curves, we need to use activation functions.'

'激活函数是一些被用于每一层的outputs（即activations）的函数，最常见的整流函数/修正函数rectifier function [max(0,x)]'
'修正函数max(0,x)，即把那层的神经元output中正数结果直接输出，将负数结果修正为零后输出'
'当把修正器attach到一个线性unit，将得到一个rectified linear unit(ReLu)，所以通常称修正函数为ReLu函数'
'Applying a ReLU activation to a linear unit means the output becomes max(0, w * x + b)'

#%%  Stacking Dense Layers 堆积密集层
'在最终输出y之前的layers有时会被叫做hidden因为我们不会直接看见这些层的outputs'
'如果最终输出y没有应用激活函数，那就说明我们可以将这个神经网络应用于a regression task, where we are trying to predict some arbitrary numeric value'

#%%  Building Sequential Models 建立有序模型
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    # the hidden ReLU layers
    layers.Dense(units=4, activation='relu', input_shape=[2]),
    layers.Dense(units=3, activation='relu'),
    # the linear output layer 
    layers.Dense(units=1),
])


model = keras.Sequential([layers.Input(shape=(2,)), 
                          layers.Dense(units=4, activation='relu'),
                          layers.Dense(units=3, activation='relu'),
                          layers.Dense(units=1),])











