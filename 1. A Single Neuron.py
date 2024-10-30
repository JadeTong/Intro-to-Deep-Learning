#深度学习是什么？

#神经网络是深度学习的标志模型，由神经元组成，单个神经元只进行一次计算，而神经网络就由神经元之间的关系复杂度来决定。

#一个神经元的数学表达法是y=wx+b, 没错就是线性回归模型，w是斜率，b是截距

#%%例子 麦片数据集
#用keras.Sequential来建立一个神经网络（多层堆叠来构建深度神经网络）（which creates a neural network as a stack of layers. We can create models like those above using a dense layer）
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#建立一个以'sugars', 'fiber', and 'protein'为自变量，'calories'为因变量的线性回归模型
# Create a network with 1 linear unit
model = Sequential([Dense(units=1, input_shape=[3])])

#‘units’定义输出，units=1即只有一个因变量；‘input_shape’定义输入的维度，input_shape=[3]即三个自变量




