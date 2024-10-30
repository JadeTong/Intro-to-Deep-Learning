'''
这节课要学习如何解读学习曲线和如何利用学习曲线来优化模型，具体来说，
是观察学习曲线，判断是否有过拟合和欠拟合的情况，并改正
'''

#%%
'''
我们可以将训练集中的信息分成两类：signal和noise，signal即可概括、可用于预测的信息；
而noise则只对训练集为准，为随机波动，non-informative，不能帮助预测
'''
'''
We train a model by choosing weights or parameters that minimize the loss on a training set.
You might know, however, that to accurately assess a model's performance,
we need to evaluate it on a new set of data, the validation data. 

在画训练集的损失函数曲线的时候，把验证集的损失函数曲线也加上，
训练集的损失曲线会在模型学习了signal或者学习了noise时，都会呈下降趋势；
但是对于验证集，损失曲线只有在模型学习到signal之后才呈下降趋势，
因为模型在训练集中学习到的噪声模式对新数据无效
所以，当两条损失曲线都呈下降趋势时，说明模型学习到了signal，但如果模型学习到了noise，
两条曲线之间会形成一个gap，这个gap的差距说明了模型学习到了多少noise

理想情况下，生成一个模型只学习到signal，不学习noise，但是这基本上不可能；
那就倾向于做一个交换，宁愿使模型学习更多signal，代价是同学习到更多的noise，
只要这个trade有利于我们，验证集的loss就会持续降低，
直到某一时间点，这个trade就会逆转形势，noise会学习到太多，验证集的loss就会上升。
'''

'''
欠拟合underfitting是模型因为还没学习到足够的signal，使得loss没有达到最低，
过拟合overfitting是模型因为学习太多的noise，验证集的数据无法被模型概括。
'''

#%% Capacity 能力
'''
一个模型的能力是指它能学习到的pattern的size和复杂度。
对于神经网络，这大部分是被神经元的数量和神经元的connection决定的。
如果你的网络看上去是欠拟合的，那就应该提高模型的capacity。

提高capacity可以通过‘使网络更宽’（即对已存在的layers增加units）或者‘使网络更深’（增加layers）
更宽的网络更容易学习线性关系，而更深的网络更偏向于非线性关系，要根据数据集做判断。

model = keras.Sequential([
    layers.Dense(16, activation='relu'),
    layers.Dense(1),
])

wider = keras.Sequential([
    layers.Dense(32, activation='relu'),   #加unit
    layers.Dense(1),
])

deeper = keras.Sequential([
    layers.Dense(16, activation='relu'),
    layers.Dense(16, activation='relu'),   #加层
    layers.Dense(1),
])

'''

#%%  Early Stopping 提早结束拟合进程
'''
前面提到如果模型过多学习noise，验证集的损失曲线就会反升，预防这种情况的发生，
我们可以在验证集损失曲线停止下降的时候就终止拟合。
'''
## 添加early stopping语句 这个程序将在每一个周期后运行一次

from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=20, # how many epochs to wait before stopping
    restore_best_weights=True,
)

'''
上面的参数定义是：
如果在上面的20个周期里面，验证集的损失向上对比改善不到0.001，
那就停止拟合并保留最佳模型
'''

#%% 例子 - Train a Model with Early Stopping
import pandas as pd
red_wine = pd.read_csv('C:/Users/Jade Tong/Desktop/KAGGLE/Intro to Deep Learning/datasets/red-wine.csv')

#切割训练集的验证集
df_train = red_wine.sample(frac=0.7, random_state=0)
df_valid = red_wine.drop(df_train.index)
df_train.head()

#%% 数据0-1缩放
max=df_train.max(axis=0)
min=df_train.min(axis=0)
df_train = (df_train-min)/(max-min)
df_valid = (df_valid-min)/(max-min)

#%% 分离特征和目标
X_train = df_train.drop('quality', axis=1)
X_valid = df_valid.drop('quality', axis=1)
y_train = df_train['quality']
y_valid = df_valid['quality'] 

#%% 建模
from keras import Sequential, callbacks
from keras.layers import Dense, Input

early_stopping = callbacks.EarlyStopping(
    min_delta=0.001,
    patience=20,
    restore_best_weights=True)

model = Sequential([
    Input([11]),
    Dense(512, activation='relu'),
    Dense(512, activation='relu'),
    Dense(512, activation='relu'),
    Dense(1)]) 

model.compile(optimizer='adam', loss='mae')

#%%
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,
    epochs=500,
    callbacks=[early_stopping], # put your callbacks in a list
    verbose=0,  # turn off training log
)

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot();
print("Minimum validation loss: {}".format(history_df['val_loss'].min()))




