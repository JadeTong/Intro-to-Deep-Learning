#load数据 spotify  任务是从歌曲的音乐特征，预测歌曲的人气
import pandas as pd
spotify = pd.read_csv('C:/Users/Jade Tong/Desktop/KAGGLE/Intro to Deep Learning/datasets/spotify.csv')

#%% 处理变量
X = spotify.copy().dropna()   ##去除缺失值
y = X.pop('track_popularity')

#%% 分组，将同一个歌手的歌曲分到一个组里，切割训练集和验证集的时候会将同一组的歌分在同一数据集里面
'n_splits指定切割的次数，train_size指定训练集切割多少数据，也可用test_size指定'
artists = X.track_artist
from sklearn.model_selection import GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, train_size=0.75, random_state=0) 
# 使用 next() 从迭代器中获取第一个（也是唯一一个）划分
train_idx, valid_idx = next(gss.split(X, y, artists))

X_train=X.iloc[train_idx]
y_train=y.iloc[train_idx]

X_valid=X.iloc[valid_idx]
y_valid=y.iloc[valid_idx]

#%% 指定数值型特征
feature_num = ['danceability', 'energy', 'key', 'loudness', 'mode',
                'speechiness', 'acousticness', 'instrumentalness',
                'liveness', 'valence', 'tempo', 'duration_ms']

feature_cat = ['playlist_genre']

#标准化缩放数值型变量，独热编码分类变量
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer

preprocessor = make_column_transformer(
    (StandardScaler(),feature_num),
    (OneHotEncoder(),feature_cat))

X_train = preprocessor.fit_transform(X_train)
X_valid = preprocessor.transform(X_valid)
y_train = y_train / 100 # popularity is on a scale 0-100, so this rescales to 0-1.
y_valid = y_valid / 100

input_shape = [X_train.shape[1]]
print("Input shape: {}".format(input_shape))

#%% 先建立一个简单的神经网络，一个具有low capacity的线性模型
from keras import Sequential
from keras.layers import Dense, Input
model = Sequential([Input(input_shape),
                    Dense(1)])

#加优化器和损失函数
model.compile(optimizer='adam',loss='mae')

#输出历史loss
history = model.fit(X_train,y_train,
                    batch_size=512,
                    epochs=50,
                    validation_data=(X_valid,y_valid),
                    verbose=False)   #是否输出记录，要画图就不输出

history_df = pd.DataFrame(history.history)
history_df.plot()  
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()));
'Minimum Validation Loss: 0.1960'

#损失曲线为‘hockey stick’shape，在epoch为10时趋向平滑，所以从epoch=10，重新画损失曲线
history_df.loc[10:, ['loss', 'val_loss']].plot()  #从index=10开始话画

#%%  1) Evaluate Baseline
'what do you think? Would you say this model is underfitting, overfitting, just right?'
#验证损失和训练损失之间的gap相当小，损失曲线并没有反升，所以应该不可能是过拟合
#试下增加capacity

#%%  2) Add Capacity 
#增加两个hidden layers，第一个带有128个units，第二个有64个units
model = Sequential([
    Input(input_shape),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)])

model.compile(optimizer='adam',loss='mae')
history=model.fit(X_train,y_train,
                  batch_size=512,
                  epochs=50,
                  validation_data=(X_valid,y_valid),
                  verbose=False)
history_df = pd.DataFrame(history.history)
history_df.plot()
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()))
'Minimum Validation Loss: 0.1942'
#从损失曲线图中看出，验证集的损失曲线在第十几个周期就开始反升了，但是此时训练集的损失曲线还在下降，
#这就说明这个神经网络开始过拟合，学习到太多noise，我们需要采取某些措施去预防它，
#可以是减少units，也可以是early stopping

#%%  3) Define Early Stopping Callback
from keras import callbacks
early_stopping = callbacks.EarlyStopping(
    min_delta=0.001,
    patience=5,
    restore_best_weights=True)

model = Sequential([
    Input(input_shape),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)])

model.compile(optimizer='adam',loss='mae')
history=model.fit(X_train,y_train,
                  batch_size=512,
                  epochs=50,
                  validation_data=(X_valid,y_valid),
                  callbacks=[early_stopping])
history_df = pd.DataFrame(history.history)
history_df.plot()
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()))


