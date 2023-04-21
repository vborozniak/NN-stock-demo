import time
import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.callbacks import EarlyStopping

def create_date_table2(start='2010-01-01', end='2019-12-31'):
    df = pd.DataFrame({"Date": pd.date_range(start, end)})
    return df

list = ['WMT', 'FMC', 'ALG', 'AGCO','CALM','IBA']
checkdf = create_date_table2()
tix = pd.DataFrame()


for ticker in list:
    period1 = int(time.mktime(datetime.datetime(2010, 1, 1, 23, 59).timetuple()))
    period2 = int(time.mktime(datetime.datetime(2019, 12, 31, 23, 59).timetuple()))
    interval = '1d' # 1d, 1m
    
    query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
    
    df = pd.read_csv(query_string, parse_dates=(['Date']))
    close = df.iloc[:,[0,4]]
    close.rename(columns={"Close": ticker}, inplace = True)
    tix = tix.append(close, ignore_index = True)
    
dfall = checkdf.merge(tix, how = "left", on = ['Date'])
teset = dfall.groupby(['Date']).first().fillna('')
#teset.ALG.astype(bool)
ts = teset[teset['ALG'].astype(bool)]
ts = ts.astype(str).astype(float)

pearsoncorr = ts.corr(method='pearson')#ALG highest
sb.heatmap(pearsoncorr, 
            xticklabels=pearsoncorr.columns,
            yticklabels=pearsoncorr.columns,
            cmap='RdBu_r',
            annot=True,
            linewidth=0.5)

ts.plot(figsize=(15,4))
ts.plot(subplots=True, figsize=(15,6))


ts = ts.reset_index()
#ts.insert(1, 'MM-DD',ts['Date'].dt.strftime('%m%d').astype(float))#Making datetime index to only show Month-Day (test)
ts = ts.set_index('Date')
validation_data = ts[(ts.index >"2018-12-31")]
ts = ts[(ts.index <"2019-01-01")]

#%%train|test split, normalization and plot loss
stck = 'ALG'
train_dataset = ts.sample(frac=0.8, random_state=0)
test_dataset = ts.drop(train_dataset.index)

train_features = train_dataset.copy()
test_features = test_dataset.copy()
val_features = validation_data.copy()

train_labels = train_features.pop(stck)
test_labels = test_features.pop(stck)
val_labels = val_features.pop(stck)

normalizer = tf.keras.layers.Normalization(
    axis=-1, mean=None, variance=None, invert=False)
normalizer.adapt(np.array(train_features))

def plot_loss(history):
  plt.plot(history.history['loss'], label='training_loss')
  plt.plot(history.history['val_loss'], label='validation_loss')
  plt.ylim([0, 25])
  plt.xlabel('Epoch')
  plt.ylabel('Error')
  plt.legend()
  plt.grid(True)
  
##%%MODEL SECTION
def build_and_compile_model(norm):
  model = tf.keras.Sequential([
      norm,
      layers.Dense(16, activation='relu'),    
      layers.Dropout(0.2), 
      layers.Dense(32, activation='relu'),
      layers.Dense(16, activation='relu'),
      layers.Dense(1,activation='relu')
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model

dnn_model = build_and_compile_model(normalizer)
#dnn_model.summary()
early_stopping = EarlyStopping(monitor = 'val_loss',patience = 12, min_delta = 1)
history = dnn_model.fit(
    train_features,
    train_labels,
    validation_split=0.2,
    verbose=1, epochs=50, batch_size = 16,callbacks = [early_stopping] )

plot_loss(history)

test_results = {}
test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)
#dnn_model.save('dnn_model_ALG')

test_predictions = dnn_model.predict(test_features).flatten()
val_predictions = dnn_model.predict(val_features).flatten()
#%% Model accuracy check
MAE = mean_absolute_error(test_labels, test_predictions)
print("training:",MAE)

MAEv = mean_absolute_error(val_labels, val_predictions)
print("validation MAE=",MAEv)

val_labels = val_labels.reset_index(drop=True)
plt.figure(dpi=150)
plt.plot(val_labels, color = 'red', label = 'Real',linewidth = 1)
plt.plot(val_predictions, color = 'blue', label = 'Predicted',linestyle = 'dashed',marker = 'o',markersize = 1,linewidth = 1)
plt.title('stock')
plt.legend()
plt.show()

from functools import reduce
def Average(lst):
    return reduce(lambda a, b: a + b, lst) / len(lst)
average_test = Average(val_labels)
average_pred = Average(val_predictions)
RE = print("Relative Error=", abs(average_pred - average_test)/average_test*100)