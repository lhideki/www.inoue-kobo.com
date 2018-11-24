
# FizzBuzz問題をニューラルネットワークで解いてみる

## TL;DR

FizzBuzz問題をニューラルネットワークで解いてみます。

## ラベルデータの作成


```python
import pandas as pd

results = []

for i in range(1, 10000 + 1):
    if i % 3 == 0 and i % 5 == 0:
        results.append((i, 'FizzBuzz'))
    elif i % 3 == 0:
        results.append((i, 'Fizz'))
    elif i % 5 == 0:
        results.append((i, 'Buzz'))
    else:
        results.append((i, 'Number'))

data_df = pd.DataFrame(results, columns=['Number', 'Results'])
display(data_df.head(15))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Number</th>
      <th>Results</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Number</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Number</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Fizz</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Number</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Buzz</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>Fizz</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>Number</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>Number</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>Fizz</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>Buzz</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>Number</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>Fizz</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>Number</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>Number</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>FizzBuzz</td>
    </tr>
  </tbody>
</table>
</div>


## 前処理


```python
feature_title = 'Number'
label_title = 'Results'

printable_labels = {k: i for i, k in enumerate(data_df[label_title].unique())}
class_count = len(printable_labels)

display(data_df.head(15))
display(data_df.describe())
display(printable_labels)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Number</th>
      <th>Results</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Number</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Number</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Fizz</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Number</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Buzz</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>Fizz</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>Number</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>Number</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>Fizz</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>Buzz</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>Number</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>Fizz</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>Number</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>Number</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>FizzBuzz</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Number</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>10000.00000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5000.50000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2886.89568</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.00000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2500.75000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5000.50000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7500.25000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>10000.00000</td>
    </tr>
  </tbody>
</table>
</div>



    {'Number': 0, 'Fizz': 1, 'Buzz': 2, 'FizzBuzz': 3}



```python
from keras import utils

labels = utils.np_utils.to_categorical([printable_labels[label] for label in data_df[label_title]], num_classes=class_count)

display(labels.shape)
display(labels)
```

    C:\Users\hidek\Anaconda3\lib\site-packages\h5py\__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    Using TensorFlow backend.
    


    (10000, 4)



    array([[1., 0., 0., 0.],
           [1., 0., 0., 0.],
           [0., 1., 0., 0.],
           ...,
           [1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.]], dtype=float32)



```python
import numpy as np

digits_count = 5

didgits_map = utils.np_utils.to_categorical(range(10), num_classes=10)
features = np.zeros((len(data_df), 5, 10), dtype = np.int32)

for i, number in enumerate(data_df[feature_title]):
    for t, digit in enumerate(str(number).zfill(digits_count)[:]):
        features[i, t] = didgits_map[int(digit)]

print(features.shape)
print(features)
```

    (10000, 5, 10)
    [[[1 0 0 ... 0 0 0]
      [1 0 0 ... 0 0 0]
      [1 0 0 ... 0 0 0]
      [1 0 0 ... 0 0 0]
      [0 1 0 ... 0 0 0]]
    
     [[1 0 0 ... 0 0 0]
      [1 0 0 ... 0 0 0]
      [1 0 0 ... 0 0 0]
      [1 0 0 ... 0 0 0]
      [0 0 1 ... 0 0 0]]
    
     [[1 0 0 ... 0 0 0]
      [1 0 0 ... 0 0 0]
      [1 0 0 ... 0 0 0]
      [1 0 0 ... 0 0 0]
      [0 0 0 ... 0 0 0]]
    
     ...
    
     [[1 0 0 ... 0 0 0]
      [0 0 0 ... 0 0 1]
      [0 0 0 ... 0 0 1]
      [0 0 0 ... 0 0 1]
      [0 0 0 ... 0 1 0]]
    
     [[1 0 0 ... 0 0 0]
      [0 0 0 ... 0 0 1]
      [0 0 0 ... 0 0 1]
      [0 0 0 ... 0 0 1]
      [0 0 0 ... 0 0 1]]
    
     [[0 1 0 ... 0 0 0]
      [1 0 0 ... 0 0 0]
      [1 0 0 ... 0 0 0]
      [1 0 0 ... 0 0 0]
      [1 0 0 ... 0 0 0]]]
    

## データの分割


```python
from sklearn.model_selection import train_test_split

idx_features = range(len(data_df[feature_title]))
idx_labels = range(len(data_df[label_title]))
tmp_data = train_test_split(idx_features, idx_labels, train_size = 0.9, test_size = 0.1)

train_features = np.array([features[i] for i in tmp_data[0]])
valid_features = np.array([features[i] for i in tmp_data[1]])
train_labels = np.array([labels[i] for i in tmp_data[2]])
valid_labels = np.array([labels[i] for i in tmp_data[3]])

print(train_features.shape)
print(valid_features.shape)
print(train_labels.shape)
print(valid_labels.shape)
```

    (9000, 5, 10)
    (1000, 5, 10)
    (9000, 4)
    (1000, 4)
    

## ネットワークの作成


```python
from keras.models import Sequential
from keras.layers.core import Activation
from keras.layers import Dense, Dropout, LSTM, Embedding, Reshape, RepeatVector, Permute, Flatten, SimpleRNN
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.optimizers import RMSprop
from keras.callbacks import LambdaCallback, EarlyStopping, ModelCheckpoint
from keras import layers
from keras.layers.normalization import BatchNormalization
from keras import Input, Model

model = Sequential()
model.add(Flatten(input_shape=(features[0].shape)))
model.add(Dense(2048, activation='relu'))
model.add(Dense(2048, activation='relu'))
model.add(Dense(class_count, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['mae'])
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    flatten_1 (Flatten)          (None, 50)                0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 2048)              104448    
    _________________________________________________________________
    dense_2 (Dense)              (None, 2048)              4196352   
    _________________________________________________________________
    dense_3 (Dense)              (None, 4)                 8196      
    =================================================================
    Total params: 4,308,996
    Trainable params: 4,308,996
    Non-trainable params: 0
    _________________________________________________________________
    

## 学習


```python
from keras.callbacks import LambdaCallback, EarlyStopping, ModelCheckpoint, TensorBoard

model_filename = 'models/fizzbuzzz-model.h5'

history = model.fit(train_features,
          train_labels,
          epochs = 300,
          validation_split = 0.1,
          batch_size = 256,
          callbacks = [
              TensorBoard(log_dir = 'logs'),
              EarlyStopping(patience=5, monitor='val_mean_absolute_error'),
              ModelCheckpoint(model_filename, monitor='val_mean_absolute_error', save_best_only=True)
          ])
```

    Train on 8100 samples, validate on 900 samples
    Epoch 1/300
    8100/8100 [==============================] - 1s 137us/step - loss: 0.7933 - mean_absolute_error: 0.2498 - val_loss: 0.6256 - val_mean_absolute_error: 0.2134
    Epoch 2/300
    8100/8100 [==============================] - 0s 41us/step - loss: 0.6548 - mean_absolute_error: 0.2246 - val_loss: 0.6258 - val_mean_absolute_error: 0.2135
    Epoch 3/300
    8100/8100 [==============================] - 0s 39us/step - loss: 0.6422 - mean_absolute_error: 0.2232 - val_loss: 0.6292 - val_mean_absolute_error: 0.2229
    Epoch 4/300
    8100/8100 [==============================] - 0s 40us/step - loss: 0.6318 - mean_absolute_error: 0.2208 - val_loss: 0.6136 - val_mean_absolute_error: 0.2144
    Epoch 5/300
    8100/8100 [==============================] - 0s 41us/step - loss: 0.5816 - mean_absolute_error: 0.2072 - val_loss: 0.5271 - val_mean_absolute_error: 0.1884
    [省略]
    Epoch 297/300
    8100/8100 [==============================] - 0s 40us/step - loss: 1.4231e-07 - mean_absolute_error: 3.3136e-08 - val_loss: 1.2583e-06 - val_mean_absolute_error: 6.0084e-07
    Epoch 298/300
    8100/8100 [==============================] - 0s 40us/step - loss: 1.4168e-07 - mean_absolute_error: 3.2659e-08 - val_loss: 1.2518e-06 - val_mean_absolute_error: 5.9729e-07
    Epoch 299/300
    8100/8100 [==============================] - 0s 39us/step - loss: 1.4122e-07 - mean_absolute_error: 3.2316e-08 - val_loss: 1.2392e-06 - val_mean_absolute_error: 5.9097e-07
    Epoch 300/300
    8100/8100 [==============================] - 0s 39us/step - loss: 1.4064e-07 - mean_absolute_error: 3.1852e-08 - val_loss: 1.2335e-06 - val_mean_absolute_error: 5.8808e-07
    

## 検証


```python
from sklearn.metrics import classification_report, confusion_matrix

predicted_valid_labels = model.predict(valid_features).argmax(axis=1)
numeric_valid_labels = np.argmax(valid_labels, axis=1)
print(classification_report(numeric_valid_labels, predicted_valid_labels, target_names=printable_labels))
```

```
                 precision    recall  f1-score   support
    
         Number       1.00      1.00      1.00       551
           Fizz       1.00      1.00      1.00       270
           Buzz       1.00      1.00      1.00       119
       FizzBuzz       1.00      1.00      1.00        60
    
    avg / total       1.00      1.00      1.00      1000
```

## Jupyter Notebook

* [Jupyter Notebook](files/fizzbuzz-ml.ipynb)