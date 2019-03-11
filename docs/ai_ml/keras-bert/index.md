# SentencePiece + 日本語WikipediaのBERTモデルをKeras BERTで利用する

## TL;DR

Googleが公開している[BERT](https://github.com/google-research/bert)の学習済みモデルは、日本語Wikipediaもデータセットに含まれていますが、Tokenizeの方法が分かち書きを前提としているため、そのまま利用しても日本語の分類問題ではあまり高い精度を得ることができません。

このため、[SentencePiece](https://github.com/google/sentencepiece)でTokenizeしたデータセットで学習し直す必要があります。

BERTのトレーニングは結構な時間やマシンリソースが必要ですが、ありがたいことにSentencePiece+日本語Wikipediaで学習済みのモデルを配布してくれている方がいらっしゃるので、今回は以下を利用します。

* [BERT with SentencePiece を日本語 Wikipedia で学習してモデルを公開しました](https://yoheikikuta.github.io/bert-japanese/)

BERTには分類問題用のスクリプトが付属していますが、今回は[Keras BERT](https://github.com/CyberZHG/keras-bert)からBERTを利用します。

例の如くKNBC(後述のベンチマーク用データを参照)を利用して自然言語分類問題を対象としています。

### ベンチマーク用データ

[京都大学情報学研究科--NTTコミュニケーション科学基礎研究所 共同研究ユニット](http://nlp.ist.i.kyoto-u.ac.jp/kuntt/index.php)が提供するブログの記事に関するデータセットを利用しました。 このデータセットでは、ブログの記事に対して以下の4つの分類がされています。

* グルメ
* 携帯電話
* 京都
* スポーツ

## Keras BERTで独自の学習済みモデルを使用するための準備

`Keras BERT`ではGoogleが配布している学習済みモデルを利用する、もしくは自分でBERTモデルをトレーニングするのであれば特に特別な準備は必要ありませんが、今回は独自(Keras BERTに付属していない)の学習済みモデルを利用するためモデルのダウンロードやBERT用の設定ファイルを準備します。

### SentencePiece + 日本語WikipediaのBERTモデルをダウンロードする

[BERT with SentencePiece を日本語 Wikipedia で学習してモデルを公開しました](https://yoheikikuta.github.io/bert-japanese/)より、SentencePieceおよびBERT用の学習済みモデルをダウンロードします。
それぞれ以下のファイルをダウンロードします。

#### SentencePiece用のファイル

* wiki-ja.vocab
* wiki-ja.model

#### BERT用のファイル

* model.ckpt-1400000.data-00000-of-00001
* model.ckpt-1400000.index
* model.ckpt-1400000.meta

### bert_config.jsonを作成する

BERT用の各種パラメータは`bert_config.json`に記載します。`Keras BERT`では`bert_config.json`をＣｏｎｆｉｇとして指定します。
記載する内容は、以下の通りです([bert-japanese](https://github.com/yoheikikuta/bert-japanese)の`config.ini`よりBERT関連を抜粋しています)。

```json
{
  "attention_probs_dropout_prob": 0.1,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "max_position_embeddings": 512,
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "type_vocab_size": 2,
  "vocab_size": 32000
}
```

## ソースコード

### BERTのロード


```python
import sys
sys.path.append('modules')
from keras_bert import load_trained_model_from_checkpoint

config_path = 'bert-wiki-ja/bert_config.json'
# `model.ckpt-1400000` のように拡張子を付けないのがポイントです。
checkpoint_path = 'bert-wiki-ja/model.ckpt-1400000'

bert = load_trained_model_from_checkpoint(config_path, checkpoint_path)
bert.summary()
```

### データロード用関数

BERTで特徴量を抽出しています。


```python
import pandas as pd
import sentencepiece as spm
from keras import utils
import logging
import numpy as np

maxlen = 512
bert_dim = 768

sp = spm.SentencePieceProcessor()
sp.Load('bert-wiki-ja/wiki-ja.model')

def _get_vector(feature):
    common_seg_input = np.zeros((1, maxlen), dtype = np.float32)
    indices = np.zeros((1, maxlen), dtype = np.float32)
    
    tokens = []
    tokens.append('[CLS]')
    tokens.extend(sp.encode_as_pieces(feature))
    tokens.append('[SEP]')
        
    for t, token in enumerate(tokens):
        try:
            indices[0, t] = sp.piece_to_id(token)
        except:
            logging.warn(f'{token} is unknown.')
            indices[0, t] = sp.piece_to_id('<unk>')
    vector =  bert.predict([indices, common_seg_input])[0]
    
    return vector

def _load_labeldata(train_dir, test_dir):
    train_features_df = pd.read_csv(f'{train_dir}/features.csv')
    train_labels_df = pd.read_csv(f'{train_dir}/labels.csv')
    test_features_df = pd.read_csv(f'{test_dir}/features.csv')
    test_labels_df = pd.read_csv(f'{test_dir}/labels.csv')
    label2index = {k: i for i, k in enumerate(train_labels_df['label'].unique())}
    index2label = {i: k for i, k in enumerate(train_labels_df['label'].unique())}
    class_count = len(label2index)
    train_labels = utils.np_utils.to_categorical([label2index[label] for label in train_labels_df['label']], num_classes=class_count)
    test_label_indices = [label2index[label] for label in test_labels_df['label']]
    test_labels = utils.np_utils.to_categorical(test_label_indices, num_classes=class_count)

    train_features = []
    test_features = []
    
    for feature in train_features_df['feature']:
        train_features.append(_get_vector(feature))
    for feature in test_features_df['feature']:
        test_features.append(_get_vector(feature))

    print(f'Trainデータ数: {len(train_features_df)}, Testデータ数: {len(test_features_df)}, ラベル数: {class_count}')

    return {
        'class_count': class_count,
        'label2index': label2index,
        'index2label': index2label,
        'train_labels': train_labels,
        'test_labels': test_labels,
        'test_label_indices': test_label_indices,
        'train_features': np.array(train_features),
        'test_features': np.array(test_features),
        'input_len': maxlen
    }
```

### モデル準備関数

BERTで得た特徴量を入力としたBi-LSTMによるモデルです。


```python
from keras.layers import Dense, Dropout, LSTM, Bidirectional
from keras import Input, Model

def _create_model(train_features):
    class_count = 4

    input_tensor = Input(train_features[0].shape)
    x1 = Bidirectional(LSTM(356))(input_tensor)
    output_tensor = Dense(class_count, activation='softmax')(x1)

    model = Model(input_tensor, output_tensor)
    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['mae', 'mse', 'acc'])

    return model
```

### データのロードとモデルの準備

前述のようにデータロード時にBERTにより特徴量を抽出しています。
このため、実行には少々(GPUが無い場合はかなり?)時間がかかります。


```python
trains_dir = '../word-or-character/data/trains'
tests_dir = '../word-or-character/data/tests'

data = _load_labeldata(trains_dir, tests_dir)
```

    Trainデータ数: 3767, Testデータ数: 419, ラベル数: 4



```python
data['train_features'][:5]
```




    array([[[ 0.35466704,  0.8344387 , -0.44097826, ...,  0.2786668 ,
             -0.11497068,  0.29313257],
            [ 0.15172665,  0.02186944,  0.5719023 , ..., -0.34355962,
             -0.76192784, -0.37710115],
            [ 0.58773553, -0.5266533 ,  1.0297949 , ...,  0.2826115 ,
              0.01596622, -0.11598644],
            ...,
            [-0.42831585,  0.16217883,  0.24022198, ...,  0.35656708,
             -0.4026342 ,  0.90145355],
            [-0.4352011 ,  0.3489631 ,  0.3045481 , ...,  0.21556729,
             -0.18176533,  0.609959  ],
            [-0.36450043,  0.02214984,  0.2564181 , ...,  0.35734403,
             -0.01305788,  0.8493353 ]],
    
           [[ 0.31687105,  1.2244819 ,  0.36265972, ...,  0.2956537 ,
             -0.7329245 ,  0.3783138 ],
            [ 0.1442154 ,  0.169483  ,  0.4410671 , ..., -0.49457642,
             -0.855622  , -0.09040152],
            [ 0.6914083 ,  0.611323  ,  0.14708832, ..., -0.12922424,
             -0.19439547,  0.0302214 ],
            ...,
            [-0.38953468,  0.83356047,  0.22081861, ...,  0.21940875,
             -0.71583635,  0.9292939 ],
            [-0.37721068,  1.1795326 ,  0.29908165, ...,  0.18421009,
             -0.48757967,  0.8818958 ],
            [-0.20600995,  1.0697701 ,  0.21132138, ...,  0.60967827,
             -0.58680713,  0.75882226]],
    
           [[-0.10706001,  0.3738867 , -0.44477016, ...,  0.20106766,
              0.43196645, -0.40388373],
            [ 0.10844216, -0.16230595,  0.81597984, ..., -0.4139093 ,
             -0.94628114, -0.29458356],
            [ 1.0240474 , -0.6265489 ,  0.11975094, ..., -0.04260744,
              0.40971422, -0.1690548 ],
            ...,
            [-0.44445795, -0.18297565, -0.81684655, ..., -0.01337491,
              0.53387684,  0.8393724 ],
            [-0.5090573 , -0.3397976 , -0.86311024, ..., -0.15371192,
              0.33833283,  0.7278649 ],
            [-0.31896925, -0.41911578, -0.7302225 , ..., -0.19937491,
              0.30634403,  0.7484876 ]],
    
           [[-0.08156536,  1.3763438 , -0.4511324 , ..., -0.2718555 ,
              0.01139554,  0.3233358 ],
            [ 0.29251447,  0.016281  ,  0.3287114 , ..., -0.04428148,
             -0.527779  ,  0.12028977],
            [ 0.61508095, -0.10685639, -0.33700344, ...,  0.58882135,
              0.5536139 ,  0.06634241],
            ...,
            [-0.1810619 , -0.22033806, -0.37033078, ...,  0.03473432,
             -0.45422173,  1.4944264 ],
            [-0.40457633, -0.02369374, -0.30867767, ..., -0.07707401,
             -0.39675236,  1.509865  ],
            [-0.49272478, -0.3240378 ,  0.0129818 , ..., -0.05501541,
             -0.13940048,  1.6560441 ]],
    
           [[ 0.33917487, -0.45505336, -0.12048204, ..., -0.15798426,
             -0.31510183,  0.08813866],
            [ 0.01879137,  0.3558463 ,  0.74166125, ..., -1.1869664 ,
             -0.48231342, -0.57868123],
            [-0.04152466, -0.14042495,  0.47751656, ..., -0.8260575 ,
             -0.12649226,  0.49701825],
            ...,
            [ 0.16141744,  0.22437172, -0.06188362, ..., -0.3822586 ,
             -0.02379168,  0.22798373],
            [ 0.35819334,  0.11869927, -0.2132361 , ..., -0.29555017,
              0.06280891,  0.37545964],
            [ 0.20598263,  0.09404632, -0.13750747, ..., -0.29785928,
              0.10465414,  0.35825673]]], dtype=float32)



### 学習の実行


```python
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

model_filename = 'models/knbc-check-bert_v2.model'

model = _create_model(data['train_features'])
model.summary()
history = model.fit(data['train_features'],
          data['train_labels'],
          epochs = 100,
          batch_size = 128,
          validation_data=(data['test_features'], data['test_labels']),
          shuffle=False,
          verbose = 1,
          callbacks = [
              EarlyStopping(patience=5, monitor='val_acc', mode='max'),
              ModelCheckpoint(monitor='val_acc', mode='max', filepath=model_filename, save_best_only=True)
          ])
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         (None, 512, 768)          0         
    _________________________________________________________________
    bidirectional_1 (Bidirection (None, 712)               3204000   
    _________________________________________________________________
    dense_1 (Dense)              (None, 4)                 2852      
    =================================================================
    Total params: 3,206,852
    Trainable params: 3,206,852
    Non-trainable params: 0
    _________________________________________________________________
    Train on 3767 samples, validate on 419 samples
    Epoch 1/100
    3767/3767 [==============================] - 55s 15ms/step - loss: 1.1582 - mean_absolute_error: 0.2398 - mean_squared_error: 0.1260 - acc: 0.6294 - val_loss: 0.9369 - val_mean_absolute_error: 0.2007 - val_mean_squared_error: 0.1196 - val_acc: 0.6730
    Epoch 2/100
    3767/3767 [==============================] - 55s 15ms/step - loss: 0.5901 - mean_absolute_error: 0.1593 - mean_squared_error: 0.0782 - acc: 0.7648 - val_loss: 0.6725 - val_mean_absolute_error: 0.1509 - val_mean_squared_error: 0.0847 - val_acc: 0.7375
    Epoch 3/100
    3767/3767 [==============================] - 53s 14ms/step - loss: 0.5058 - mean_absolute_error: 0.1361 - mean_squared_error: 0.0674 - acc: 0.8044 - val_loss: 0.7014 - val_mean_absolute_error: 0.1513 - val_mean_squared_error: 0.0862 - val_acc: 0.7399
    Epoch 4/100
    3767/3767 [==============================] - 54s 14ms/step - loss: 0.3879 - mean_absolute_error: 0.1101 - mean_squared_error: 0.0522 - acc: 0.8535 - val_loss: 0.6635 - val_mean_absolute_error: 0.1347 - val_mean_squared_error: 0.0784 - val_acc: 0.7709
    Epoch 5/100
    3767/3767 [==============================] - 53s 14ms/step - loss: 0.2774 - mean_absolute_error: 0.0811 - mean_squared_error: 0.0368 - acc: 0.8967 - val_loss: 0.7371 - val_mean_absolute_error: 0.1338 - val_mean_squared_error: 0.0844 - val_acc: 0.7613
    Epoch 6/100
    3767/3767 [==============================] - 56s 15ms/step - loss: 0.2810 - mean_absolute_error: 0.0779 - mean_squared_error: 0.0364 - acc: 0.9055 - val_loss: 0.7168 - val_mean_absolute_error: 0.1222 - val_mean_squared_error: 0.0801 - val_acc: 0.7852
    Epoch 7/100
    3767/3767 [==============================] - 57s 15ms/step - loss: 0.1383 - mean_absolute_error: 0.0403 - mean_squared_error: 0.0164 - acc: 0.9562 - val_loss: 0.7089 - val_mean_absolute_error: 0.1261 - val_mean_squared_error: 0.0784 - val_acc: 0.7828
    Epoch 8/100
    3767/3767 [==============================] - 57s 15ms/step - loss: 0.1013 - mean_absolute_error: 0.0324 - mean_squared_error: 0.0123 - acc: 0.9700 - val_loss: 0.8280 - val_mean_absolute_error: 0.1213 - val_mean_squared_error: 0.0830 - val_acc: 0.7876
    Epoch 9/100
    3767/3767 [==============================] - 55s 15ms/step - loss: 0.0370 - mean_absolute_error: 0.0136 - mean_squared_error: 0.0038 - acc: 0.9915 - val_loss: 0.9463 - val_mean_absolute_error: 0.1124 - val_mean_squared_error: 0.0836 - val_acc: 0.7971
    Epoch 10/100
    3767/3767 [==============================] - 54s 14ms/step - loss: 0.0218 - mean_absolute_error: 0.0074 - mean_squared_error: 0.0023 - acc: 0.9944 - val_loss: 0.9268 - val_mean_absolute_error: 0.1112 - val_mean_squared_error: 0.0832 - val_acc: 0.7947
    Epoch 11/100
    3767/3767 [==============================] - 52s 14ms/step - loss: 0.0124 - mean_absolute_error: 0.0042 - mean_squared_error: 0.0013 - acc: 0.9965 - val_loss: 1.0381 - val_mean_absolute_error: 0.1047 - val_mean_squared_error: 0.0829 - val_acc: 0.8019
    Epoch 12/100
    3767/3767 [==============================] - 53s 14ms/step - loss: 0.0083 - mean_absolute_error: 0.0026 - mean_squared_error: 0.0010 - acc: 0.9971 - val_loss: 1.1083 - val_mean_absolute_error: 0.1052 - val_mean_squared_error: 0.0858 - val_acc: 0.8091
    Epoch 13/100
    3767/3767 [==============================] - 54s 14ms/step - loss: 0.0067 - mean_absolute_error: 0.0021 - mean_squared_error: 8.8009e-04 - acc: 0.9976 - val_loss: 1.1346 - val_mean_absolute_error: 0.1051 - val_mean_squared_error: 0.0861 - val_acc: 0.8019
    Epoch 14/100
    3767/3767 [==============================] - 51s 14ms/step - loss: 0.0057 - mean_absolute_error: 0.0018 - mean_squared_error: 8.3039e-04 - acc: 0.9973 - val_loss: 1.1624 - val_mean_absolute_error: 0.1028 - val_mean_squared_error: 0.0853 - val_acc: 0.7971
    Epoch 15/100
    3767/3767 [==============================] - 52s 14ms/step - loss: 0.0055 - mean_absolute_error: 0.0016 - mean_squared_error: 8.2199e-04 - acc: 0.9973 - val_loss: 1.1879 - val_mean_absolute_error: 0.1033 - val_mean_squared_error: 0.0862 - val_acc: 0.7947
    Epoch 16/100
    3767/3767 [==============================] - 52s 14ms/step - loss: 0.0049 - mean_absolute_error: 0.0014 - mean_squared_error: 7.9869e-04 - acc: 0.9973 - val_loss: 1.2090 - val_mean_absolute_error: 0.1033 - val_mean_squared_error: 0.0864 - val_acc: 0.7971
    Epoch 17/100
    3767/3767 [==============================] - 51s 14ms/step - loss: 0.0045 - mean_absolute_error: 0.0013 - mean_squared_error: 7.6631e-04 - acc: 0.9973 - val_loss: 1.2209 - val_mean_absolute_error: 0.1030 - val_mean_squared_error: 0.0864 - val_acc: 0.7947



```python
df = pd.DataFrame(history.history)
display(df)
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
      <th>val_loss</th>
      <th>val_mean_absolute_error</th>
      <th>val_mean_squared_error</th>
      <th>val_acc</th>
      <th>loss</th>
      <th>mean_absolute_error</th>
      <th>mean_squared_error</th>
      <th>acc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.936901</td>
      <td>0.200690</td>
      <td>0.119645</td>
      <td>0.673031</td>
      <td>1.158211</td>
      <td>0.239806</td>
      <td>0.125970</td>
      <td>0.629413</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.672482</td>
      <td>0.150946</td>
      <td>0.084675</td>
      <td>0.737470</td>
      <td>0.590081</td>
      <td>0.159289</td>
      <td>0.078209</td>
      <td>0.764800</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.701436</td>
      <td>0.151316</td>
      <td>0.086166</td>
      <td>0.739857</td>
      <td>0.505808</td>
      <td>0.136076</td>
      <td>0.067353</td>
      <td>0.804354</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.663509</td>
      <td>0.134704</td>
      <td>0.078389</td>
      <td>0.770883</td>
      <td>0.387925</td>
      <td>0.110072</td>
      <td>0.052156</td>
      <td>0.853464</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.737064</td>
      <td>0.133766</td>
      <td>0.084423</td>
      <td>0.761337</td>
      <td>0.277368</td>
      <td>0.081109</td>
      <td>0.036827</td>
      <td>0.896735</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.716836</td>
      <td>0.122190</td>
      <td>0.080062</td>
      <td>0.785203</td>
      <td>0.281022</td>
      <td>0.077929</td>
      <td>0.036403</td>
      <td>0.905495</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.708884</td>
      <td>0.126144</td>
      <td>0.078431</td>
      <td>0.782816</td>
      <td>0.138349</td>
      <td>0.040343</td>
      <td>0.016417</td>
      <td>0.956199</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.828004</td>
      <td>0.121335</td>
      <td>0.082972</td>
      <td>0.787590</td>
      <td>0.101332</td>
      <td>0.032350</td>
      <td>0.012329</td>
      <td>0.970003</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.946291</td>
      <td>0.112425</td>
      <td>0.083580</td>
      <td>0.797136</td>
      <td>0.037030</td>
      <td>0.013608</td>
      <td>0.003801</td>
      <td>0.991505</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.926845</td>
      <td>0.111206</td>
      <td>0.083175</td>
      <td>0.794749</td>
      <td>0.021777</td>
      <td>0.007377</td>
      <td>0.002344</td>
      <td>0.994425</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1.038088</td>
      <td>0.104662</td>
      <td>0.082922</td>
      <td>0.801909</td>
      <td>0.012358</td>
      <td>0.004177</td>
      <td>0.001337</td>
      <td>0.996549</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1.108292</td>
      <td>0.105200</td>
      <td>0.085754</td>
      <td>0.809069</td>
      <td>0.008341</td>
      <td>0.002552</td>
      <td>0.001044</td>
      <td>0.997080</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1.134614</td>
      <td>0.105058</td>
      <td>0.086104</td>
      <td>0.801909</td>
      <td>0.006718</td>
      <td>0.002110</td>
      <td>0.000880</td>
      <td>0.997611</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1.162371</td>
      <td>0.102757</td>
      <td>0.085296</td>
      <td>0.797136</td>
      <td>0.005706</td>
      <td>0.001753</td>
      <td>0.000830</td>
      <td>0.997345</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1.187874</td>
      <td>0.103300</td>
      <td>0.086247</td>
      <td>0.794749</td>
      <td>0.005452</td>
      <td>0.001592</td>
      <td>0.000822</td>
      <td>0.997345</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1.209019</td>
      <td>0.103252</td>
      <td>0.086414</td>
      <td>0.797136</td>
      <td>0.004915</td>
      <td>0.001417</td>
      <td>0.000799</td>
      <td>0.997345</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1.220853</td>
      <td>0.102980</td>
      <td>0.086429</td>
      <td>0.794749</td>
      <td>0.004455</td>
      <td>0.001282</td>
      <td>0.000766</td>
      <td>0.997345</td>
    </tr>
  </tbody>
</table>
</div>


### クラシフィケーションレポート


```python
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import load_model
from keras_bert import get_custom_objects

model = load_model(model_filename)

predicted_test_labels = model.predict(data['test_features']).argmax(axis=1)
numeric_test_labels = np.array(data['test_labels']).argmax(axis=1)
print(classification_report(numeric_test_labels, predicted_test_labels, target_names = ['グルメ', '携帯電話', '京都', 'スポーツ']))
```

                  precision    recall  f1-score   support
    
             グルメ       0.76      0.86      0.81       137
            携帯電話       0.85      0.82      0.84       145
              京都       0.73      0.68      0.70        47
            スポーツ       0.89      0.78      0.83        90
    
       micro avg       0.81      0.81      0.81       419
       macro avg       0.80      0.79      0.79       419
    weighted avg       0.81      0.81      0.81       419
    


## まとめ

F値で81と高い精度が得られました。これは同じコーパスを利用したWord2Vecの結果よりも高い数値であり、コーパスを追加したものを上回っています。

* Wikipediaja with BERT(Weighted Avg F1): **0.81**
* Wikipediaja(Weighted Avg F1): 0.77 [^1][^2]
* Wikipediaja+現代日本語書き言葉均衡コーパス(Weighted Avg F1): 0.79 [^1][^2]

[^1]: [SentencePiece+word2vecでコーパスによる差を確認してみる](https://www.inoue-kobo.com/ai_ml/comp-corpus/index.html)より。
[^2]: 独自にトレーニングしたWord2Vecを使用しています。

### ToDo

Keras BERT経由でFine Tuningを行うと結果が収束しなかったため、今回はFineTuningを行っていません。
さらに高い精度を得られる可能性があるため、原因を特定してFine Tuningした結果も確認したいと考えています。

## BERTの構造

BERTはトレーニング時と推論時で入出力部分が少し異なります。以下は推論時のネットワークを可視化したものです。

```python
from keras.utils import plot_model

plot_model(bert, to_file='train-bert.png', show_shapes=True)
```

<div style="width: 100%; height: 500px; overflow: scroll">
<img src="images/train-bert.png" style="width: 100%" />
</div>

## 参考文献

* [BERT](https://github.com/google-research/bert)
* [Keras BERT](https://github.com/CyberZHG/keras-bert)
* [BERT with SentencePiece を日本語 Wikipedia で学習してモデルを公開しました](https://yoheikikuta.github.io/bert-japanese/)
