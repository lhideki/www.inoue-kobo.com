---
title: "SentencePiece+word2vecでコーパスによる差を確認してみる"
date: "2019-03-01"
tags:
  - "AI/ML"
  - "NLP"
thumbnail: "ai_ml/comp-corpus/images/thumbnail.png"
---
# SentencePiece+word2vecでコーパスによる差を確認してみる

## TL;DR

[SentencePiece](https://github.com/google/sentencepiece)と[word2vec](https://github.com/dav/word2vec)を前提として、学習に利用したコーパスの違いでどの程度の差がでるか確認しました。
今回は以下のコーパスで比較しました。

* [Wikipediaja](https://ja.wikipedia.org/wiki/Wikipedia:%E3%83%87%E3%83%BC%E3%82%BF%E3%83%99%E3%83%BC%E3%82%B9%E3%83%80%E3%82%A6%E3%83%B3%E3%83%AD%E3%83%BC%E3%83%89)
* Wikipediaja+[現代日本語書き言葉均衡コーパス](https://pj.ninjal.ac.jp/corpus_center/bccwj/)

それぞれのコーパスで学習したモデルを使用し、後述のベンチマーク用データを利用した分類問題の結果を評価しています。

### ベンチマーク用データ

[京都大学情報学研究科--NTTコミュニケーション科学基礎研究所 共同研究ユニット](http://nlp.ist.i.kyoto-u.ac.jp/kuntt/index.php)が提供するブログの記事に関するデータセットを利用しました。 このデータセットでは、ブログの記事に対して以下の4つの分類がされています。

* グルメ
* 携帯電話
* 京都
* スポーツ

### word2vecのハイパーパラメータ

* size: 300
* window: 50
* min_count: 1
* hs: 0
* negative: 10
* iter: 8

## ソースコード

### モジュールの準備


```python
import pandas as pd
import numpy as np
import keras
from keras.layers import Dense, Dropout, LSTM
from keras.layers.wrappers import Bidirectional
from keras.callbacks import LambdaCallback, EarlyStopping, ModelCheckpoint
from keras import Input, Model, utils
from keras.preprocessing.sequence import pad_sequences
from text_vectorian import SentencePieceVectorian

# Wikipediajaで学習したモデル
wikija_tokenizer_filename = '../../train-nlp-models/models/wikija-sentencepiece_300.model'
wikija_vectorizer_filename = '../../train-nlp-models/models/wikija-sentencepieced_word2vec_300.model'

wikija_vectorian = SentencePieceVectorian(tokenizer_filename=wikija_tokenizer_filename, vectorizer_filename=wikija_vectorizer_filename)

# Wikipediaja+現代日本語書き言葉均衡コーパスで学習したモデル
jc_tokenizer_filename = '../../train-nlp-models/models/wikija_jc-sentencepiece_300.model'
jc_vectorizer_filename = '../../train-nlp-models/models/wikija_jc-sentencepieced_word2vec_300.model'

jc_vectorian = SentencePieceVectorian(tokenizer_filename=jc_tokenizer_filename, vectorizer_filename=jc_vectorizer_filename)
```

### データロード用関数


```python
def _load_labeldata(train_dir, test_dir, vectorian):
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
        train_features.append(vectorian.fit(feature).indices)
    for feature in test_features_df['feature']:
        test_features.append(vectorian.fit(feature).indices)
    train_features = pad_sequences(train_features, maxlen=vectorian.max_tokens_len)
    test_features = pad_sequences(test_features, maxlen=vectorian.max_tokens_len)

    print(f'Trainデータ数: {len(train_features_df)}, Testデータ数: {len(test_features_df)}, ラベル数: {class_count}')

    return {
        'class_count': class_count,
        'label2index': label2index,
        'index2label': index2label,
        'train_labels': train_labels,
        'test_labels': test_labels,
        'test_label_indices': test_label_indices,
        'train_features': train_features,
        'test_features': test_features,
        'input_len': vectorian.max_tokens_len
    }
```

### モデル準備関数


```python
def _create_model(input_shape, hidden, class_count, vectorian):
    input_tensor = Input(input_shape)
    common_input = vectorian.get_keras_layer(trainable=True)(input_tensor)
    x1 = Bidirectional(LSTM(hidden))(common_input)
    output_tensor = Dense(class_count, activation='softmax')(x1)

    model = Model(input_tensor, output_tensor)
    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['acc', 'mse', 'mae'])

    return model
```

### データのロードとモデルの準備


```python
trains_dir = '../../word-or-character/data/trains'
tests_dir = '../../word-or-character/data/tests'
hidden = 356
```


```python
wikija_data = _load_labeldata(trains_dir, tests_dir, wikija_vectorian)
wikija_model = _create_model(wikija_data['train_features'][0].shape, hidden, wikija_data['class_count'], wikija_vectorian)
```

    TOE was not in vecabs, so use default token(▁).
    TLED was not in vecabs, so use default token(▁).
    MNP was not in vecabs, so use default token(▁).
    MNP was not in vecabs, so use default token(▁).
    KDD was not in vecabs, so use default token(▁).
    MNP was not in vecabs, so use default token(▁).
    MNP was not in vecabs, so use default token(▁).
    ADR was not in vecabs, so use default token(▁).
    QWERTY was not in vecabs, so use default token(▁).
    MNP was not in vecabs, so use default token(▁).
    DMA was not in vecabs, so use default token(▁).
    USJ was not in vecabs, so use default token(▁).
    TOE was not in vecabs, so use default token(▁).
    MNP was not in vecabs, so use default token(▁).
    MNP was not in vecabs, so use default token(▁).
    MNP was not in vecabs, so use default token(▁).
    TOE was not in vecabs, so use default token(▁).
    KDD was not in vecabs, so use default token(▁).
    ORZ was not in vecabs, so use default token(▁).
    ATOK was not in vecabs, so use default token(▁).
    POB was not in vecabs, so use default token(▁).
    KDD was not in vecabs, so use default token(▁).
    KDD was not in vecabs, so use default token(▁).
    PDA was not in vecabs, so use default token(▁).
    PXA was not in vecabs, so use default token(▁).
    NTTD was not in vecabs, so use default token(▁).
    SPAM was not in vecabs, so use default token(▁).
    〓〓〓 was not in vecabs, so use default token(▁).
    PDA was not in vecabs, so use default token(▁).
    KDD was not in vecabs, so use default token(▁).
    HDD was not in vecabs, so use default token(▁).
    KANSA was not in vecabs, so use default token(▁).
    KANSA was not in vecabs, so use default token(▁).
    MNP was not in vecabs, so use default token(▁).
    KULAS was not in vecabs, so use default token(▁).
    HOOP was not in vecabs, so use default token(▁).
    MNP was not in vecabs, so use default token(▁).
    KDD was not in vecabs, so use default token(▁).
    KANSA was not in vecabs, so use default token(▁).
    KANSA was not in vecabs, so use default token(▁).
    TOE was not in vecabs, so use default token(▁).
    DEMODE was not in vecabs, so use default token(▁).
    DMA was not in vecabs, so use default token(▁).
    DMA was not in vecabs, so use default token(▁).
    QOL was not in vecabs, so use default token(▁).


    Trainデータ数: 3767, Testデータ数: 419, ラベル数: 4



```python
jc_data = _load_labeldata(trains_dir, tests_dir, jc_vectorian)
jc_model = _create_model(jc_data['train_features'][0].shape, hidden, jc_data['class_count'], jc_vectorian)
```

    ▁PC was not in vecabs, so use default token(▁).
    HDD was not in vecabs, so use default token(▁).
    ▁PC was not in vecabs, so use default token(▁).


    Trainデータ数: 3767, Testデータ数: 419, ラベル数: 4


### Wikipediajaを利用したモデルによる学習


```python
wikija_model_filename = 'models/sentencepiece-model_wikija.model'

wikija_history = wikija_model.fit(wikija_data['train_features'], wikija_data['train_labels'],
                    epochs=50,
                    batch_size=256,
                    validation_data=(wikija_data['test_features'], wikija_data['test_labels']),
                    shuffle=False,
                    callbacks = [
                        EarlyStopping(patience=5, monitor='val_acc', mode='max'),
                        ModelCheckpoint(monitor='val_acc', mode='max', filepath=wikija_model_filename, save_best_only=True)
                    ])
```

    Train on 3767 samples, validate on 419 samples
    Epoch 1/50
    3767/3767 [==============================] - 12s 3ms/step - loss: 1.5952 - acc: 0.4518 - mean_squared_error: 0.1787 - mean_absolute_error: 0.3214 - val_loss: 0.9896 - val_acc: 0.5919 - val_mean_squared_error: 0.1329 - val_mean_absolute_error: 0.2693
    Epoch 2/50
    3767/3767 [==============================] - 7s 2ms/step - loss: 0.7083 - acc: 0.7420 - mean_squared_error: 0.0924 - mean_absolute_error: 0.2026 - val_loss: 0.8059 - val_acc: 0.6897 - val_mean_squared_error: 0.1061 - val_mean_absolute_error: 0.1924
    Epoch 3/50
    3767/3767 [==============================] - 7s 2ms/step - loss: 0.4515 - acc: 0.8447 - mean_squared_error: 0.0584 - mean_absolute_error: 0.1305 - val_loss: 0.6363 - val_acc: 0.7518 - val_mean_squared_error: 0.0832 - val_mean_absolute_error: 0.1572
    Epoch 4/50
    3767/3767 [==============================] - 7s 2ms/step - loss: 0.2426 - acc: 0.9230 - mean_squared_error: 0.0307 - mean_absolute_error: 0.0793 - val_loss: 0.7010 - val_acc: 0.7566 - val_mean_squared_error: 0.0877 - val_mean_absolute_error: 0.1462
    Epoch 5/50
    3767/3767 [==============================] - 7s 2ms/step - loss: 0.1460 - acc: 0.9551 - mean_squared_error: 0.0178 - mean_absolute_error: 0.0499 - val_loss: 0.7003 - val_acc: 0.7613 - val_mean_squared_error: 0.0858 - val_mean_absolute_error: 0.1356
    Epoch 6/50
    3767/3767 [==============================] - 7s 2ms/step - loss: 0.0521 - acc: 0.9894 - mean_squared_error: 0.0049 - mean_absolute_error: 0.0208 - val_loss: 0.7577 - val_acc: 0.7685 - val_mean_squared_error: 0.0857 - val_mean_absolute_error: 0.1271
    Epoch 7/50
    3767/3767 [==============================] - 7s 2ms/step - loss: 0.0213 - acc: 0.9968 - mean_squared_error: 0.0016 - mean_absolute_error: 0.0090 - val_loss: 0.8129 - val_acc: 0.7685 - val_mean_squared_error: 0.0861 - val_mean_absolute_error: 0.1220
    Epoch 8/50
    3767/3767 [==============================] - 7s 2ms/step - loss: 0.0125 - acc: 0.9976 - mean_squared_error: 9.5615e-04 - mean_absolute_error: 0.0052 - val_loss: 0.8750 - val_acc: 0.7709 - val_mean_squared_error: 0.0879 - val_mean_absolute_error: 0.1205
    Epoch 9/50
    3767/3767 [==============================] - 7s 2ms/step - loss: 0.0090 - acc: 0.9976 - mean_squared_error: 7.8005e-04 - mean_absolute_error: 0.0036 - val_loss: 0.9233 - val_acc: 0.7709 - val_mean_squared_error: 0.0895 - val_mean_absolute_error: 0.1198
    Epoch 10/50
    3767/3767 [==============================] - 7s 2ms/step - loss: 0.0072 - acc: 0.9976 - mean_squared_error: 7.1293e-04 - mean_absolute_error: 0.0028 - val_loss: 0.9594 - val_acc: 0.7709 - val_mean_squared_error: 0.0909 - val_mean_absolute_error: 0.1196
    Epoch 11/50
    3767/3767 [==============================] - 7s 2ms/step - loss: 0.0062 - acc: 0.9976 - mean_squared_error: 6.8587e-04 - mean_absolute_error: 0.0023 - val_loss: 0.9930 - val_acc: 0.7685 - val_mean_squared_error: 0.0916 - val_mean_absolute_error: 0.1190
    Epoch 12/50
    3767/3767 [==============================] - 7s 2ms/step - loss: 0.0057 - acc: 0.9976 - mean_squared_error: 6.8072e-04 - mean_absolute_error: 0.0021 - val_loss: 1.0169 - val_acc: 0.7661 - val_mean_squared_error: 0.0930 - val_mean_absolute_error: 0.1196
    Epoch 13/50
    3767/3767 [==============================] - 7s 2ms/step - loss: 0.0052 - acc: 0.9976 - mean_squared_error: 6.6341e-04 - mean_absolute_error: 0.0018 - val_loss: 1.0429 - val_acc: 0.7637 - val_mean_squared_error: 0.0931 - val_mean_absolute_error: 0.1187


### Wikipediaja+現代日本語書き言葉均衡コーパスを利用したモデルによる学習


```python
jc_model_filename = 'models/sentencepiece-model_jc.model'

jc_history = jc_model.fit(jc_data['train_features'], jc_data['train_labels'],
                    epochs=50,
                    batch_size=256,
                    validation_data=(jc_data['test_features'], jc_data['test_labels']),
                    shuffle=False,
                    callbacks = [
                        EarlyStopping(patience=5, monitor='val_acc', mode='max'),
                        ModelCheckpoint(monitor='val_acc', mode='max', filepath=jc_model_filename, save_best_only=True)
                    ])
```

    Train on 3767 samples, validate on 419 samples
    Epoch 1/50
    3767/3767 [==============================] - 11s 3ms/step - loss: 1.5468 - acc: 0.4906 - mean_squared_error: 0.1721 - mean_absolute_error: 0.3055 - val_loss: 0.9502 - val_acc: 0.5823 - val_mean_squared_error: 0.1312 - val_mean_absolute_error: 0.2514
    Epoch 2/50
    3767/3767 [==============================] - 7s 2ms/step - loss: 0.5973 - acc: 0.7783 - mean_squared_error: 0.0779 - mean_absolute_error: 0.1769 - val_loss: 0.6062 - val_acc: 0.7709 - val_mean_squared_error: 0.0778 - val_mean_absolute_error: 0.1626
    Epoch 3/50
    3767/3767 [==============================] - 7s 2ms/step - loss: 0.3432 - acc: 0.8943 - mean_squared_error: 0.0430 - mean_absolute_error: 0.1080 - val_loss: 0.6303 - val_acc: 0.7757 - val_mean_squared_error: 0.0801 - val_mean_absolute_error: 0.1479
    Epoch 4/50
    3767/3767 [==============================] - 7s 2ms/step - loss: 0.1769 - acc: 0.9501 - mean_squared_error: 0.0207 - mean_absolute_error: 0.0612 - val_loss: 0.6331 - val_acc: 0.7924 - val_mean_squared_error: 0.0761 - val_mean_absolute_error: 0.1358
    Epoch 5/50
    3767/3767 [==============================] - 7s 2ms/step - loss: 0.1037 - acc: 0.9758 - mean_squared_error: 0.0111 - mean_absolute_error: 0.0384 - val_loss: 0.6611 - val_acc: 0.7685 - val_mean_squared_error: 0.0778 - val_mean_absolute_error: 0.1265
    Epoch 6/50
    3767/3767 [==============================] - 6s 2ms/step - loss: 0.0411 - acc: 0.9944 - mean_squared_error: 0.0034 - mean_absolute_error: 0.0168 - val_loss: 0.7257 - val_acc: 0.7757 - val_mean_squared_error: 0.0811 - val_mean_absolute_error: 0.1221
    Epoch 7/50
    3767/3767 [==============================] - 6s 2ms/step - loss: 0.0204 - acc: 0.9965 - mean_squared_error: 0.0015 - mean_absolute_error: 0.0086 - val_loss: 0.7761 - val_acc: 0.7852 - val_mean_squared_error: 0.0827 - val_mean_absolute_error: 0.1183
    Epoch 8/50
    3767/3767 [==============================] - 7s 2ms/step - loss: 0.0129 - acc: 0.9973 - mean_squared_error: 9.5385e-04 - mean_absolute_error: 0.0054 - val_loss: 0.8343 - val_acc: 0.7757 - val_mean_squared_error: 0.0847 - val_mean_absolute_error: 0.1166
    Epoch 9/50
    3767/3767 [==============================] - 6s 2ms/step - loss: 0.0095 - acc: 0.9979 - mean_squared_error: 7.8065e-04 - mean_absolute_error: 0.0039 - val_loss: 0.8751 - val_acc: 0.7780 - val_mean_squared_error: 0.0862 - val_mean_absolute_error: 0.1158


### Wikipediajaを利用したモデルのクラシフィケーションレポート


```python
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import load_model

wikija_model = load_model(wikija_model_filename)

predicted_labels = wikija_model.predict(wikija_data['test_features']).argmax(axis=1)
print(classification_report(wikija_data['test_label_indices'], predicted_labels, target_names=wikija_data['index2label'].values()))
```

```
                  precision    recall  f1-score   support

              京都       0.72      0.81      0.76       137
            携帯電話       0.81      0.77      0.79       145
            スポーツ       0.71      0.74      0.73        47
             グルメ       0.83      0.72      0.77        90

       micro avg       0.77      0.77      0.77       419
       macro avg       0.77      0.76      0.76       419
    weighted avg       0.78      0.77      0.77       419
```

### Wikipediaja+現代日本語書き言葉均衡コーパスを利用したモデルのクラシフィケーションレポート


```python
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import load_model

jc_model = load_model(jc_model_filename)

predicted_labels = jc_model.predict(jc_data['test_features']).argmax(axis=1)
print(classification_report(jc_data['test_label_indices'], predicted_labels, target_names=jc_data['index2label'].values()))
```

```
                  precision    recall  f1-score   support

              京都       0.79      0.73      0.76       137
            携帯電話       0.86      0.81      0.84       145
            スポーツ       0.75      0.77      0.76        47
             グルメ       0.72      0.87      0.79        90

       micro avg       0.79      0.79      0.79       419
       macro avg       0.78      0.79      0.79       419
    weighted avg       0.80      0.79      0.79       419
```

## まとめ

以下の様に`Wikipediaja`だけで学習するよりも`Wikipediaja+現代日本語書き言葉均衡コーパス`で学習した場合の方が、分類問題で高い精度を得ることができました。

* Wikipediaja(Weighted Avg F1): 0.77
* Wikipediaja+現代日本語書き言葉均衡コーパス(Weighted Avg F1): **0.79**

OOV(Out of vocaburary)の数も以下の様に`Wikipediaja+現代日本語書き言葉均衡コーパス`の方が少なくなっており、学習データ量の差が直接精度に影響した形となっています。

* Wikipediaja: 45
* Wikipediaja+現代日本語書き言葉均衡コーパス: **3**
