---
title: "Keras BERTでファインチューニングしてみる"
date: "2019-05-27"
tags:
  - "AI/ML"
  - "NLP"
  - "BERT"
thumbnail: "ai_ml/keras-bert-with-finetuning/images/thumbnail.png"
---
# Keras BERTでファインチューニングしてみる

## TL;DR

[SentencePiece + 日本語WikipediaのBERTモデルをKeras BERTで利用する](https://www.inoue-kobo.com/ai_ml/keras-bert/index.html)において、`Keras BERT`を利用して日本語データセットの分類問題を扱って見ましたが、今回はファインチューニングを行ってみました。

BERTのモデルやベンチマーク用のデータなどは[SentencePiece + 日本語WikipediaのBERTモデルをKeras BERTで利用する](https://www.inoue-kobo.com/ai_ml/keras-bert/index.html)と同様です。

## Keras BERTでファインチューニングする際のポイント

`Keras BERT`のGitHubにデモとして公開されている[keras_bert_classification_tpu.ipynb](https://github.com/CyberZHG/keras-bert/tree/master/demo/tune)を参考にしました。

ポイントは以下のとおりです。私が試した範囲では、以下の両方を適切に設定しないと、Lossが収束しませんでした。

* `bert_config.json`の`max_position_embedding`、`max_seq_length`をデータセットの最大トークン数にする
* `optimizer`を`AdamWarmup`にする

### bert_config.json

対象データセットである`KNBC`を事前に調査し、BERTモデルで使用しているSetenencePieceでtokenizeした後の最大トークン数が`101`であることを調べておきます。

これにBERT用の区切り文字である`[CLS]`と`[SEP]`を足した`103`が最大トークン数になります。

```json
{
  "attention_probs_dropout_prob": 0.1,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "max_position_embeddings": 103,
  "max_seq_length": 103,
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "type_vocab_size": 2,
  "vocab_size": 32000
}
```

### AdamWarmup

スケジューリングされた学習をするための`Adam`のバリエーションだと思いますが、詳細は調べていません。
基本的に`keras_bert_classification_tpu.ipynb`のままです。

以下の様に`calc_train_steps`で減衰率などを事前に計算する必要があります。

```python
    decay_steps, warmup_steps = calc_train_steps(
        input_shape[0],
        batch_size=BATCH_SIZE,
        epochs=EPOCH,
    )
    # [省略]
    model.compile(loss='categorical_crossentropy',
                  optimizer=AdamWarmup(decay_steps=decay_steps, warmup_steps=warmup_steps, lr=LR),
                  #optimizer='nadam',
                  metrics=['mae', 'mse', 'acc'])
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
SEQ_LEN = 103 # 先にデータセットの最大のトークン数を調べています。
BATCH_SIZE = 16
BERT_DIM = 768
LR = 1e-4
EPOCH = 10

# ファインチューニング用にtraining,trainableをTrueに設定し、最大トークン数をseq_lenに設定します。
bert = load_trained_model_from_checkpoint(config_path, checkpoint_path, training=True,  trainable=True, seq_len=SEQ_LEN)
bert.summary()
```

### データロード用関数

```python
import pandas as pd
import sentencepiece as spm
from keras import utils
from keras.preprocessing.sequence import pad_sequences
import logging
import numpy as np

maxlen = SEQ_LEN

sp = spm.SentencePieceProcessor()
sp.Load('bert-wiki-ja/wiki-ja.model')

def _get_indice(feature):
    indices = np.zeros((maxlen), dtype = np.int32)

    tokens = []
    tokens.append('[CLS]')
    tokens.extend(sp.encode_as_pieces(feature))
    tokens.append('[SEP]')

    for t, token in enumerate(tokens):
        if t >= maxlen:
            break
        try:
            indices[t] = sp.piece_to_id(token)
        except:
            logging.warn(f'{token} is unknown.')
            indices[t] = sp.piece_to_id('<unk>')

    return indices

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
        train_features.append(_get_indice(feature))
    train_segments = np.zeros((len(train_features), maxlen), dtype = np.float32)
    for feature in test_features_df['feature']:
        test_features.append(_get_indice(feature))
    test_segments = np.zeros((len(test_features), maxlen), dtype = np.float32)

    print(f'Trainデータ数: {len(train_features_df)}, Testデータ数: {len(test_features_df)}, ラベル数: {class_count}')

    return {
        'class_count': class_count,
        'label2index': label2index,
        'index2label': index2label,
        'train_labels': train_labels,
        'test_labels': test_labels,
        'test_label_indices': test_label_indices,
        'train_features': np.array(train_features),
        'train_segments': np.array(train_segments),
        'test_features': np.array(test_features),
        'test_segments': np.array(test_segments),
        'input_len': maxlen
    }
```

### モデル準備関数

```python
from keras.layers import Dense, Dropout, LSTM, Bidirectional, Flatten, GlobalMaxPooling1D
from keras_bert.layers import MaskedGlobalMaxPool1D
from keras import Input, Model
from keras_bert import AdamWarmup, calc_train_steps

def _create_model(input_shape, class_count):
    decay_steps, warmup_steps = calc_train_steps(
        input_shape[0],
        batch_size=BATCH_SIZE,
        epochs=EPOCH,
    )

    bert_last = bert.get_layer(name='NSP-Dense').output
    x1 = bert_last
    output_tensor = Dense(class_count, activation='softmax')(x1)
    # Trainableの場合は、Input Masked Layerが3番目の入力なりますが、
    # FineTuning時には必要無いので1, 2番目の入力だけ使用します。
    # Trainableでなければkeras-bertのModel.inputそのままで問題ありません。
    model = Model([bert.input[0], bert.input[1]], output_tensor)
    model.compile(loss='categorical_crossentropy',
                  optimizer=AdamWarmup(decay_steps=decay_steps, warmup_steps=warmup_steps, lr=LR),
                  #optimizer='nadam',
                  metrics=['mae', 'mse', 'acc'])

    return model
```

### データのロードとモデルの準備

```python
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

trains_dir = '../word-or-character/data/trains'
tests_dir = '../word-or-character/data/tests'

data = _load_labeldata(trains_dir, tests_dir)
model_filename = 'models/knbc-check-bert_v3.model'
model = _create_model(data['train_features'].shape, data['class_count'])

model.summary()
```

### 学習の実行

```python
history = model.fit([data['train_features'], data['train_segments']],
          data['train_labels'],
          epochs = EPOCH,
          batch_size = BATCH_SIZE,
          validation_data=([data['test_features'], data['test_segments']], data['test_labels']),
          shuffle=False,
          verbose = 1,
          callbacks = [
              #EarlyStopping(patience=5, monitor='val_acc', mode='max'),
              ModelCheckpoint(monitor='val_acc', mode='max', filepath=model_filename, save_best_only=True)
          ])
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
      <td>0.561248</td>
      <td>0.131881</td>
      <td>0.071544</td>
      <td>0.792363</td>
      <td>0.782363</td>
      <td>0.202023</td>
      <td>0.102747</td>
      <td>0.679055</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.739699</td>
      <td>0.115426</td>
      <td>0.085423</td>
      <td>0.782816</td>
      <td>0.379809</td>
      <td>0.096035</td>
      <td>0.049494</td>
      <td>0.856384</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.062872</td>
      <td>0.125169</td>
      <td>0.098740</td>
      <td>0.761337</td>
      <td>0.192548</td>
      <td>0.046621</td>
      <td>0.024535</td>
      <td>0.934165</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.176542</td>
      <td>0.119584</td>
      <td>0.099672</td>
      <td>0.763723</td>
      <td>0.086767</td>
      <td>0.022215</td>
      <td>0.011102</td>
      <td>0.971861</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.921495</td>
      <td>0.113555</td>
      <td>0.088224</td>
      <td>0.778043</td>
      <td>0.061029</td>
      <td>0.013554</td>
      <td>0.007306</td>
      <td>0.980887</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.043859</td>
      <td>0.104909</td>
      <td>0.087325</td>
      <td>0.794749</td>
      <td>0.023458</td>
      <td>0.006163</td>
      <td>0.002836</td>
      <td>0.991505</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.082306</td>
      <td>0.102066</td>
      <td>0.085363</td>
      <td>0.804296</td>
      <td>0.022063</td>
      <td>0.005459</td>
      <td>0.002826</td>
      <td>0.992567</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.048261</td>
      <td>0.098721</td>
      <td>0.083447</td>
      <td>0.809069</td>
      <td>0.015968</td>
      <td>0.003016</td>
      <td>0.001616</td>
      <td>0.995753</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.027535</td>
      <td>0.096534</td>
      <td>0.081463</td>
      <td>0.816229</td>
      <td>0.007195</td>
      <td>0.001710</td>
      <td>0.000927</td>
      <td>0.996814</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1.040992</td>
      <td>0.096410</td>
      <td>0.081862</td>
      <td>0.811456</td>
      <td>0.003626</td>
      <td>0.001289</td>
      <td>0.000520</td>
      <td>0.997876</td>
    </tr>
  </tbody>
</table>
</div>

### クラシフィケーションレポート

```python
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import load_model
from keras_bert import get_custom_objects

model = load_model(model_filename, custom_objects=get_custom_objects())

predicted_test_labels = model.predict([data['test_features'], data['test_segments']]).argmax(axis=1)
numeric_test_labels = np.array(data['test_labels']).argmax(axis=1)

report_filename = 'models/knbc-check-bert_v3.txt'

with open(report_filename, 'w', encoding='utf-8') as f:
    print(classification_report(numeric_test_labels, predicted_test_labels, target_names = ['グルメ', '携帯電話', '京都', 'スポーツ']), file=f)
    print(classification_report(numeric_test_labels, predicted_test_labels, target_names = ['グルメ', '携帯電話', '京都', 'スポーツ']))
```

```
              precision    recall  f1-score   support

         グルメ       0.80      0.80      0.80       137
        携帯電話       0.85      0.77      0.81       145
          京都       0.74      0.85      0.79        47
        スポーツ       0.84      0.90      0.87        90

   micro avg       0.82      0.82      0.82       419
   macro avg       0.81      0.83      0.82       419
weighted avg       0.82      0.82      0.82       419
```

## まとめ

F値で82と綺麗に過去最高性能でした。
FineTuning前後で1%の差ですが、元々が高い精度ですので、有意な差がでることは素晴らしいと思います。

* Wikipediaja with BERT/Fine Tuned(Weighted Avg F1): **0.82**
* Wikipediaja with BERT(Weighted Avg F1): 0.81 [^1]
* Wikipediaja(Weighted Avg F1): 0.77 [^2][^3]
* Wikipediaja+現代日本語書き言葉均衡コーパス(Weighted Avg F1): 0.79 [^2][^3]

[^1]: [SentencePiece + 日本語WikipediaのBERTモデルをKeras BERTで利用する](https://www.inoue-kobo.com/ai_ml/keras-bert/index.html)
[^2]: [SentencePiece+word2vecでコーパスによる差を確認してみる](https://www.inoue-kobo.com/ai_ml/comp-corpus/index.html)より。
[^3]: 独自にトレーニングしたWord2Vecを使用しています。

## BERTの構造

今回のモデルの構造です。正確にはBERT+分類用のDenseの構造になっています。

```python
from keras.utils import plot_model

plot_model(model, to_file='train-bert.png', show_shapes=True)
```

<div style="width: 100%; height: 500px; overflow: scroll">
<img src="images/train-bert.png" style="width: 100%" />
</div>

## 参考文献

* [Keras BERT](https://github.com/CyberZHG/keras-bert)
* [BERT with SentencePiece を日本語 Wikipedia で学習してモデルを公開しました](https://yoheikikuta.github.io/bert-japanese/)
* [SentencePiece + 日本語WikipediaのBERTモデルをKeras BERTで利用する](https://www.inoue-kobo.com/ai_ml/keras-bert/index.html)