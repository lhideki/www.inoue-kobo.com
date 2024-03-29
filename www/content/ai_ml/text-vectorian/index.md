---
title: "自然言語ベクトル化用Pythonモジュール(text-vectorian)をリリースしました"
date: "2019-01-22"
tags:
  - "AI/ML"
  - "NLP"
thumbnail: "ai_ml/text-vectorian/images/thumbnail.png"
---
# 自然言語ベクトル化用Pythonモジュール(text-vectorian)をリリースしました

## TL;DR

`text-vectgorian`は、自然言語をベクトル化するためのPythonモジュールです。
TokenizerやVectorizerの詳細を気にすることなく、任意のテキストから簡単にベクトル表現を取得することが可能です。
日本語Wikipediaで学習済みのモデルを同梱しているため、モジュールをインストールしてすぐに利用することができます。

`text-vectorian`はオープンソースとしてGitHubで公開しています。ライセンスは`MIT License`です。

* [text-vectorian](https://github.com/lhideki/text-vectorian)

## 開発理由

自然言語を機械学習で処理する場合に、何らかの方法で数値化する必要があります。
数値化する方法も様々ですが、何れの場合も文を単語や文字のトークン単位に分割(Tokenize)した上で、トークンの出現順で連番を振ったり、(特定コンテキストにおける)トークンの出現確率で分散表現を得る(Vectorize)ことになります。

日本語の場合はTokenizeの方法も選択する必要があったり、専用の辞書が必要であったりと英語に比べて数値化するまでの手順が煩雑です。
このような手間を省略するために、任意のテキストから簡単にベクトル表現を取得するためのPythonモジュールとして`text-vectorian`を作成しました。

## 利用方法

```bash
pip install text-vectorian
```

## 利用例

`SentencePieceVectorianクラス`のオブジェクトを作成し、任意のテキストをfitメソッドに渡す事でベクトル化した結果を得る事ができます。

```python
from text_vectorian import SentencePieceVectorian

vectorian = SentencePieceVectorian()
text = 'これはテストです。'
vectors = vectorian.fit(text).vectors

print(vectors)
```

Keras用のEmeddingレイヤーを取得することができるため、インデックスを渡す事でファインチューニングも可能です。

```python
from text_vectorian import SentencePieceVectorian

vectorian = SentencePieceVectorian()
text = 'これはテストです。'
indices = vectorian.fit(text).indices

print(indices)

from keras import Input, Model
from keras.layers import Dense, LSTM

input_tensor = Input((vectorian.max_tokens_len,))
common_input = vectorian.get_keras_layer(trainable=True)(input_tensor)
l1 = LSTM(32)(common_input)
output_tensor = Dense(3)(l1)

model = Model(input_tensor, output_tensor)
model.summary()
```

## Editor's Note

### 2018/01/22

文字ベースエンベディング(Char2VecVectorian)の実装をリリースしました。