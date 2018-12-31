# Auto-KerasでTextClassifierを使って見る

## TL;DR

[Auto-Keras](https://autokeras.com/)でTextClassifierを使って見ました。
`Auto-Keras`を利用した画像分類(ImageClassifier)については、[Auto-Kerasを使って見る](../auto-keras)を参照してください。

`Auto-Keras`のバージョンは0.3.5です。

テスト用のデータとして[GCP AutoML Natural Languageのベンチマーク](../automl-benchmark/index.md)と同じデータセットを使用しました。

## 利用方法

### Windowsで実行する場合の注意事項

TextClassifierでは単語をベクトル化するために[GloVe](https://nlp.stanford.edu/projects/glove/)を利用します。
学習済みモデル(重み)が自動的にダウンロードされますが、Windows環境だとエンコードの問題によりエラーになります。

このため学習済みモデルを読み込む際のエンコードを`utf-8`に固定します。
修正するファイルは`autokeras/text/text_preprocessor.py`です。

```python
def read_embedding_index(extract_path):
    """Read pre train file convert to embedding vector.

    Read the pre trained file into a dictionary where key is the word and value is embedding vector.

    Args:
        extract_path: String contains pre trained file path.

    Returns:
        embedding_index: Dictionary contains word with pre trained index.
    """
    embedding_index = {}
    # encoding='utf-8'を追加します。
    f = open(os.path.join(extract_path, Constant.PRE_TRAIN_FILE_NAME), encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs
    f.close()
    return embedding_index
```

### 学習の実行

最初に、scikit-learnの`train_test_split`を利用して学習用とテスト用でデータセットを分割しています。
その後12時間学習を実行しました。

```python
import pandas as pd
import keras
from autokeras import TextClassifier
from sklearn.model_selection import train_test_split
from keras.utils import plot_model
from keras.models import load_model

def read_csv(filename):
    data = pd.read_csv(filename, header=None)

    features = []
    labels = []
    for i in range(data[0].shape[0]):
        features.append(data[0][i])
        labels.append(data[1][i])

    return features, labels


filename = 'data/happiness.csv'
features, labels = read_csv(filename)
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, train_size=0.9, test_size=0.1)
classifier = TextClassifier(verbose=True)
classifier.fit(x=train_features, y=train_labels, time_limit=12 * 60 * 60)
#classifier.fit(x=train_features, y=train_labels, time_limit=3 * 60)
classifier.final_fit(train_features, train_labels, test_features, test_labels, retrain=True)
try:
    y = classifier.evaluate(test_features, test_labels)
    print(y)
except:
    pass
```

### 学習結果

自動的に生成されたモデルは44個でした。

| model_id | loss        | metric_value |
|----------|-------------|--------------|
| 0        | 5.017728031 | 0.6652       |
| 1        | 3.318194246 | 0.7508       |
| 2        | 4.438105488 | 0.6828       |
| 3        | 58.8590793  | 0.2392       |
| 4        | 22.77968587 | 0.346        |
| 5        | 29.73425417 | 0.332        |
| 6        | 13.21220874 | 0.6072       |
| 7        | 3.7186481   | 0.7236       |
| 8        | 48.23267536 | 0.4172       |
| 9        | 19.32834734 | 0.5632       |
| 10       | 3.698359919 | 0.744        |
| 11       | 16.14621212 | 0.398        |
| 12       | 23.86960423 | 0.5848       |
| 13       | 14.68351077 | 0.4724       |
| 14       | 26.15039973 | 0.6032       |
| 15       | 41.88511585 | 0.4388       |
| 16       | 14.77084085 | 0.5504       |
| 17       | 11.983115   | 0.5524       |
| 18       | 6.362913013 | 0.6424       |
| 19       | 3.5870453   | 0.7264       |
| 20       | 3.078068686 | 0.7648       |
| 21       | 8.84124558  | 0.538        |
| 22       | 35.84746119 | 0.5756       |
| 23       | 3.613662577 | 0.7244       |
| 24       | 2.990151328 | 0.768        |
| 25       | 3.625242233 | 0.7208       |
| 26       | 56.36195887 | 0.5404       |
| 27       | 3.815605688 | 0.7468       |
| 28       | 48.31595427 | 0.4764       |
| 29       | 4.617294157 | 0.6672       |
| 30       | 28.11585816 | 0.4696       |
| 31       | 4.101071692 | 0.6884       |
| 32       | 4.93131451  | 0.6744       |
| 33       | 11.7408888  | 0.6008       |
| 34       | 4.010589319 | 0.7356       |
| 35       | 3.323160946 | 0.7604       |
| 36       | 22.48387127 | 0.5428       |
| 37       | 30.88982011 | 0.4712       |
| 38       | 6.769645929 | 0.5692       |
| 39       | 17.03418162 | 0.4648       |
| 40       | 4.72360332  | 0.7012       |
| 41       | 48.37914122 | 0.4352       |
| 42       | 34.12850676 | 0.574        |
| 43       | 3.412533689 | 0.7664       |

最も精度の高いモデルのAccuracyは以下でした。

* 0.768

#### モデル詳細

現在のAuto-Kerasのバージョンではkerasとしてのモデル出力ができなかったため、以下の様にpytorchとしてモデルの出力を行いました。

```python
import torch

model_filename = 'models/text_model.h5'

# autokeras-0.3.5ではTextClassifierに対するkerasモデルのエクスポートが動作しませんでした。
# https://github.com/jhfjhfj1/autokeras/issues/394
#classifier.export_keras_model(model_filename)
model = classifier.cnn.best_model.produce_model()
torch.save(model, model_filename)
pd.DataFrame(classifier.cnn.searcher.history).to_csv('text_history.csv')
print(model)
```

```
TorchModel(
  (0): Conv1d(100, 64, kernel_size=(3,), stride=(1,), padding=(1,))
  (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (2): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (3): ReLU()
  (4): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
  (5): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (6): ReLU()
  (7): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
  (8): ReLU()
  (9): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
  (10): TorchAdd()
  (11): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (12): ReLU()
  (13): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
  (14): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (15): ReLU()
  (16): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,))
  (17): ReLU()
  (18): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
  (19): TorchAdd()
  (20): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (21): ReLU()
  (22): Conv1d(64, 128, kernel_size=(3,), stride=(2,), padding=(1,))
  (23): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (24): ReLU()
  (25): Conv1d(128, 128, kernel_size=(3,), stride=(1,), padding=(1,))
  (26): ReLU()
  (27): Conv1d(64, 128, kernel_size=(1,), stride=(2,))
  (28): TorchAdd()
  (29): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (30): ReLU()
  (31): Conv1d(128, 128, kernel_size=(3,), stride=(1,), padding=(1,))
  (32): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (33): ReLU()
  (34): Conv1d(128, 128, kernel_size=(3,), stride=(1,), padding=(1,))
  (35): ReLU()
  (36): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
  (37): TorchAdd()
  (38): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (39): ReLU()
  (40): Conv1d(128, 256, kernel_size=(3,), stride=(2,), padding=(1,))
  (41): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (42): ReLU()
  (43): Conv1d(256, 256, kernel_size=(3,), stride=(1,), padding=(1,))
  (44): ReLU()
  (45): Conv1d(128, 256, kernel_size=(1,), stride=(2,))
  (46): TorchAdd()
  (47): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (48): ReLU()
  (49): Conv1d(256, 256, kernel_size=(3,), stride=(1,), padding=(1,))
  (50): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (51): ReLU()
  (52): Conv1d(256, 256, kernel_size=(3,), stride=(1,), padding=(1,))
  (53): ReLU()
  (54): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
  (55): TorchAdd()
  (56): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (57): ReLU()
  (58): Conv1d(256, 512, kernel_size=(3,), stride=(2,), padding=(1,))
  (59): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (60): ReLU()
  (61): Conv1d(512, 512, kernel_size=(3,), stride=(1,), padding=(1,))
  (62): ReLU()
  (63): Conv1d(256, 512, kernel_size=(1,), stride=(2,))
  (64): TorchAdd()
  (65): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (66): ReLU()
  (67): Conv1d(512, 512, kernel_size=(3,), stride=(1,), padding=(1,))
  (68): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (69): ReLU()
  (70): Conv1d(512, 512, kernel_size=(3,), stride=(1,), padding=(1,))
  (71): ReLU()
  (72): Conv1d(512, 512, kernel_size=(1,), stride=(1,))
  (73): TorchAdd()
  (74): GlobalAvgPool1d()
  (75): Linear(in_features=512, out_features=7, bias=True)
  (76): ReLU()
  (77): Conv1d(128, 128, kernel_size=(3,), stride=(2,), padding=(1,))
  (78): Conv1d(128, 128, kernel_size=(3,), stride=(2,), padding=(1,))
  (79): ReLU()
  (80): Conv1d(128, 512, kernel_size=(1,), stride=(1,))
  (81): TorchAdd()
)
```

## 使って見た感想

事前処理として行うTokenizeやVetorizeが英語用のものしかないため、現在は日本語を利用することができません。
また、試した範囲では生成されたモデルの性能的は、`LSTM`を利用したものよりも劣るようです。

ただ、時間を掛ければGCP AutoML Natural Languageと同程度の性能は出ても良さそうなので、日本語対応を含めて色々試してみようと思います。