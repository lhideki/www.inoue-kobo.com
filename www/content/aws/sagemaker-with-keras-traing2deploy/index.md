---
title: "SageMakerでKerasの独自モデルをトレーニングしてデプロイするまで(Python3対応)"
date: "2019-08-14"
tags:
  - "AWS"
  - "SageMaker"
  - "Keras"
  - "AI/ML"
thumbnail: "aws/sagemaker-with-keras-traing2deploy/images/thumbnail.png"
---
# SageMakerでKerasの独自モデルをトレーニングしてデプロイするまで(Python3対応)

## TL;DR

[AWS SageMaker](https://aws.amazon.com/jp/sagemaker/)において、Kerasによる独自モデルをトレーニングし、SageMakerのエンドポイントとしてデプロイします。
また、形態素解析やベクトル化のような前処理を、個別にDockerコンテナを作成することなしにエンドポイント内で行うようにします。このために、[SageMaker TensorFlow Serving Container](https://github.com/aws/sagemaker-tensorflow-serving-container)を利用します。

`SageMaker TensorFlow Serving Container`を利用するメリットは以下のとおりです。

* 学習時はスクリプトモードでOK。
* 前処理用に別に専用インスタンスが不要。エンドポイントで完結。
* 形態素解析やベクトル化のような前処理を行う独自の推論用スクリプトを利用できる。
* 個別にDockerコンテナを作成する必要が無い。
* [TensorFlowModel](https://sagemaker.readthedocs.io/en/stable/sagemaker.tensorflow.html?highlight=TensorFlowMOdel#tensorflow-model)と違ってPython3が利用できる。

### KerasとTensorFlow付属のKerasの違いに注意

Kerasは独立したモジュールとTensorFlow付属のモジュールの2種類があります。

SageMakerのエンドポイント用のモデルは、Kerasのモデルではなく、TensorFlowのモデルとして保存する必要があり、Estimator内で実行する学習用コードの中でTensorFlowのモデルとしてKerasモデルを保存する必要があります。

しかしながら、SageMakerのEstimatorがTensorFlowのセッションを初期化するため、独立したモジュールのKerasを利用するとセッション初期化のタイミングの問題により、Estimator内で実行する学習用コードの中でTensorFlowのモデルとして保存する場合に、未初期化の変数が発生して失敗します。

このため、以下の様にTensorFlow付属のKerasモジュールのみを利用してKerasモデルを作成します。

```python
# 以下のように`keras`を利用することはできません。
# from keras import Input, Model
# from keras.layers.wrappers import Bidirectional
# from keras.layers import Dense, Dropout, LSTM, Embedding

# 以下のように`tensorflow.python.keras`を使用します。
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers.wrappers import Bidirectional
from tensorflow.python.keras.layers import Dense, Dropout, LSTM, Embedding
```

### ラベルデータの準備

ラベルデータはAWS S3上に配置されていることが前提となっています。

### ベンチマーク用データ

[京都大学情報学研究科--NTTコミュニケーション科学基礎研究所 共同研究ユニット](http://nlp.ist.i.kyoto-u.ac.jp/kuntt/index.php)が提供するブログの記事に関するデータセットを利用しました。 このデータセットでは、ブログの記事に対して以下の4つの分類がされています。

* グルメ
* 携帯電話
* 京都
* スポーツ

## 全体の流れ

* 事前準備
* 学習用スクリプトとモジュールの準備
* トレーニングの実行
* 学習したモデルのダウンロード
* 推論用スクリプトを含めてパッケージング
* エンドポイントの作成
* エンドポイントの削除

## 事前準備

### 学習用スクリプトの準備

`train/train_v1.py`として用意します。

前述のようにTensorFlow付属のKerasモジュールのみを利用してモデルを作成する必要があります。
また、モデルの保存はTensorFlowのモデルとして保存するために`tf.saved_model.simple_save`を使用します。

その他、Kerasに依存したサードパーティーモジュールを利用する場合、それらのサードパーティーモジュールもTensorFlow付属のKerasを利用する必要があります。
今回は`keras-self-attention`を利用するため、環境変数として`TF_KERAS`に`1`を設定してから`import`することで、`keras-self-attention`がTensorFlow付属のKerasを利用するようになります(これは`keras-self-attention`独自の実装です)。

### 推論用スクリプトの準備

`predict/code/inference.py`として用意します。この際、`code/inference.py`は固定である点に注意してください。
`SageMaker TensorFlow Serving Container`は、以下のように推論するための学習済みモデルと推論用スクリプトが並んで存在することを前提としています。

```
model.tar.gz
├── [Model Version]
│   ├── variables
│   └── saved_model.pb
└── code
    ├── inference.py
    └── requirements.txt

```

学習時と違い、`code`配下に`requirements.txt`を配置することで、デプロイ時に依存したモジュールを自動的にインストールしてくれます。

### ディレクトリ構成

```
├── predict
│   ├── [Model Version] <- ダウンロードした学習済みモデル。
│   └── code
│       ├── inference.py
│       └── requirements.txt
└── train
    ├── train_v1.py
    └── [Moduels] <- 学習用スクリプトが依存するモジュール。
```

`train/[Modules]`は学習用スクリプトと同じディレクトリに配置することで、学習用スクリプトの方でパスの違いを意識せずに`import`することができます。

## 学習用スクリプトとモジュールの準備

学習用スクリプトと依存するモジュールを準備します。
依存するモジュールは`pip install [Module] --target train`で対象のディレクトリに配置します。

Estimatorでは、この`train`配下を`src`に指定することで、学習で使用するコンテナにまるごと配置します。


```python
!pip install text-vectorian --target train
!pip install keras-self-attention --target train
```


```python
!cat train/train_v1.py
```

```python
    import os

    # SeqSelfAttentionをTensorFlow付属のKerasモジュールを利用するために必要な設定です。
    # 以下の設定が無効な場合は、Kerasのバージョンが合わずに_create_modelが失敗します。
    os.environ['TF_KERAS'] = '1'

    import sys
    import numpy as np
    import pandas as pd
    from keras_self_attention import SeqSelfAttention
    import pickle
    import logging
    from tensorflow.python.keras import utils
    from tensorflow.python.keras.callbacks import LambdaCallback, EarlyStopping, ModelCheckpoint
    from tensorflow.python.keras.preprocessing.sequence import pad_sequences
    from tensorflow.python.keras import Input, Model
    from tensorflow.python.keras.layers.wrappers import Bidirectional
    from tensorflow.python.keras.layers import Dense, Dropout, LSTM, Embedding
    from tensorflow.python.keras.models import Sequential, load_model
    from tensorflow.python.keras import backend as K
    import tensorflow as tf
    import argparse
    from text_vectorian import SentencePieceVectorian

    input_len = 64
    logger = logging.getLogger(__name__)
    vectorian = SentencePieceVectorian()


    def _save_model(checkpoint_filename, history, model_dir, output_dir):
        '''
        モデルや実行履歴を保存します。
        '''
        # Checkpointからモデルをロードし直し(ベストなモデルのロード)
        model = load_model(checkpoint_filename,
                           custom_objects=SeqSelfAttention.get_custom_objects())
        # Historyの保存
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(output_dir + f'/history.csv')

        # Endpoint用のモデル保存
        with tf.keras.backend.get_session() as sess:
            tf.saved_model.simple_save(
                sess,
                args.model_dir + '/1',
                inputs={'inputs': model.input},
                outputs={t.name: t for t in model.outputs})


    def _save_labels(data, output_dir):
        '''
        推論で利用するためのラベルデータの情報を保存します。
        '''

        del data['labels']
        del data['features']

        data_filename = output_dir + f'/labels.pickle'

        with open(data_filename, mode='wb')as f:
            pickle.dump(data, f)


    def _load_labeldata(train_dir):
        '''
        ラベルデータをロードします。
        '''
        features_df = pd.read_csv(
            f'{train_dir}/features.csv', dtype=str)
        labels_df = pd.read_csv(f'{train_dir}/labels.csv', dtype=str)
        label2index = {k: i for i, k in enumerate(labels_df['label'].unique())}
        index2label = {i: k for i, k in enumerate(labels_df['label'].unique())}
        class_count = len(label2index)
        labels = utils.np_utils.to_categorical(
            [label2index[label] for label in labels_df['label']], num_classes=class_count)

        features = []

        for feature in features_df['feature']:
            features.append(vectorian.fit(feature).indices)
        features = pad_sequences(features, maxlen=input_len, dtype='float32')

        logger.info(
            f'データ数: {len(features_df)}, ラベル数: {class_count}, Labels Shape: {labels.shape}, Features Shape: {features.shape}')

        return {
            'class_count': class_count,
            'label2index': label2index,
            'index2label': index2label,
            'labels': labels,
            'features': features,
            'input_len': input_len
        }


    def _create_model(input_shape, hidden, class_count):
        '''
        モデルの構造を定義します。
        '''

        wv = vectorian._vectorizer.model.wv
        input_tensor = Input(input_shape)

        # TensorFlow付属のKerasモジュールを使用する必要があるため、
        # gensimのget_keras_embeddingを利用せずに、kerasのEmbeddingレイヤーを
        # 直接使用します。
        x1 = Embedding(input_dim=wv.vectors.shape[0], output_dim=wv.vectors.shape[1], weights=[wv.vectors], trainable=False)(input_tensor)
        x1 = SeqSelfAttention(
            attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL)(x1)
        x1 = Bidirectional(LSTM(hidden))(x1)
        x1 = Dense(hidden * 2)(x1)
        x1 = Dropout(0.1)(x1)
        output_tensor = Dense(class_count, activation='softmax')(x1)

        model = Model(input_tensor, output_tensor)
        model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=[
                      'acc', 'mse', 'mae', 'top_k_categorical_accuracy'])
        model.summary()

        return model


    if __name__ == '__main__':
        # Estimatorのパラメータは本スクリプトの引数として指定されるため、argparseで解析します。
        # Pythonの文法上Estimatorのパラメータはスネークケースですが、引数として渡される場合は`--`をプレフィクスにしたチェーンケースになっています。
        # なおhyperparameterとして渡すdictオブジェクトは、名前の変換は行われないのでそのまま引数名として受け取ります。
        parser = argparse.ArgumentParser()
        # Estimatorのパラメータとして渡されるパラメータ
        parser.add_argument('--epochs', type=int, default=10)
        parser.add_argument('--batch-size', type=int, default=100)
        parser.add_argument('--hidden', type=int)
        parser.add_argument('--container-log-level',
                            type=int, default=logging.INFO)
        parser.add_argument('--validation-split', type=float, default=0.1)
        # 環境変数として渡されるパラメータ
        parser.add_argument('--model-dir', type=str,
                            default=os.environ['SM_MODEL_DIR'])
        parser.add_argument('--train-dir', type=str,
                            default=os.environ['SM_CHANNEL_TRAIN'])
        parser.add_argument('--test-dir', type=str,
                            default=os.environ['SM_CHANNEL_TEST'])
        parser.add_argument('--output-dir', type=str,
                            default=os.environ['SM_OUTPUT_DATA_DIR'])
        parser.add_argument('--model-version', type=str,
                            default='')

        args, _ = parser.parse_known_args()
        # ログレベルを引数で渡されたコンテナのログレベルと合わせます。
        logging.basicConfig(level=args.container_log_level)

        # ラベルデータをロードします。
        data = _load_labeldata(args.train_dir)
        # モデルの定義を行います。
        model = _create_model(data['features'][0].shape,
                              args.hidden,
                              data['class_count'])
        # 学習用のデータを準備します。
        train_features = data['features']
        train_labels = data['labels']
        # 学習を実行します。
        # verboseを2に指定するのはポイントです。デフォルトは1ですが、そのままではプログレッシブバーの出力毎にログが記録されるため冗長です。
        # 2にすることで、epochごとの結果だけ出力されるようになります。
        if args.validation_split > 0:
            monitor_target = 'val_acc'
        else:
            monitor_target = 'acc'

        checkpoint_filename = f'model_{args.model_version}.h5'
        history = model.fit(train_features, train_labels,
                            batch_size=args.batch_size,
                            validation_split=args.validation_split,
                            epochs=args.epochs,
                            verbose=2,
                            callbacks=[
                                EarlyStopping(
                                    patience=3, monitor=monitor_target, mode='max'),
                                ModelCheckpoint(filepath=checkpoint_filename,
                                                save_best_only=True, monitor=monitor_target, mode='max')
                            ])

        # Checkpointからロードし直すため、一度モデルを削除します。
        del model
        # 学習したモデルを保存します。
        _save_model(checkpoint_filename, history, args.model_dir, args.output_dir)
        # 推論時に利用するラベルデータの情報を保存します。
        _save_labels(data, args.output_dir)
```

## トレーニングの実行

Estimatorを利用してトレーニングを実行します。

なお、ハイパーパラメータのチューニングについては[SageMakerでTensorFlow+Kerasによる独自モデルをトレーニングする方法](https://www.inoue-kobo.com/aws/sagemaker-with-mymodel/index.html#hyperparametertuner)を参照してください。


```python
PROJECT_NAME = 'sagemaker-with-keras-traing2deploy'
TAGS = [{ 'Key': 'example.ProjectName', 'Value': PROJECT_NAME }]
VERSION = 'v1'
BUCKET_NAME = f'sagemaker-us-east-1.example.com'
DATA_ROOT = f's3://{BUCKET_NAME}/{PROJECT_NAME}'
TRAINS_DIR = f'{DATA_ROOT}/data/trains'
TESTS_DIR = f'{DATA_ROOT}/data/tests'
OUTPUTS_DIR = f'{DATA_ROOT}/outputs'
ROLE = 'arn:aws:iam::012345678901:role/service-role/AmazonSageMaker-ExecutionRole-20181129T043923'
```

```python
from sagemaker.tensorflow import TensorFlow
import logging

params = {
    'batch-size': 256,
    'epochs': 10,
    'hidden': 32,
    'validation-split': 0.1,
    'model_version': VERSION
}
metric_definitions = [
    {'Name': 'train:acc', 'Regex': 'acc: (\S+)'},
    {'Name': 'train:mse', 'Regex': 'mean_squared_error: (\S+)'},
    {'Name': 'train:mae', 'Regex': 'mean_absolute_error: (\S+)'},
    {'Name': 'train:top-k', 'Regex': 'top_k_categorical_accuracy: (\S+)'},
    {'Name': 'valid:acc', 'Regex': 'val_acc: (\S+)'},
    {'Name': 'valid:mse', 'Regex': 'val_mean_squared_error: (\S+)'},
    {'Name': 'valid:mae', 'Regex': 'val_mean_absolute_error: (\S+)'},
    {'Name': 'valid:top-k', 'Regex': 'val_top_k_categorical_accuracy: (\S+)'},
]
estimator = TensorFlow(
    role=ROLE,
    source_dir='train',
    entry_point=f'train_{VERSION}.py',
    train_instance_count=1,
    train_instance_type='ml.p2.xlarge',
    framework_version='1.12.0',
    py_version='py3',
    script_mode=True,
    hyperparameters=params,
    output_path=OUTPUTS_DIR,
    container_log_level=logging.INFO,
    metric_definitions=metric_definitions,
    tags=TAGS
)
inputs = {'train': TRAINS_DIR, 'test': TESTS_DIR}
```


```python
import shortuuid

uuid = shortuuid.ShortUUID().random(length=8)
estimator.fit(job_name=f'{PROJECT_NAME}-{VERSION}-s-{uuid}', inputs=inputs)
```

## 学習したモデルのダウンロード

`SageMaker TensorFlow Serving Container`では、推論用スクリプトと学習したモデルを一緒に含めて`model.tar.gz`として再パッケージする必要があります。
このため、まずは学習済みモデルをダウンロードします。

また、推論時に人が読んで理解できるラベルとして推論結果を出力するために、学習時に保存しておいた`labels.pickle`を含む`output.tar.gz`もダウンロードします。


```python
import boto3
import urllib

s3 = boto3.resource('s3')
bucket = s3.Bucket(BUCKET_NAME)

model_url = urllib.parse.urlparse(estimator.model_data)
output_url = urllib.parse.urlparse(f'{estimator.output_path}/{estimator.latest_training_job.job_name}/output/output.tar.gz')

bucket.download_file(model_url.path[1:], 'predict/model.tar.gz')
bucket.download_file(output_url.path[1:], 'predict/output.tar.gz')
```


```python
!cd predict; tar zxvf model.tar.gz
!cd predict; tar zxvf output.tar.gz
```

```
    1/
    1/variables/
    1/variables/variables.data-00000-of-00001
    1/variables/variables.index
    1/saved_model.pb
    history.csv
    labels.pickle
```

### ダウンロードしたモデルの動作確認

ダウンロードしたモデルをTensorflowで直接推論して動作を確認します。
この手順は実施しなくても問題ありません。


```python
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants

# TensorFlowによるモデルのロード
session = tf.keras.backend.get_session()
tf_model = tf.saved_model.loader.load(session, [tag_constants.SERVING], 'predict/1');

# input/outputのシグネチャ名確認
model_signature = tf_model.signature_def['serving_default']
input_signature = model_signature.inputs
output_signature = model_signature.outputs

for k in input_signature.keys():
    print(k)
for k in output_signature.keys():
    print(k)
```


```python
# input/output Tensorの取得
input_tensor_name = input_signature['inputs'].name
label_tensor_name = output_signature['dense_1_1/Softmax:0'].name

input_name = session.graph.get_tensor_by_name(input_tensor_name)
label_name = session.graph.get_tensor_by_name(label_tensor_name)
```


```python
# 推論の実行
import numpy as np
from text_vectorian import SentencePieceVectorian

vectorian = SentencePieceVectorian()
max_len = 64
features = np.zeros((1, max_len))
inputs = vectorian.fit('これはグルメです。').indices

for i, index in enumerate(inputs):
    pos = max_len - len(inputs) + i
    features[0, pos] = index

label_pred = session.run([label_name], feed_dict={input_name: features})
label_pred
```

## 推論用スクリプトを含めてパッケージング

推論用スクリプトと学習したモデルを一緒に`model.tar.gz`としてパッケージングします。
また、推論結果を人が読んで理解できるラベルにマッピングするためdictである`labels.pickle`も`codeディレクトリ`に含めるようにすることで、推論用スクリプトから参照できるようにします。


```python
!cat predict/code/inference.py
```

```python
    import os
    import io
    import json
    import requests
    import logging
    import numpy as np
    import pickle
    import pandas as pd
    from text_vectorian import SentencePieceVectorian

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    vectorian = SentencePieceVectorian()
    input_len = 64
    dim = 100

    def handler(data, context):
        """Handle request.
        Args:
            data (obj): the request data
            context (Context): an object containing request and configuration details
        Returns:
            (bytes, string): data to return to client, (optional) response content type
        """
        processed_input = _process_input(data, context)
        response = requests.post(context.rest_uri, data=processed_input)

        return _process_output(response, context)


    def _process_input(data, context):
        if context.request_content_type == 'application/json':
            body = data.read().decode('utf-8')

            param = json.loads(body)
            query = param['q']
            features = np.zeros((1, input_len))
            inputs = vectorian.fit(query).indices

            for i, index in enumerate(inputs):
                if i >= input_len:
                    break
                pos = input_len - len(inputs) + i
                features[0, pos] = index

            return json.dumps({
                'inputs': features.tolist()
            })

        raise ValueError('{"error": "unsupported content type {}"}'.format(
            context.request_content_type or "unknown"))


    def _process_output(data, context):
        if data.status_code != 200:
            raise ValueError(data.content.decode('utf-8'))

        response_content_type = 'application/json'

        body = json.loads(data.content.decode('utf-8'))
        predicts = body['outputs'][0]

        labels_path = '/opt/ml/model/code/labels.pickle'

        with open(labels_path, mode='rb') as f:
            labels = pickle.load(f)
        rets = _create_response(predicts, labels)

        logger.warn(rets)

        return json.dumps(rets), response_content_type

    def _create_response(predicts, labels):
        rets = []

        for index in np.argsort(predicts)[::-1]:
            label = labels['index2label'][index]
            prob = predicts[index]
            rets.append({
                'label': label,
                'prob': prob
            })

        return rets
```


```python
!cd predict; mv labels.pickle code
!cd predict; tar zcvf model.tar.gz 1 code
```

```
    1/
    1/variables/
    1/variables/variables.index
    1/variables/variables.data-00000-of-00001
    1/saved_model.pb
    code/
    code/inference.py
    code/.ipynb_checkpoints/
    code/.ipynb_checkpoints/requirements-checkpoint.txt
    code/.ipynb_checkpoints/inference-checkpoint.py
    code/labels.pickle
    code/requirements.txt
```


```python
import urllib

predict_model_url = urllib.parse.urlparse(f'{estimator.output_path}/{estimator.latest_training_job.job_name}/predict/model.tar.gz')
bucket.upload_file('predict/model.tar.gz', predict_model_url.path[1:])
```

## エンドポイントの作成

`sagemaker.tensorflow.serving.Model`を利用してdeployを行います。
この際、以下の注意事項があります。

### frame_versionのバージョンによってはPythonの`f-string`が使えない

`framework_version`に`1.13`を指定します。`1.12`だとPythonのバージョンが`3.5`であり`f-string`が使えないため要注意です。

### インスタンスタイプの指定

`ml.t2.mideum`ではメモリ不足で起動しなかったため、`ml.t2.large`にしています。
なお、2019/07時点ではデプロイ時にt3系インスタンスを指定することができません。


```python
from sagemaker.tensorflow.serving import Model

tensorflow_serving_model = Model(model_data=f'{predict_model_url.scheme}://{predict_model_url.hostname}{predict_model_url.path}',
                                 role=ROLE,
                                 framework_version='1.13')
```


```python
predictor = tensorflow_serving_model.deploy(initial_instance_count=1,
                                            instance_type='ml.t2.large',
                                            tags=TAGS)
```

### エンドポイントによる推論結果確認

boto3を使用してエンドポイントに文字列を入力することで、意図した推論結果が得られることを確認します。


```python
import json

client = boto3.client('sagemaker-runtime')
query = {
    'q': '電波が悪い'
}
res = client.invoke_endpoint(
    EndpointName=predictor.endpoint,
    Body=json.dumps(query),
    ContentType='application/json',
    Accept='application/json'
)
body = res['Body']
ret = json.load(body)
print(ret)
```

```
    [{'label': '携帯電話', 'prob': 0.96622}, {'label': '京都', 'prob': 0.0231401}, {'label': 'グルメ', 'prob': 0.00758497}, {'label': 'スポーツ', 'prob': 0.00305547}]
```

## エンドポイントの削除

最後に不要になったエンドポイントを削除します。
エンドポイントの利用を継続する場合は、実施不要です。


```python
predictor.delete_endpoint()
```

## 参考文献

* [SageMaker TensorFlow Serving Container](https://github.com/aws/sagemaker-tensorflow-serving-container/blob/master/README.md)
* [SageMakerでTensorFlow+Kerasによる独自モデルをトレーニングする方法](https://www.inoue-kobo.com/aws/sagemaker-with-mymodel/index.html#hyperparametertuner)