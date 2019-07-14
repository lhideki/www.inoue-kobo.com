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
