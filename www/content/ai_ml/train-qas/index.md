---
title: 'Huggingfaceで公開されている日本語モデルを使ってQAタスクをファインチューニングする'
date: '2023-01-14'
tags:
    - 'Huggingface'
    - 'AI/ML'
    - 'NLP'
    - 'JGLUE'
thumbnail: 'ai_ml/train-qas/images/thumbnail.png'
---

# Huggingface で公開されている日本語モデルを使って QA タスクをファインチューニングする

## TL;DR

Huggingface で公開されている事前学習済み日本語モデルを利用し、Question-Answering タスク用のデータセットでファインチューニングする際のサンプルコードです。

Question-Answering タスク用のデータセットは[JGLUE](https://github.com/yahoojapan/JGLUE)の`JSQuAD`を利用しています。

JSQuAD は以下のようなデータセットです。

![](images/jsquad-sample.png)

事前学習モデル済み日本語モデルは以下を利用しています。

-   [ku-nlp/deberta-v2-base-japanese](https://huggingface.co/ku-nlp/deberta-v2-base-japanese)

## モジュールを準備する

```python
!pip install -U pip
!pip install pandas
!pip install transformers
!pip install torch torchvision torchaudio
!pip install datasets
```

## データセットを読み込む

JGLUE の JSQuAD は`data`配下に準備してある前提です。

```python
import pandas as pd
import json
import numpy as np

train_filename = 'data/jsquad-v1.0/train-v1.0.json'
test_filename = 'data/jsquad-v1.0/valid-v1.0.json'

# JSQuADはSQuADと同じ形式であるため、SQuADからDataFrameに変換するコードを利用できます。
# https://www.kaggle.com/code/sanjay11100/squad-stanford-q-a-json-to-pandas-dataframe
def squad_json_to_dataframe_train(input_file_path, record_path=['data', 'paragraphs', 'qas', 'answers']):
    file = json.loads(open(input_file_path).read())
    m = pd.io.json.json_normalize(file, record_path[:-1])
    r = pd.io.json.json_normalize(file, record_path[:-2])

    # combining it into single dataframe
    idx = np.repeat(r['context'].values, r.qas.str.len())
    m['context'] = idx
    m['answers'] = m['answers'].apply(lambda x: x[0])

    return m[['question', 'context', 'answers']]


train_df = squad_json_to_dataframe_train(train_filename).sample(1000) # 全データで学習すると時間がかかるため1000だけ抽出しています。
test_df = squad_json_to_dataframe_train(test_filename).sample(100) # 評価用として100だけ取り出しています。

display(train_df)
display(test_df)
```

## 事前学習モデルを準備する

```python
model_ckpt = 'ku-nlp/deberta-v2-base-japanese'

model = AutoModelForQuestionAnswering.from_pretrained(model_ckpt)
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
```

## データ形式を変換する

`max_length`はモデルが許容できる最大の sequence サイズを指定します。

```python
max_length = 512  # The maximum length of a feature (question and context)
doc_stride = (
    0  # The authorized overlap between two part of the context when splitting
)

# Huggingfaceのチュートリアルで提示されているコードで学習用のデータ形式に変換します。
# https://huggingface.co/docs/transformers/tasks/question_answering
def prepare_train_features(examples):
    #
    # Tokenize our examples with truncation and padding, but keep the overflows using a
    # stride. This results in one example possible giving several features when a context is long,
    # each of those features having a context that overlaps a bit the context of the previous
    # feature.
    examples['question'] = [q.strip() for q in examples['question']]
    examples['context'] = [c.strip() for c in examples['context']]
    inputs = tokenizer(
        text=examples['question'],
        text_pair=examples['context'],
        truncation='only_second',
        max_length=max_length,
        stride=doc_stride,
        return_offsets_mapping=True,
        padding='max_length',
    )

    offset_mapping = inputs.pop('offset_mapping')
    answers = examples['answers']
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer['answer_start']
        end_char = answer['answer_start'] + len(answer['text'])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs['start_positions'] = start_positions
    inputs['end_positions'] = end_positions

    return inputs
```

```python
from datasets import Dataset

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
```

```python
tokenized_train_dataset = train_dataset.map(
    prepare_train_features,
    batched=True,
    remove_columns=train_dataset.column_names,
    num_proc=3,
)
tokenized_test_dataset = test_dataset.map(
    prepare_train_features,
    batched=True,
    remove_columns=test_dataset.column_names,
    num_proc=3,
)
```

## トレーニングする

```python
from transformers import DefaultDataCollator

data_collator = DefaultDataCollator()
```

```python
import os

os.environ['WANDB_DISABLED'] = 'true'

training_args = TrainingArguments(
    output_dir='./outputs',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
```

## 推論して評価する

```python
from transformers import pipeline

pipe = pipeline('question-answering', model=trainer.model,
                tokenizer=trainer.tokenizer)
```

pipeline で推論する際に`align_to_words=False`を指定するのは注意事項です。なお、`transformers==4.25.1`未満の場合`align_to_words=False`が機能しない事象を確認しているため、transformers のバージョンも注意が必要です。

```python
def predict(row):
    ret_df = pd.DataFrame()
    ret = pipe(question=row['question'], context=row['context'],
               top_k=1, handle_impossible_answer=False,  align_to_words=False)

    return ret['answer'], ret['score']


pred_df = test_df[['question', 'context', 'answers']].copy()
pred_df['actual_answer'] = pred_df['answers'].apply(lambda x: x['text'])
pred_df[['pred_answer', 'prob']] = pred_df.apply(
    predict, axis=1, result_type='expand')
pred_df = pred_df.drop('answers', axis=1)
pred_df['correct'] = pred_df.apply(
    lambda row: row['pred_answer'] == row['actual_answer'], axis=1)
```

```python
pd.set_option('display.max_colwidth', 300)

print(f'Accuracy: {len(pred_df[pred_df.correct]) / len(pred_df)}')
display(pred_df)
```

![](images/predict-sample.png)
