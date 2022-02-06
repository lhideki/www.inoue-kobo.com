# Huggingface Transformersによる日本語GPT-2モデルのrinnaを利用した推論の例

## TL;DR

[Huggingface Transformers](https://huggingface.co/)により、日本語GPT-2モデルである[rinnaの公開モデル](https://huggingface.co/rinna/japanese-gpt2-medium)で以下の推論を行う場合のサンプルです。

* Zero-shot Learning
* Few-shot Learning
* 文書生成

## モジュールのインストール

```python
# Huggingface Transformersのインストール
!pip install transformers==4.4.2
# Sentencepieceのインストール
!pip install sentencepiece==0.1.91
```

## Modelのロード

```python
from transformers import T5Tokenizer, AutoModelForCausalLM

tokenizer = T5Tokenizer.from_pretrained('rinna/japanese-gpt2-medium')
tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading

model = AutoModelForCausalLM.from_pretrained('rinna/japanese-gpt2-medium')
```

## Zero-shot Learning

```python
original_text = '''全身がだるい ='''

input = tokenizer.encode(original_text, return_tensors='pt')
output = model.generate(input, do_sample=False, max_length=60, num_return_sequences=1)
generated = tokenizer.batch_decode(output, skip_special_tokens=True)

predict_part = generated[0][(len(original_text.strip())):]
predict_part = predict_part[:predict_part.find(' ')]
print(original_text.strip() + ' ' + predict_part)
```

```
全身がだるい = 疲れている
```

```python
original_text = '''病名:
全身がだるい =
'''

input = tokenizer.encode(original_text, return_tensors='pt')
output = model.generate(input, do_sample=False, max_length=60, num_return_sequences=1)
generated = tokenizer.batch_decode(output, skip_special_tokens=True)

predict_part = generated[0][(len(original_text.strip())):]
predict_part = predict_part[:predict_part.find(' ')]
print(original_text.strip() + ' ' + predict_part)
```

```
病名:
全身がだるい = 風邪
```

## Few-shot Learning

```python
original_text = '''病名:
熱がある = 発熱
熱が高い = 高熱
風邪を引いた = 感冒
全身がだるい =
'''

input = tokenizer.encode(original_text, return_tensors='pt')
output = model.generate(input, do_sample=False, max_length=60, num_return_sequences=1)
generated = tokenizer.batch_decode(output, skip_special_tokens=True)

predict_part = generated[0][(len(original_text.strip())):]
predict_part = predict_part[:predict_part.find(' ')]
print(original_text.strip() + ' ' + predict_part)
```

```
病名:
熱がある = 発熱
熱が高い = 高熱
風邪を引いた = 感冒
全身がだるい = 倦怠感
```

## 文書生成

```python
original_text = '''本日はお日柄も良く、'''

input = tokenizer.encode(original_text, return_tensors='pt')
output = model.generate(input, do_sample=True, max_length=100, num_return_sequences=1, truncation=True)
generated = tokenizer.batch_decode(output, skip_special_tokens=True)

predict_part = generated[0][(len(original_text.strip())):]
print(original_text.strip() + ' ' + predict_part[0:predict_part.find('。')] + '。')
```

```
本日はお日柄も良く、 朝からたくさんの方にお越しいただき、 幸せいっぱいのオープンから無事に終了致しました。
```
