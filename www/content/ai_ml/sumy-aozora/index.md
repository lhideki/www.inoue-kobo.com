---
title: "sumyを使って青空文庫を要約してみる"
date: "2019-05-04"
tags:
  - "AI/ML"
  - "NLP"
thumbnail: "ai_ml/sumy-aozora/images/thumbnail.png"
---
# sumyを使って青空文庫を要約してみる

## TL;DR

テキスト要約モジュールである[sumy](https://github.com/miso-belica/sumy)を使って青空文庫の書籍を要約してみました。

`sumy`を使う部分は[け日記 - Python: LexRankで日本語の記事を要約する](https://ohke.hateblo.jp/entry/2018/11/17/230000)を参考にさせていただきました。
同記事ではTokenizerに`Janome`を使用していますが、今回は[ginza(spacy)](https://github.com/megagonlabs/ginza)を使用しています。

### 実行環境

* Python 3.6.8
* sumy 0.7.0
* ginza(spacy) 1.0.2
* beautifulsoup4 4.7.1

## ソースコード

### spacyのロード

```python
import spacy

nlp = spacy.load('ja_ginza_nopn')
```

### 青空文庫のスクレイピング

```python
from urllib import request
from bs4 import BeautifulSoup
import bs4

# urlに要約対象とする書籍のURLを指定します。以下は「だしの取り方 by 北大路魯山人」のURLです。
url = 'https://www.aozora.gr.jp/cards/001403/files/49986_37674.html'
html = request.urlopen(url)
soup = BeautifulSoup(html, 'html.parser')
body = soup.select('.main_text')
```

```python
text = ''
for b in body[0]:
    if type(b) == bs4.element.NavigableString:
        text += b
        continue
    # ルビの場合、フリガナは対象にせずに、漢字のみ使用します。
    text += ''.join([e.text for e in b.find_all('rb')])
```

### 分析用コーパスの準備

活用などで表記が異なると違う単語として計算されてしまうため、分析用コーパスではレンマ化します。

```python
corpus = []
originals = []
doc = nlp(text)
for s in doc.sents:
    originals.append(s)
    tokens = []
    for t in s:
        tokens.append(t.lemma_)
    corpus.append(' '.join(tokens))

print(len(corpus))
print(len(originals))
```

### sumyによる要約

```python
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# 連結したcorpusを再度tinysegmenterでトークナイズさせる
parser = PlaintextParser.from_string(''.join(corpus), Tokenizer('japanese'))

summarizer = LexRankSummarizer()
summarizer.stop_words = [' ']  # スペースも1単語として認識されるため、ストップワードにすることで除外する

# sentencres_countに要約後の文の数を指定します。
summary = summarizer(document=parser.document, sentences_count=3)

# 元の文を表示
for sentence in summary:
    print(originals[corpus.index(sentence.__str__())])
```

## まとめ

`だしの取り方 by 北大路魯山人`を3行で要約すると以下のようになりました。良いだしを取るためには、鉋(かんな)から準備しましょう。

```
どんなふうに削ったのがいいだしになるかというと、削ったかつおぶしがまるで雁皮紙のごとく薄く、ガラスのように光沢のあるものでなければならない。
現在、鉋でかつおぶしを削っているのは料理屋のみであって、たいがいは道具もなくて我慢しているようである。
鉋があっても、切れない場合が多いし、それを使用して削れないと思うくらいなら、日本料理をやめた方がいい。
```

`蜘蛛の糸 by 芥川龍之介`の場合は以下のとおりです。

```
陀多は両手を蜘蛛の糸にからみながら、ここへ来てから何年にも出した事のない声で、「しめた。
ところがふと気がつきますと、蜘蛛の糸の下の方には、数限もない罪人たちが、自分ののぼった後をつけて、まるで蟻の行列のように、やはり上へ上へ一心によじのぼって来るではございませんか。
が、そう云う中にも、罪人たちは何百となく何千となく、まっ暗な血の池の底から、うようよと這い上って、細く光っている蜘蛛の糸を、一列になりながら、せっせとのぼって参ります。
```

## 参考文献

* [け日記 - Python: LexRankで日本語の記事を要約する](https://ohke.hateblo.jp/entry/2018/11/17/230000)
* [sumy](https://github.com/miso-belica/sumy)
* [ginza(spacy)](https://github.com/megagonlabs/ginza)