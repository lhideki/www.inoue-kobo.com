---
title: '約0.2秒で文章を判定するTypeSafe AIのJevを試してみる'
date: '2026-09-19'
thumbnail: 'llm/typesafe-jev/images/thumbnail.png'
tags:
    - 'TypeSafe AI'
    - 'Jev'
    - 'Python'
    - 'LLM'
---

# 約0.2秒で文章を判定するTypeSafe AIのJevを試してみる

[TypeSafe AIのJev](https://docs.typesafe.ai/introduction)は、分類先やスコアなど、プログラムが次の処理を決めるための判断結果を高速に返すことに特化したAIモデルです。ChatGPT(GPT)やClaudeなどの従来のLLMとは異なり、生成モデルとして機能するのではなく、あらかじめ定義した選択肢からの選択やスコアリングに特化しています。業務システムなどの組み込みで単独でも利用可能ですが、従来型のLLMと組み合わせて利用することで、コストや応答時間の面での削減が期待できます。

今回、桃太郎のあらすじを使って、PythonからJevを利用する方法と処理速度を確認してみました。

## Jevの特徴

TypeSafe AIでは、Jevのような判断に特化したモデルを「System Oneモデル」と呼んでいます。入力する文章と判定基準を用意し、以下の3種類の質問を使い分けます。返された値は、処理の分岐や振り分け、優先順位の決定などに利用できます。

| 種類 | 用途 | 主な戻り値 |
| --- | --- | --- |
| `Noul` | はい・いいえの判定 | `noul`(「はい」である確率) |
| `Choice` | 選択肢から1つを選ぶ | `choice`(選択結果)、`probabilities`(確率分布)、`confidence`(信頼度) |
| `Score` | 順序のある基準に沿って評価する | `score`(スコア)、`probabilities`(確率分布)、`confidence`(信頼度)、`legend`(各段階の基準) |

こうした判断をシステムの処理に組み込む場合、応答の速さが重要になります。一般的な自己回帰型LLMはトークンを順番に生成しますが、Jevは定義した選択肢に対する確率を並列に出力します。複数の独立した質問も1回のリクエストで評価できるため、自由文の生成やAPIの繰り返し呼び出しにかかる時間を抑えられます。

主な仕様は以下のとおりです。[公式のModels](https://docs.typesafe.ai/models)、[Choice](https://docs.typesafe.ai/primitives/choice)、[Score](https://docs.typesafe.ai/primitives/score)に記載された内容をまとめています。

| 項目 | 仕様 |
| --- | --- |
| モデル | Jev 1.13 (`jev-1.13.0`) |
| コンテキストサイズ(リクエスト全体) | `state`とすべての質問の合計で最大64kトークン |
| コンテキストサイズ(質問ごと) | `state`と最も長い質問の合計で最大32kトークン |
| 入力形式 | テキスト。文字列、JSONオブジェクト、配列を指定可能。画像・音声・動画の直接入力には非対応 |
| Choiceの選択肢 | 1問につき最大255個 |
| Scoreの評価段階 | 1問につき2〜10段階。スコア値は0始まり |
| 料金 | 入力100万トークンあたり0.042米ドル。出力トークンは無料 |
| レート制限 | 毎秒250,000トークン、毎分1,200リクエスト |
| 対応言語 | 英語を中心に学習。日本語を含むCJKのテキストも扱えるが、精度は言語によって異なる |

コンテキストサイズは、入力する本文(`state`)と最も長い質問の合計で32kトークン以内、本文と全質問の合計で64kトークン以内という2つの条件があります。長い文章に複数の質問を付ける場合は、両方の上限を確認する必要があります。単位は文字数ではなくトークン数です。

## Jevの料金とLLMの比較

Jevの料金(2026年9月時点)は、入力100万トークン(MTok)当たり0.042ドル、出力トークンは無料となっています。以下は従来のLLMの料金との比較です。

| 提供元 | モデル | 入力単価(米ドル/MTok) | 出力単価(米ドル/MTok) | Jev比(入力単価) |
| --- | --- | ---: | ---: | ---: |
| TypeSafe AI | Jev | 0.042 | 無料 | 1倍 |
| OpenAI | GPT-5.6 Luna | 0.20 | 1.20 | 約4.8倍 |
| OpenAI | GPT-5.6 Terra | 2.00 | 12.00 | 約47.6倍 |
| OpenAI | GPT-5.6 Sol | 4.00 | 20.00 | 約95.2倍 |
| Anthropic | Claude Haiku 4.5 | 1.00 | 5.00 | 約23.8倍 |
| Anthropic | Claude Sonnet 5 | 2.00 | 10.00 | 約47.6倍 |
| Anthropic | Claude Fable 5.1 | 10.00 | 50.00 | 約238.1倍 |

用途が異なるため単純に比較するものではありませんが、Jevの低コストと低レイテンシは、分類や選択、スコアリングといった特定のタスクにおいて大きな利点となります。

## 前提条件

今回は以下の環境で確認しました。

* macOS / Apple Silicon(arm64)
* Python 3.13.14
* typesafe-sdk==0.6.0
* python-dotenv==1.2.3
* TypeSafeのAPIキーを取得済みであること

APIキーは[TypeSafeのコンソール](https://console.typesafe.ai/)から取得できます。手順は[公式Quick start](https://docs.typesafe.ai/introduction/quickstart)を参照してください。

以下のコードでは`jev-latest`を指定していますが、処理時間の測定では`jev-1.13.0`に固定しています。`jev-latest`が指すモデルは更新されるため、同じモデルで比較する場合はバージョンを指定してください。実際に使用されたモデルは`response.model`で確認できます。

## 事前準備

まず、必要なパッケージをインストールします。以下はmacOS/Linuxでの例です。

```bash
mkdir jev-example
cd jev-example
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install typesafe-sdk==0.6.0 python-dotenv==1.2.3
```

作業ディレクトリに`.env`を作成し、APIキーを記載します。

```dotenv
TYPESAFE_API_KEY=your-api-key-here
```

サンプルコードは以下からダウンロードできます。Pythonスクリプトの場合は、`.env`と同じディレクトリに保存して`python typesafe_examples.py`で実行します。全体を実行すると、4回のAPIリクエストが発生します。

* [Pythonスクリプト](./sample/typesafe_examples.py)
* [ノートブック](./sample/typesafe_examples.ipynb)

ノートブックを利用する場合は、仮想環境に`ipykernel`を追加し、JupyterやVS Codeでカーネルに選択します。作業ディレクトリは`.env`のある場所にしてください。

```bash
python -m pip install ipykernel
```

以降のコードは上から順に実行します。まず、環境変数を読み込みます。

```python
from dotenv import load_dotenv

load_dotenv(".env")
```

判定対象の文章を用意します。今回は、同じ桃太郎のあらすじに対して質問を変え、二択判定・分類・段階評価の違いを確認します。

```python
momotaro_story = (
    "昔々、あるところにおじいさんとおばあさんが住んでいました。おばあさんが川で洗濯をしていると、"
    "大きな桃がどんぶらこと流れてきました。家に持ち帰り桃を割ると、中から元気な男の子が生まれ、"
    "二人は桃太郎と名付けて育てました。桃太郎はすくすくと成長し、ある日、鬼ヶ島の鬼たちが村人から"
    "金銀財宝を奪い、乱暴を働いていることを知ります。桃太郎は鬼退治を決意し、おばあさんが作ってくれた"
    "きび団子を持って旅に出ました。道中、犬・猿・雉に出会い、きび団子を分け与えることで仲間にします。"
    "桃太郎たちは力を合わせて鬼ヶ島に上陸し、鬼の大将と激しく戦った末に鬼たちを降伏させ、奪われていた"
    "財宝を取り戻しました。桃太郎たちは財宝を持って村に帰り、おじいさんとおばあさんや村人たちと共に"
    "平和な暮らしを取り戻しました。"
)
```

この文章を`state`に、質問を`questions`に渡します。それぞれの質問には、何を判定するかを`instructions`で、選択肢や判定基準を`criteria`で指定します。業務データを扱う場合も、この文章と質問を置き換えることで同じように利用できます。

以下は実行例です。

## Noulで二択判定する

[Noul](https://docs.typesafe.ai/primitives/noul)を使って、動物たちの協力によって鬼退治に成功したかを判定します。

```python
from typesafe_sdk import Noul, TypeSafeClient

with TypeSafeClient() as client:
    response = client.system_one(
        model="jev-latest",
        state=momotaro_story,
        questions={
            "succeeded_with_animal_help": Noul(
                instructions="桃太郎は犬・猿・雉など動物たちの協力を得て鬼退治に成功しましたか？",
                criteria={
                    "true": "動物たちが仲間になり、鬼退治の成功に貢献している",
                    "false": "動物の協力がない、または鬼退治に失敗している",
                },
            ),
        },
    )

print(response.nouls["succeeded_with_animal_help"].noul)
```

出力結果は以下のとおりです。

```text
0.99
```

「動物たちの協力で鬼退治に成功した」という質問に対し、「はい」の確率が0.99と返りました。`succeeded_with_animal_help`は質問に付けた名前で、同じ名前を指定して結果を取り出します。`.noul`の値は1に近いほど「はい」、0に近いほど「いいえ」に傾いた判定です。

## Choiceで分類する

[Choice](https://docs.typesafe.ai/primitives/choice)を使って、物語のジャンルを分類します。

```python
from typesafe_sdk import Choice, TypeSafeClient

with TypeSafeClient() as client:
    response = client.system_one(
        model="jev-latest",
        state=momotaro_story,
        questions={
            "genre": Choice(
                instructions="この物語のジャンルとして最も適切なものはどれですか？",
                criteria={
                    "昔話・おとぎ話": "口承で伝わる伝統的な民話、不思議な出来事や教訓を含む",
                    "恋愛小説": "登場人物の恋愛関係が主題の物語",
                    "推理小説": "謎解きや犯人捜しが主題の物語",
                    "伝記": "実在した人物の生涯を記録した書物",
                },
            ),
        },
    )

result = response.choices["genre"]
print(result.choice)
print(result.probabilities)
print(result.confidence)
```

出力結果は以下のとおりです。

```text
昔話・おとぎ話
{'伝記': 0.0, '恋愛小説': 0.0, '昔話・おとぎ話': 1.0, '推理小説': 0.0}
1.0
```

上から順に、`.choice`で取得した選択結果、`.probabilities`の確率分布、`.confidence`の信頼度です。「昔話・おとぎ話」の確率が1.0で、ほかの3種類は0.0になっています。`criteria`のキーが選択肢、値がその説明です。返される選択結果も、このキーになります。

## Scoreで段階評価する

[Score](https://docs.typesafe.ai/primitives/score)を使って、鬼が村にもたらす脅威を評価します。今回は以下の3段階としました。

| スコア値 | 基準 |
| --- | --- |
| 0 | 村人に軽いいたずらをする程度 |
| 1 | 村人の生活や財産に大きな被害を与えている |
| 2 | 村人の命や安全を脅かすほど深刻な脅威となっている |

```python
from typesafe_sdk import Score, TypeSafeClient

with TypeSafeClient() as client:
    response = client.system_one(
        model="jev-latest",
        state=momotaro_story,
        questions={
            "oni_threat_level": Score(
                instructions="鬼ヶ島の鬼が村にもたらしていた脅威はどの程度深刻でしたか？",
                criteria=[
                    {
                        # スコア値 0
                        "what": "村人に軽いいたずらをする程度",
                        "examples": ["作物を少し荒らす", "夜中に物音を立てて驚かせる"],
                    },
                    {
                        # スコア値 1
                        "what": "村人の生活や財産に大きな被害を与えている",
                        "examples": ["金銀財宝を略奪する", "家屋を破壊する"],
                    },
                    {
                        # スコア値 2
                        "what": "村人の命や安全を脅かすほど深刻な脅威となっている",
                        "examples": ["村人に暴力を振るう", "村を占拠し支配する"],
                    },
                ],
            ),
        },
    )

result = response.scores["oni_threat_level"]
print(result.score)
print(result.legend)
print(result.probabilities)
print(result.confidence)
```

出力結果は以下のとおりです。

```text
1.3
{0: {'what': '村人に軽いいたずらをする程度', 'examples': ['作物を少し荒らす', '夜中に物音を立てて驚かせる']},
 1: {'what': '村人の生活や財産に大きな被害を与えている', 'examples': ['金銀財宝を略奪する', '家屋を破壊する']},
 2: {'what': '村人の命や安全を脅かすほど深刻な脅威となっている',
     'examples': ['村人に暴力を振るう', '村を占拠し支配する']}}
{0: 0.0, 1: 0.7, 2: 0.3}
0.54
```

上から順に、`.score`で取得したスコア、`.legend`の各段階の基準、`.probabilities`の確率分布、`.confidence`の信頼度です。レベル1「生活や財産への大きな被害」に0.7、レベル2「命や安全への深刻な脅威」に0.3の確率が付いています。スコアの番号は`criteria`の配列の位置で決まります。先頭が0なので、今回の範囲は0〜2です。`what`と`examples`は、基準を説明と具体例に分けて書くために使用しています。

スコアは各段階の番号を確率で重み付けした平均なので、小数になることがあります。上記の出力では、以下の計算で1.3になります。

```text
0 × 0.0 + 1 × 0.7 + 2 × 0.3 = 1.3
```

Python SDK 0.6.0では、`legend`のキーは整数です。`result.legend[2]`で、レベル2の`what`と`examples`を含む辞書を取得できます(HTTP APIのJSONではキーは文字列です)。

障害の影響度や問い合わせの緊急度を評価する場合も、同じように段階ごとの基準を定義し、スコアを使って対応の優先順位を付けることができます。

## 複数の質問をまとめて実行する

今度は、同じ文章に対する複数の判断を1回のリクエストにまとめてみます。Noulで「ハッピーエンドか」、Choiceで「主人公の性格」、Scoreで「教訓性」を判定します。

```python
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

with TypeSafeClient() as client:
    response = client.system_one(
        model="jev-latest",
        state=momotaro_story,
        questions={
            "is_happy_ending": Noul(
                instructions="この物語はハッピーエンドで終わりますか？"
            ),
            "protagonist_trait": Choice(
                instructions="主人公の桃太郎の性格として最も近いものはどれですか？",
                criteria={"勇敢": None, "臆病": None, "狡猾": None, "怠惰": None},
            ),
            "moral_lesson_strength": Score(
                instructions="この物語の教訓性（子供向けの教育的価値）はどの程度強いですか？",
                criteria=[
                    {
                        # スコア値 0
                        "what": "教訓性はほとんどない",
                        "examples": ["単なる出来事の羅列で教訓的な結論がない"],
                    },
                    {
                        # スコア値 1
                        "what": "ある程度の教訓が含まれている",
                        "examples": ["努力や協力の大切さが部分的に読み取れる"],
                    },
                    {
                        # スコア値 2
                        "what": "勧善懲悪など明確な教訓が中心テーマとなっている",
                        "examples": ["悪が裁かれ、善が報われる結末が明示されている"],
                    },
                ],
            ),
        },
    )

score_result = response.scores["moral_lesson_strength"]
print(response.nouls["is_happy_ending"].noul)
print(response.choices["protagonist_trait"].choice)
print(score_result.score)
print(score_result.legend)
print(response.model)
```

出力例は以下のとおりです。

```text
0.99
勇敢
1.96
{0: {'what': '教訓性はほとんどない', 'examples': ['単なる出来事の羅列で教訓的な結論がない']},
 1: {'what': 'ある程度の教訓が含まれている', 'examples': ['努力や協力の大切さが部分的に読み取れる']},
 2: {'what': '勧善懲悪など明確な教訓が中心テーマとなっている',
     'examples': ['悪が裁かれ、善が報われる結末が明示されている']}}
jev-1.13.0
```

## 処理時間を比較する

桃太郎のあらすじに対して、3問(動物の協力・ジャンル・鬼の脅威)を個別に送信した場合と、まとめて送信した場合を比較しました。さらに、結末・主人公の性格・教訓性を追加し、6問でも測定しています。

モデルは`jev-1.13.0`を使用し、同じクライアントでウォームアップ後に各20回測定しました。以下は通信時間を含む応答時間の中央値です。

| 送信方法 | APIリクエスト数 | 処理時間(中央値) |
| --- | ---: | ---: |
| 3問を個別に順次送信 | 3回 | 679.8ms(3回の合計) |
| 同じ3問を一括送信 | 1回 | 212.4ms |
| 6問を一括送信 | 1回 | 243.6ms |

3問をまとめると約3.2倍速くなり、6問に増やしても約0.24秒でした。独立した質問はまとめて送信することで、複数の判断を短い待ち時間で取得できることが確認できました。

## 業務システムやLLMと組み合わせる

今回の例を業務システムに置き換えると、問い合わせの分類や、処理の実行条件、対応の優先度などをJevに判定させる構成が考えられます。判定結果は選択肢や数値で返るので、既存のプログラムから利用しやすいと思います。

| 利用方法 | Jevに任せる処理 | 判定後の処理 |
| --- | --- | --- |
| 業務システムへの組み込み | 担当部署や緊急度の判定 | 分類結果やスコアに従って担当部署へ振り分ける |
| LLMとの組み合わせ | 問い合わせの種類や、回答に必要な処理の判定 | データベース検索などで対応できるものは既存処理へ、文章生成が必要なものはLLMへ渡す |
| 人による確認との組み合わせ | 分類や段階評価と、その信頼度の算出 | 信頼度が低い場合は自動処理を保留し、担当者に確認を依頼する |

例えば、注文状況の確認はデータベース検索へ、商品に関する説明はLLMへ振り分ける、といった使い分けです。[公式のIntent routing](https://docs.typesafe.ai/patterns/intent-routing)でも、Jevの判定結果に応じて既存処理・LLM・人へ振り分ける例が紹介されています。

このように、LLMで行っていた分類や振り分けをJevに任せ、LLMを呼び出す回数や処理を減らせれば、全体のコストや応答時間の削減が期待できます。他にも、Jevを判断処理に利用する実験的なプログラミング言語[Probably](https://probably-lang.southpolesteve.workers.dev/)も登場しており、様々な応用例が考えられそうです。現在はテキストのみですが、今後のマルチモーダル対応にも期待したいところです。LLMにおける久々のブレークスルーという感じです。

## 参考文献

* [TypeSafe AI公式ドキュメント](https://docs.typesafe.ai/introduction)
* [TypeSafe Python SDK](https://docs.typesafe.ai/sdk/python)
* [Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
