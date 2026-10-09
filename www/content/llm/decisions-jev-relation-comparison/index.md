---
title: 'OpenAI Decisions APIとJevを精度・コスト・処理速度で比較する'
date: '2026-10-08'
thumbnail: 'llm/decisions-jev-relation-comparison/images/quality-comparison.png'
tags:
    - 'OpenAI'
    - 'TypeSafe AI'
    - 'Jev'
    - 'Knowledge Graph'
    - 'Python'
---

# OpenAI Decisions APIとJevを精度・コスト・処理速度で比較する

[前回のJev評価](https://www.inoue-kobo.com/llm/jev-graph-relation-evaluation/)と同じデータを使い、OpenAI Decisions APIを比較しました。日本語1,418件、英語1,418件について、文章から36候補の関係を1つ選ぶ課題です。

まず、精度・コスト・処理速度の3点をまとめます。

| 比較軸 | Jev | Decisions API |
| --- | --- | --- |
| 精度: Accuracy（日本語 / 英語） | 92.95% / 93.51% | 86.04% / 86.32% |
| コスト: 入力100万トークン単価 | 0.042米ドル | 0.10米ドル |
| コスト: 日英2,836件 | 約0.46米ドル（参考推計） | 約1.13米ドル（API報告トークンから換算） |
| 速度: p50（日本語 / 英語） | 232ms / 247ms（過去の参考値） | 248ms / 233ms |
| 速度: p95（日本語 / 英語） | 311ms / 324ms（過去の参考値） | 349ms / 358ms |

Jevの総費用は現行単価と代替tokenizerによる参考推計で、実請求額ではありません。速度も測定時期・環境が異なります。推計方法と比較条件は後半に記載します。

- **精度**: このデータではJevが日英とも約7ポイント上回りました。
- **コスト**: 公表入力単価はJevが低く、総費用は上表の条件付き推計で比較できます。
- **処理速度**: p50はどちらも約0.2秒台。速度の優劣は同じ環境での再測定が必要です。

## 1. 精度: 今回はJevが高い

![AccuracyとMacro F1の比較。Accuracyのみ参考95%Wilson区間を表示](images/quality-comparison.png)

Accuracyの差は、日本語で6.91ポイント、英語で7.19ポイントでした。関係ごとのF1を平均したMacro F1もJevが上回りました。Accuracyには、後述するDecisionsの拒否回答も分母に含めています。

## 2. コスト: 単価と、同じ入力での参考費用を見る

![入力100万トークン当たりの公表単価](images/input-price-comparison.png)

Jevは入力100万トークン当たり0.042米ドル、Decisionsは0.10米ドルで、どちらも出力トークン料金はありません。今回の入力を代替tokenizerで数えたJevの参考費用は約0.46米ドル、別のtokenizerでは約0.58米ドルでした。

DecisionsはAPIが報告した入力トークン数から約1.13米ドルと換算できます。**Jevは推計、Decisionsは実測トークン換算なので、実際の処理費用の倍率までは確定できません。**

## 3. 処理速度: どちらもp50は約0.2秒台

![API呼び出し時間のp50とp95。Jevは過去の参考値](images/latency-comparison.png)

p50は半分、p95は95%の呼び出しがその時間以内に終わる目安です。記録上、p50は日本語でJev、英語でDecisionsが短く、p95は日英ともJevが短い結果でした。ただし、Jevは2026年9月の保存結果で、同じ環境での速度比較ではありません。

## 選択肢を選ぶAPIなのに、答えないことがある

Decisionsでは、日本語4件（0.28%）、英語27件（1.90%）が`refusal`でした。これは低い信頼度をこちらで除外した結果やHTTPエラーではなく、APIが質問への回答を断ったことを表す応答型です。説明用に`answers`の1要素を示すと、次の形になります。

```json
{"type": "refusal", "name": "relation"}
```

この型には関係名・確率・信頼度・拒否理由のフィールドがありません。今回の31件がなぜ拒否されたかは分からず、安全性や文章の曖昧さが原因だとは断定できません。

Jevの公開されたChoice成功応答は、最高確率の候補と確率・信頼度を返す構造で、同等の`refusal`型は掲載されていません。今回の保存結果も全件が選択回答でしたが、全リクエストで回答を保証する意味ではありません。組み込む際は、**Decisionsの拒否への分岐と、両APIの低信頼度回答を保留する処理を分ける**必要があります。

## 比較条件と計算方法

### 同じ本文・指示・候補で比較したか

主語と目的語が既知の状態で、`(主語, ?, 目的語)`の関係を選びます。CoDEx公式のリンク予測や、本文からの主語・目的語の抽出は含みません。

| 項目 | 揃えた条件 |
| --- | --- |
| 対象 | CoDEx-S test由来の日英共通1,418件。同じtriple ID |
| モデル | Decisionsはgpt-6-lunaを指定し、全応答でも確認。Jevの当時の版は未記録 |
| 本文 | 前回保存した日本語記事抜粋・英語extract。再取得しない |
| 指示文 | 主語・目的語を含め、元のJevコードと同じ文字列 |
| 選択肢 | 対象データの正解関係から定義した36候補。名前・説明・順序を共通化 |
| 各レコードの正解 | APIへ渡さない。本評価で追加学習・few-shot例・閾値調整にも使わない |
| リクエスト | 1件につき1質問。Jevはstate/criteria、Decisionsはinput/choicesに設定 |

日英全2,836件で、本文・指示文・候補名・説明・順序の一致を再確認しました。Decisionsは再構成した送信データのハッシュと、実行時に保存したリクエストハッシュも全件一致しています。Jevの過去のHTTP送信本文は未保存のため、元コードと固定データからの再構成による確認です。

両APIの日本語・英語の指示文は、それぞれ次の形式です。

```text
日本語Wikipedia本文に基づき、主語「{主語の名称}」と目的語「{目的語の名称}」の間に成立する関係を候補から選んでください。

Using the English Wikipedia extract, choose the relation that holds between the subject “{subject label}” and object “{object label}”.
```

**同じ本文と36択を与える比較として、確認できる入力条件は揃っています。** ただし、API内部のテンプレートやトークン化までは揃えられず、Jevの当時のモデル版・SDK版・内部再試行の有無も不明です。モデルだけの差を切り分けた実験ではありません。

### 精度の集計とエラーバー

| 本文 | API | 正解 / 全件 | Accuracy | 参考95%Wilson区間 | Macro F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 日本語 | Jev | 1,318 / 1,418 | 92.95% | 91.50〜94.17% | 0.7478 |
| 日本語 | Decisions | 1,220 / 1,418 | 86.04% | 84.14〜87.74% | 0.6833 |
| 英語 | Jev | 1,326 / 1,418 | 93.51% | 92.11〜94.68% | 0.7295 |
| 英語 | Decisions | 1,224 / 1,418 | 86.32% | 84.43〜88.01% | 0.6546 |

拒否は正答に数えず、各1,418件の分母に残します。Macro F1は固定した36関係で平均し、拒否は正解クラスのFalse Negative、分母が0のクラスのF1は0とします。

選択回答だけを分母にしたDecisionsのAccuracyは、日本語86.28%（1,414件）、英語87.99%（1,391件）でした。拒否を除いてもJevを下回るため、拒否だけが差の原因ではありません。

エラーバーは**各設問を独立な二値試行と仮定した参考95%信頼区間（Wilson法）**です。全1,418件を使い、`z = 1.95996398454`で計算しました。同じ主語・記事の重複があるため独立性は保証されず、一般の業務データへの精度保証やAPI再実行時のばらつきではありません。区間の重なりだけで、両APIの差の統計的有意性を判定することもできません。[計算式（NIST）](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm)

### 費用の実測値と参考推計

| 算定方法 | 日本語の入力トークン数 | 英語の入力トークン数 | 日英合計の換算費用 |
| --- | ---: | ---: | ---: |
| Jev: o200k_base・compact JSON（基準の参考推計） | 8,606,473 | 2,350,218 | 約0.46米ドル |
| Jev: o200k_base・各フィールドを別々に計数 | 8,473,378 | 2,220,863 | 約0.45米ドル |
| Jev: cl100k_base・compact JSON | 11,506,900 | 2,348,299 | 約0.58米ドル |
| Jev: cl100k_base・各フィールドを別々に計数 | 11,375,049 | 2,227,720 | 約0.57米ドル |
| Decisions: API報告値 | 8,833,591 | 2,470,749 | 約1.13米ドル |

Decisionsの入力は日本語8,833,591トークン、英語2,470,749トークンで、拒否分も含みます。標準単価0.10米ドル/100万入力トークンで換算しました。10%の地域追加料金を仮定すると合計約1.24米ドルですが、適用や実請求額は確認していません。

Jevは当時のusageが未保存なので、固定データと元コードから入力を再構成しました。本文・指示文に加え、**36候補の名前・説明を毎回送る分も含めます**。同じ主語の記事が繰り返されても、元の実行は別リクエストなので重複を除きません。

オフラインで[tiktoken](https://github.com/openai/tiktoken) 0.14.0を使い、入力内容をcompact JSONにしたものを`o200k_base`で数え、現行単価0.042米ドル/100万入力トークンを掛けた値を基準としました。JSON方式は本文・指示文・候補名・説明に加え、質問名`relation`などの構造用キーや記号も含む、明示的な仮定です。モデル指定やHTTPメタデータは含めず、Jev内部の形式を復元したものではありません。感度確認として、`cl100k_base`と、構造用キーを含めず文字列フィールドを別々に数えて合算する方法でも計算しました。

o200k_baseとcl100k_baseはいずれも代替tokenizerで、Jev公式の課金用tokenizerではありません。4通りの値は仮定を変えたシナリオであり、実費の上下限や信頼区間ではありません。内部の追加入力や当時の再試行は復元できないため、この推計から実費の比率を断定しません。

単価は[TypeSafe Models](https://docs.typesafe.ai/models)と[Decisionsの料金説明](https://developers.openai.com/api/docs/guides/decisions#pricing-and-availability)に基づきます。Jevの現行モデル`jev-1.13.0`と現行単価を、過去の評価モデルや請求額を特定する情報とは扱っていません。

### 処理時間の計測

Decisionsは2026年10月8日にmacOS上のPythonから逐次実行しました。時間はHTTPSリクエスト開始から応答全体の読み取り・JSON解析までで、初回TCP/TLS接続と拒否も含みます。p50/p95はnearest-rank方式です。本評価は約737秒で完了し、通信・APIエラー0件、自動再試行0回でした。事前の診断1件と初回HTTP 429の試行は、本評価の精度・時間・トークン数から除外しています。

Jevのモデル版・SDK・実行地域・接続再利用の条件は未記録です。ネットワークとクライアントも処理時間に含まれるため、速度を理由に選ぶ際には同じ実行環境で測り直す必要があります。

### 拒否と「保留」の仕様上の違い

| 状態 | Decisions API | Jev |
| --- | --- | --- |
| APIによる拒否 | 質問単位のrefusal型を定義 | 公開回答型に同等のrefusal型はない |
| 信頼度が低い選択回答 | choiceを返し、採用・保留は利用側で判断 | 同様に利用側で判断 |
| その他・該当なし | 必要なら候補として明示追加 | 同様に候補として明示追加 |
| HTTP 429など | 選択回答とは別のAPIエラー | 同様に別のAPIエラー |

Decisionsのrefusal型は`type`と`name`のみで、未命名の質問では`name`が`null`です。一方、Jevの`criteria`内の`null`は候補の説明文を省略する指定です。「その他」の選択も通常の分類であり、どちらも拒否とは別です。[Decisions APIリファレンス](https://developers.openai.com/api/reference/resources/decisions/methods/create)、[Jev Choice](https://docs.typesafe.ai/primitives/choice)、[回答型定義](https://docs.typesafe.ai/sdk/python/api/types/responses)

JevのChoiceのconfidenceは、候補数`n`と最大確率`p_max`から`(p_max - 1/n) / (1 - 1/n)`で計算します。Decisionsと同じ計算式・閾値で使えるとは扱わず、保留の閾値は別の業務データで調整します。[Jev confidence](https://docs.typesafe.ai/confidence)、[Decisionsの回答の解釈](https://developers.openai.com/api/docs/guides/decisions#interpret-the-answers)

### この結果から言える範囲

- 候補は評価データの正解関係から作った閉じた36択です。「関係なし」や未知の関係の発見、候補集合の作成能力は評価していません。
- 正解はグラフ上の事実で、本文抜粋に根拠が残っているかは全件検証していません。事前知識で正解した可能性もあり、本文からの抽出精度とは区別します。
- 日英の本文は翻訳対ではありません。日本語は平均約6,658文字、英語は約2,114文字。日本語571件は10,000文字の上限に達していますが、元キャッシュがないため実際の切り詰め件数は断定できません。
- 主語は810種類で、上位3関係が約62.3%を占め、10関係は各2件以下です。少数関係のMacro F1、繰り返し実行の安定性、確率の較正の優劣は、この1回の比較から保証できません。

## 再現用ファイル

| ファイル | 内容 |
| --- | --- |
| [evaluate_relations.py](https://www.inoue-kobo.com/llm/decisions-jev-relation-comparison/sample/evaluate_relations.py) | 同じ本文・指示・候補からDecisionsを実行し、拒否を含めて記録・集計 |
| [run_experiment.py](https://www.inoue-kobo.com/llm/decisions-jev-relation-comparison/sample/run_experiment.py) | 日英の入力検証、マスク入力、合算予算確認、逐次実行 |
| [build_comparison.py](https://www.inoue-kobo.com/llm/decisions-jev-relation-comparison/sample/build_comparison.py) | 保存結果のオフライン集計と図の生成 |
| [aggregate-results.json](https://www.inoue-kobo.com/llm/decisions-jev-relation-comparison/sample/aggregate-results.json) | 記事に使った公開用の集計値と入力ハッシュ |
| [estimate_jev_cost.py](https://www.inoue-kobo.com/llm/decisions-jev-relation-comparison/sample/estimate_jev_cost.py) | 固定入力とローカルのtokenizer資産でJev参考費用を再計算。API呼び出しなし |
| [jev-cost-estimates.json](https://www.inoue-kobo.com/llm/decisions-jev-relation-comparison/sample/jev-cost-estimates.json) | 4シナリオのトークン数・参考費用・tokenizerのバージョンとハッシュ |
| [input-comparison-audit.json](https://www.inoue-kobo.com/llm/decisions-jev-relation-comparison/sample/input-comparison-audit.json) | 入力条件の一致件数・ハッシュ照合結果・確認できない条件 |
| [README.txt](https://www.inoue-kobo.com/llm/decisions-jev-relation-comparison/sample/README.txt) | 実行条件、検証・図の再生成手順、費用に関する注意 |

生のWikipedia本文、個別応答、リクエストID、APIキーは同梱していません。同じ評価には、公開したハッシュと一致する固定入力が必要です。本文を再取得すると条件が変わる可能性があります。Wikipedia本文にはCoDExのコードとは別の再配布条件があります。[Wikimediaの再利用案内](https://www.mediawiki.org/wiki/Wikimedia_APIs/Content_reuse)
