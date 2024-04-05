---
title: 'PandasのDataFrameに対して、Amazon Bedrockを利用した処理を並列で呼び出す方法'
date: '2024-04-05'
tags:
    - 'AWS'
    - 'Bedrock'
    - 'LLM'
thumbnail: 'llm/parallel-processing-amazon-bedrock/images/thumbnail.webp'
---

# PandasのDataFrameに対して、Amazon Bedrockを利用した処理を並列で呼び出す方法

PandasのDataFrameに対して、[Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html)を利用した処理を並列で呼び出す方法を調べてみました。[PandasのDataFrameに対して、OpenAI APIを利用した処理を並列で呼び出す方法](https://www.inoue-kobo.com/llm/parallel-processing-openai/)のAmazon Bedrockバージョンです。こちらも端的には[pandarallel](https://github.com/nalepae/pandarallel)を使いましょう、です。

```python
from pandarallel import pandarallel

pandarallel.initialize()

summary_df["summary"] = summary_df["url"].parallel_apply(get_summary)

display(summary_df)
```

## 前提条件

* boto3==1.34.51
* pandarallel==1.6.5

## 事前準備

```python
from __future__ import annotations
import boto3
import json
import urllib
import pandas as pd

urls = [
    "https://www.inoue-kobo.com/llm/openai-reduce-embedding-dim/",
    "https://www.inoue-kobo.com/aws/selenium-serverless/",
    "https://www.inoue-kobo.com/aws/aws-service-summary/",
    "https://www.inoue-kobo.com/ai_ml/duckduckgo-langchain-langsmith/",
    "https://www.inoue-kobo.com/ai_ml/llamaindex-pdf-gradio/",
]

summary_df = pd.DataFrame(urls, columns=["url"])
pd.set_option("display.max_colwidth", None)
```

## 単純に apply するだけ

```python
def get_summary(url: str) -> str | None:
    res_web = urllib.request.urlopen(url)  # type: ignore
    content = res_web.read().decode("utf-8")

    res_bedrock = bedrock.invoke_model(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        accept="application/json",
        contentType="application/json",
        body=json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4096,
                "system": "以下はWebサイトの内容です。HTMLタグを削除した上で、150文字以内で要約してください。",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": content,
                            }
                        ],
                    },
                ],
            }
        ),
    )
    response_body = json.loads(res_bedrock.get("body").read())

    return response_body.get("content")[0].get("text")
```

```python
summary_df["summary"] = summary_df["url"].apply(get_summary)

display(summary_df)
```

実行時間は17.0sでした。applyしただけでは並列処理は行われないため、この処理時間が基準になります。

## asyncを使う

```python
import asyncio


async def aget_summary(url: str) -> str | None:
    res_web = urllib.request.urlopen(url)  # type: ignore
    content = res_web.read().decode("utf-8")

    res_bedrock = bedrock.invoke_model(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        accept="application/json",
        contentType="application/json",
        body=json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4096,
                "system": "以下はWebサイトの内容です。HTMLタグを削除した上で、150文字以内で要約してください。",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": content,
                            }
                        ],
                    },
                ],
            }
        ),
    )
    response_body = json.loads(res_bedrock.get("body").read())

    return response_body.get("content")[0].get("text")
```

```python
summary_df["summary"] = await asyncio.gather(*[aget_summary(url) for url in urls])

display(summary_df)
```

実行時間は15.8sでした。asyncioでは期待したような並列処理(正確にはノンブロッキングIO)が行われていないことが確認できます。

## ThreadPoolExecutorを使う

```python
from concurrent.futures import ThreadPoolExecutor
```

```python
with ThreadPoolExecutor() as executor:
    summary_df["summary"] = list(executor.map(get_summary, urls))

display(summary_df)
```

実行時間は5.8sでした。applyを適用しただけの順次処理よりも大幅に処理時間が短くなっています。並列処理が行われていることが確認できます。

## pandarallelを使う

```python
from pandarallel import pandarallel

pandarallel.initialize()
```

```python
summary_df["summary"] = summary_df["url"].parallel_apply(get_summary)

display(summary_df)
```

実行時間は5.3sでした。こちらも並列処理が行われていることが確認できます。

## 参考文献

* [pandarallel](https://github.com/nalepae/pandarallel)
* [Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html)