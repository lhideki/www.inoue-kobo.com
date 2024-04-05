---
title: 'PandasのDataFrameに対して、OpenAI APIを利用した処理を並列で呼び出す方法'
date: '2024-04-05'
tags:
    - 'OpenAI'
thumbnail: 'llm/parallel-processing-openai/images/thumbnail.webp'
---

# PandasのDataFrameに対して、OpenAI APIを利用した処理を並列で呼び出す方法

PandasのDataFrameに対して、OpenAI APIを利用した処理を並列で呼び出す方法を調べてみました。結論としては、`pandarallel`を使うのが最も簡単だろうという感じです。

```python
from pandarallel import pandarallel

pandarallel.initialize()

summary_df["summary"] = summary_df["url"].parallel_apply(get_summary)

display(summary_df)
```

## 前提条件

* openai==1.16.2
* pandarallel==1.6.5

## 事前準備

```python
from __future__ import annotations
import openai
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

    res_openai = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "以下はWebサイトの内容です。HTMLタグを削除した上で、150文字以内で要約してください。",
            },
            {"role": "user", "content": content},
        ],
    )

    return res_openai.choices[0].message.content
```

```python
summary_df["summary"] = summary_df["url"].apply(get_summary)

display(summary_df)
```

実行時間は12.3sでした。applyしただけでは並列処理は行われないため、この処理時間が基準になります。

## asyncを使う

```python
import asyncio


async def aget_summary(url: str) -> str | None:
    res_web = urllib.request.urlopen(url)  # type: ignore
    content = res_web.read().decode("utf-8")

    res_openai = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "以下はWebサイトの内容です。HTMLタグを削除した上で、150文字以内で要約してください。",
            },
            {"role": "user", "content": content},
        ],
    )

    return res_openai.choices[0].message.content
```

```python
summary_df["summary"] = await asyncio.gather(*[aget_summary(url) for url in urls])

display(summary_df)
```

実行時間は12.8sでした。asyncioでは期待したような並列処理(正確にはノンブロッキングIO)が行われていないことが確認できます。

## ThreadPoolExecutorを使う

```python
from concurrent.futures import ThreadPoolExecutor
```

```python
with ThreadPoolExecutor() as executor:
    summary_df["summary"] = list(executor.map(get_summary, urls))

display(summary_df)
```

実行時間は4.3sでした。applyを適用しただけの順次処理よりも大幅に処理時間が短くなっています。並列処理が行われていることが確認できます。

## pandarallelを使う

```python
from pandarallel import pandarallel

pandarallel.initialize()
```

```python
summary_df["summary"] = summary_df["url"].parallel_apply(get_summary)

display(summary_df)
```

実行時間は3.0sでした。こちらも並列処理が行われていることが確認できます。なお、`ThreadPoolExecutor`よりも処理時間が早くなっていますが、これはOpenAI APIのレスポンス時間の揺れによるものです。

## おまけ

[async-openai](https://github.com/GrowthEngineAI/async-openai)というものがあるらしいですが(非公式ライブラリです)、今回は未検証です。