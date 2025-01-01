---
title: 'AWS LambdaでBrowser Useを動かす'
date: '2025-01-01'
tags:
    - 'AWS'
    - 'browser-use'
    - 'AI Agent'
    - 'Lambda'
thumbnail: 'aws/browser-use-on-aws-lambda/images/thumbnail.png'
---

# AWS Lambda で Browser Use を動かす

AWS Lambda で[Browser Use(browser-use)](https://github.com/browser-use/browser-use)を動かすためのテンプレートです。利用しているパッケージの容量的に Docker Lambda を利用しています。

## browser-use について

[browser-use](https://github.com/browser-use/browser-use)は LLM を利用した AI エージェントとして、自然言語で処理したタスクを実行するために、ブラウザを自動的に操作してくれるフレームワークです。以下が実行イメージです。

```python
agent = Agent(
    task="Yahoo Japanの今日のニュースを5つ箇条書きにしてください。",
    llm=ChatOpenAI(model="gpt-4o"),
)
result = await agent.run()
```

![](images/browser-use-movie.gif)

以下が出力結果です。

| No. | Title                                     | Summary                                                                                                                                                                                                                   |
| --- | ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Outsourcing Business System Development   | The article discusses the benefits and strategies of outsourcing business system development, focusing on efficiency and cost-effectiveness. Key points include choosing the right partner and maintaining communication. |
| 2   | Notion Database for Reading List          | This article explains how to use Notion to manage a reading list. It covers creating a database, tagging, and organizing entries, and using Notion Web Clipper for easy entry addition.                                   |
| 3   | Exporting Notion Reading List to Markdown | The article provides a script for exporting Notion reading lists to Markdown format. It details the setup of a Notion database and the process of exporting tagged and categorized entries into Markdown files.           |

## なぜ Lambda で動かすのか?

処理を自動化したい場合に、サーバーレスである AWS Lambda を利用することで、処理実行中のみ課金されるため、コストを抑えることができます。また、Lambda はスケーラブルであるため、大量のリクエストにも対応できます。

## browser-use を Lambda で動かすためのポイント

Dockerfile と browser-use 実行時の Browser Config がポイントです。

### Dockerfile

[How to run Playwright with Python in AWS Lambda](https://www.cloudtechsimplified.com/playwright-aws-lambda-python/)を参考にさせていただきました。

```Dockerfile
# ref: https://github.com/browser-use/browser-use
# Define function directory
ARG FUNCTION_DIR="/function"

FROM --platform=linux/x86_64 mcr.microsoft.com/playwright/python:v1.49.1-noble as build-image

# Install aws-lambda-cpp build dependencies
RUN apt-get update && \
    apt-get install -y \
    g++ \
    make \
    cmake \
    unzip \
    libcurl4-openssl-dev

# Include global arg in this stage of the build
ARG FUNCTION_DIR
# Create function directory
RUN mkdir -p ${FUNCTION_DIR}

# Copy function code
COPY . ${FUNCTION_DIR}


RUN pip3 install  \
    --target ${FUNCTION_DIR} \
    awslambdaric
RUN pip install --no-cache-dir -r ${FUNCTION_DIR}/requirements.txt --target ${FUNCTION_DIR}

# Multi-stage build: grab a fresh copy of the base image
FROM --platform=linux/x86_64 mcr.microsoft.com/playwright/python:v1.49.1-noble

# Include global arg in this stage of the build
ARG FUNCTION_DIR
# Set working directory to function root directory
WORKDIR ${FUNCTION_DIR}

# Copy in the build image dependencies
COPY --from=build-image ${FUNCTION_DIR} ${FUNCTION_DIR}

ENTRYPOINT [ "/usr/bin/python", "-m", "awslambdaric" ]
CMD [ "browser_use_on_aws_lambda.functions.main_function.lambda_handler" ]
```

### Browser Config

`browser-use` では、ブラウザの操作に `Playwright` を使用しています。[How to run Playwright with Python in AWS Lambda](https://www.cloudtechsimplified.com/playwright-aws-lambda-python/)に記載があるように、Lambda で Playwright を動かすためには、Chrome の引数として以下の設定が必要です。

-   --disable-gpu
-   --single-process

Lambda は実行できるけど、Playwright(browser-use)が Page の Load で失敗する場合は、上記の設定を確認してください。browser-use では、以下のように設定します。

```python
browser = Browser(
    config=BrowserConfig(
        headless=True, extra_chromium_args=["--disable-gpu", "--single-process"]
    )
)
agent = Agent(
        ...
        browser=browser,
    )
```

詳細は[GitHub - browser-use-on-aws-lambda](https://github.com/lhideki/browser-use-on-aws-lambda)を参照してください。

## 注意事項

-   AI エージェントで操作しているブラウザからアクセスして良いかは、各サイトの利用規約を確認する必要があります。現状は明確に禁止しているところは少ないと思いますが、確認が取れない場合は少なくとも商用利用は避けた方が良いかもしれません。

## 参考文献

-   [How to run Playwright with Python in AWS Lambda](https://www.cloudtechsimplified.com/playwright-aws-lambda-python/)
-   [GitHub - browser-use](https://github.com/browser-use/browser-use)
-   [GitHub - browser-use-on-aws-lambda](https://github.com/lhideki/browser-use-on-aws-lambda)
