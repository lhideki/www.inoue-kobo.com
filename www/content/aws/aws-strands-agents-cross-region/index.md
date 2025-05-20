---
title: 'AWS Strands Agentsをクロスリージョン推論で利用する'
date: '2025-05-20'
tags:
    - 'LLM'
    - 'AI Agent'
    - 'Amazon Bedrock'
thumbnail: 'aws/aws-strands-agents-cross-region/images/thumbnail.png'
---

# AWS Strands Agentsをクロスリージョン推論で利用する

先日、AWSからOSSとして[Strands Agents](https://github.com/strands-agents/sdk-python)がリリースされました。Strands Agentsは、AIエージェントを簡単に作成できるフレームワークです。以下のブログでMCP Serverを利用するためのデモが紹介されています。今回、Strands AgentsをAmazon Bedrockのクロスリージョン推論を利用して、東京リージョンで利用する方法を確認してみました。

* [Strands Agents – オープンソース AI エージェント SDK の紹介](https://aws.amazon.com/jp/blogs/news/introducing-strands-agents-an-open-source-ai-agents-sdk/)

## 前提条件

* Strands Agents: 0.1.2

## まずはデモとおり動かしてみる

事前準備として、必要なモジュールをインストールします。

```bash
pip install strands-agents strands-agents-tools
```

Agentのコードは以下のとおりです。ブログで紹介されているものと同じ内容です。

* agent.py

```python
from strands import Agent
from strands.tools.mcp import MCPClient
from strands_tools import http_request
from mcp import stdio_client, StdioServerParameters

# 命名にフォーカスしたシステムプロンプトの定義
NAMING_SYSTEM_PROMPT = """
あなたはオープンソースプロジェクトの命名を支援するアシスタントです。

オープンソースのプロジェクト名を提案する際は、必ずプロジェクトに使用できる
ドメイン名と GitHub の組織名を 1 つ以上提供してください。

提案する前に、ドメイン名がすでに登録されていないか、GitHub の組織名が
すでに使用されていないか、ツールを使って確認してください。
"""

# ドメイン名が利用可能かを判定する MCP サーバーを読み込む
domain_name_tools = MCPClient(lambda: stdio_client(
    StdioServerParameters(command="uvx", args=["fastdomaincheck-mcp-server"])
))

# GitHub の組織名が利用可能かを判定するためのリクエストを
# GitHub に送る Strands Agents の事前構築済みツール
github_tools = [http_request]

with domain_name_tools:
    # ツールとシステムプロンプトと共に命名エージェントを定義
    tools = domain_name_tools.list_tools_sync() + github_tools
    naming_agent = Agent(
        system_prompt=NAMING_SYSTEM_PROMPT,
        tools=tools
    )

    # エンドユーザーのプロンプトと共に命名エージェントを実行
    naming_agent("AI エージェント構築のためのオープンソースプロジェクトの名前を考えてください。")
```

あとは実行するだけですが、ブログの内容ではバージニア北部(us-east-1)で実行することを前提としているため、そのままでは動きません。このため、以下のように実行時のリージョンをus-east-1に指定する必要があります。

```bash
AWS_REGION=us-east-1 python -u agent.py
```

### uvxがないと動かない

MCP Serverを実行するためにuvxが必要です。このため、uvを事前にインストールしておく必要があります。uvのインストールは公式サイトを参考にしてください。

* [Installing uv](https://docs.astral.sh/uv/getting-started/installation/)

### デフォルトモデルは何を使用しているのか

ブログの内容では使用するLLMを指定していません。この場合Strands Agentsのデフォルトモデルが使用されます。Strands Agentsのデフォルトモデルは以下が使用されるようです(Version 0.1.2時点)。

* us.anthropic.claude-3-7-sonnet-20250219-v1:0

## クロスリージョン推論を利用する

今度は、Bedrockのクロスリージョン推論を利用して東京リージョンで動かしてみます。クロスリージョン推論を利用するためにはModelIDを指定する必用があります。このため、`BedrockModel`を明示的に指定するようにします。

* agent-ap-northeast-1.py

```python
from strands import Agent
from strands.tools.mcp import MCPClient
from strands_tools import http_request
from mcp import stdio_client, StdioServerParameters
from strands.models import BedrockModel

# 命名にフォーカスしたシステムプロンプトの定義
NAMING_SYSTEM_PROMPT = """
あなたはオープンソースプロジェクトの命名を支援するアシスタントです。

オープンソースのプロジェクト名を提案する際は、必ずプロジェクトに使用できる
ドメイン名と GitHub の組織名を 1 つ以上提供してください。

提案する前に、ドメイン名がすでに登録されていないか、GitHub の組織名が
すでに使用されていないか、ツールを使って確認してください。
"""

# ドメイン名が利用可能かを判定する MCP サーバーを読み込む
domain_name_tools = MCPClient(
    lambda: stdio_client(
        StdioServerParameters(command="uvx", args=["fastdomaincheck-mcp-server"])
    )
)

# GitHub の組織名が利用可能かを判定するためのリクエストを
# GitHub に送る Strands Agents の事前構築済みツール
github_tools = [http_request]

# クロスリージョン推論を利用するためのモデルを指定です。
# 使用するModelIDの先頭に、apac.を付ける必要があります。
bedrock_model = BedrockModel(
    model_id="apac.anthropic.claude-3-7-sonnet-20250219-v1:0", temperature=0.3
)

with domain_name_tools:
    # ツールとシステムプロンプトと共に命名エージェントを定義
    tools = domain_name_tools.list_tools_sync() + github_tools
    # model=bedrock_modelを指定することで、クロスリージョン推論を利用します。
    naming_agent = Agent(
        system_prompt=NAMING_SYSTEM_PROMPT, tools=tools, model=bedrock_model
    )

    # エンドユーザーのプロンプトと共に命名エージェントを実行
    naming_agent(
        "AI エージェント構築のためのオープンソースプロジェクトの名前を考えてください。"
    )
```

以下のように`AWS_REGION`を指定せずに実行することができます(実行環境のデフォルトリージョンがap-northeast-1である前提です)。

```bash
python -u agent-ap-northeast-1.py
```

### クロスリージョン推論を利用する場合はModel catalogのModel IDにプレフィクスを追加する

クロスリージョン推論を利用する場合は、Model catalogのModel IDに呼び出したいリージョンに応じたプレフィクス指定を追加する必要があります。東京リージョン(ap-northeast-1)で利用する場合の例は以下のとおりです。勿論、クロスリージョン推論に対応しているモデルである必要があります。

* 元のModel ID: anthropic.claude-3-7-sonnet-20250219-v1:0
* 東京リージョンのプレフィクス: apac.
* クロスリージョンを利用する場合のModel ID: apac.anthropic.claude-3-7-sonnet-20250219-v1:0

## 参考文献

* [Strands Agents – オープンソース AI エージェント SDK の紹介](https://aws.amazon.com/jp/blogs/news/introducing-strands-agents-an-open-source-ai-agents-sdk/)
* [Strands Agents](https://github.com/strands-agents/sdk-python)
* [[Amazon Bedrock] クロスリージョン推論が東京リージョンを含むアジアパシフィック地域で利用可能になりました](https://dev.classmethod.jp/articles/amazon-bedrock-cross-region-inference-apac/)