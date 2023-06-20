---
title: 'LangChain 0.0.200での履歴を持ったChat、シンプルなAgent、長文の要約の実装サンプル'
date: '2023-06-20'
tags:
    - 'LangChain'
    - 'OpenAI'
thumbnail: 'ai_ml/langchain-sample/images/thumbnail.png'
---

# LangChain 0.0.200での履歴を持ったChat、シンプルなAgent、長文の要約の実装サンプル

## TL;DR

LangChain は進化が早いということもあり、頻繁にインターフェイスが変更になります。この記事は`0.0.200`の時点での以下のタスクに対する実装サンプルです。

-   履歴を持った Chat
-   シンプルな Agent
-   長文の要約

### 前提条件

-   langchain==0.0.200
-   openai==0.27.0

## 履歴を持った Chat

```python
!pip install langchain
!pip install openai
!pip install tiktoken
```

```python
import os

openai_api_key = "OPENAI_API_KEY"#@param {type:"string"}

os.environ["OPENAI_API_KEY"] = openai_api_key
```

```python
from langchain.prompts import PromptTemplate, MessagesPlaceholder
from langchain import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0613")
system_template = """以下は、HumanとAIが仲良く会話している様子です。AIは饒舌で、その文脈から具体的な内容をたくさん教えてくれます。AIは質問に対する答えを知らない場合、正直に「知らない」と答えます。"""
system_prompt = SystemMessagePromptTemplate.from_template(
    system_template
)
memory = ConversationSummaryBufferMemory(
    llm=llm, max_token_limit=100, return_messages=True
)
prompts = ChatPromptTemplate.from_messages([
    system_prompt,
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])
chain = ConversationChain(
    llm=llm, prompt=prompts, memory=memory, verbose=True
)
```

```python
mesg = chain.run("こんにちは")
print(mesg)
```

```
> Entering new  chain...
Prompt after formatting:
System: 以下は、HumanとAIが仲良く会話している様子です。AIは饒舌で、その文脈から具体的な内容をたくさん教えてくれます。AIは質問に対する答えを知らない場合、正直に「知らない」と答えます。
Human: こんにちは

> Finished chain.
こんにちは！どのようにお手伝いできますか？
```

```python
mesg = chain.run("お薦めの映画を教えてください。")
print(mesg)
```

```
> Entering new  chain...
Prompt after formatting:
System: 以下は、HumanとAIが仲良く会話している様子です。AIは饒舌で、その文脈から具体的な内容をたくさん教えてくれます。AIは質問に対する答えを知らない場合、正直に「知らない」と答えます。
Human: こんにちは
AI: こんにちは！どのようにお手伝いできますか？
Human: お薦めの映画を教えてください。

> Finished chain.
もちろんです！映画のおすすめは人それぞれですが、いくつかのジャンルからいくつかの映画をご紹介します。まず、アクション映画では「マトリックス」や「ダークナイト」がおすすめです。これらの映画はスリリングなアクションシーンと深いストーリーが特徴です。次に、ドラマ映画では「ショーシャンクの空に」や「グリーンマイル」が感動的なストーリーで人気です。また、コメディ映画では「ハングオーバー」や「スーパーバッド」が笑いを提供してくれます。さらに、ファンタジー映画では「ハリー・ポッターシリーズ」や「ロード・オブ・ザ・リングシリーズ」が魔法や冒険の世界を描いています。これらは一部のおすすめ映画ですが、もちろん他にもたくさんの素晴らしい映画がありますので、お好みのジャンルやテーマに合わせて探してみてください！
```

```python
mesg = chain.run("Markdown形式の箇条書きにしてください。")
print(mesg)
```

```
> Entering new  chain...
Prompt after formatting:
System: 以下は、HumanとAIが仲良く会話している様子です。AIは饒舌で、その文脈から具体的な内容をたくさん教えてくれます。AIは質問に対する答えを知らない場合、正直に「知らない」と答えます。
System: The human greets the AI in Japanese and asks for movie recommendations. The AI responds by suggesting movies from various genres, such as action, drama, comedy, and fantasy. It recommends movies like "The Matrix" and "The Dark Knight" for action, "The Shawshank Redemption" and "The Green Mile" for drama, "The Hangover" and "Superbad" for comedy, and the "Harry Potter" and "Lord of the Rings" series for fantasy. The AI encourages the human to explore different genres and themes to find movies they enjoy.
Human: Markdown形式の箇条書きにしてください。

> Finished chain.
- アクション映画のおすすめ: 「マトリックス」や「ダークナイト」など
- ドラマ映画のおすすめ: 「ショーシャンクの空に」や「グリーンマイル」など
- コメディ映画のおすすめ: 「ハングオーバー」や「スーパーバッド」など
- ファンタジー映画のおすすめ: 「ハリーポッター」シリーズや「ロード・オブ・ザ・リング」シリーズなど
- 異なるジャンルやテーマを探求して、自分が楽しめる映画を見つけることをおすすめします。
```

## シンプルな Agent

```python
!pip install langchain
!pip install openai
!pip install tiktoken
```

```python
import os

openai_api_key = "OPENAI_API_KEY"#@param {type:"string"}
google_api_key = "GOOGLE_API_KEY"#@param {type:"string"}
google_cse_id = "GOOGLE_CSE_ID"#@param {type:"string"}

os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["GOOGLE_API_KEY"] = google_api_key
os.environ["GOOGLE_CSE_ID"] = google_cse_id
```

```python
from langchain import (
    LLMMathChain,
    OpenAI,
    SerpAPIWrapper,
    SQLDatabase,
    SQLDatabaseChain,
)
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.utilities.google_search import GoogleSearchAPIWrapper
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0613")
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
google_search = GoogleSearchAPIWrapper()
tools = [
    Tool(
        name = "GoogleSearch",
        func=google_search.run,
        description="最新の話題について答える場合に利用することができます。また、今日の日付や今日の気温、天気、為替レートなど現在の状況についても確認することができます。入力は検索内容です。"
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="計算をする場合に利用することができます。"
    )
]

agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)
agent.run("最新のドル円為替レートを2で割った計算結果を教えてください。")
```

```
> Entering new  chain...

Invoking: `GoogleSearch` with `最新のドル円為替レート`

19日午後の東京外国為替市場で、円相場が下げ幅を縮めている。14時時点では1ドル=141円48～50銭と前週末17時時点と比べて34銭の円安・ドル高だった。日経平均株価は ... 通貨（通貨単位）, 為替レート（円）. 外貨→円貨（TTB）, 円貨→外貨（TTS）. 米ドル（1 USD）, 141.18, 142.18. ユーロ（1 EUR）, 154.17, 155.57. また、ロシアルーブル/円のみ現在新規注文受付停止になっているのでご留意ください。 ※GMOクリック証券のトルコリラ円、ユーロポンド、カナダドル円、スイスフラン円の ... 7 days ago ... 米ドル/円のチャートをリアルタイムで紹介。今後の為替レートの動きや相場、投資家に人気のおすすめの仮想通貨取引所などについて解説しています。 また、最近では「有事のドル買い」といった格言も定着しつつあり、地政学リスクが高まった場合の緊急避難先として選考されるケースも見られます。 主な経済指標: FOMC（ ... Jun 8, 2023 ... 通貨名, TTS （日本円→外貨）, TTB （外貨→日本円）. 001, USD（米ドル）, 139.25, 138.75. 020, EUR（ユーロ）, 149.95, 149.45. お取り引きの際は、必ずログイン後のお取り引き画面にて最新の為替レートをご確認ください。 【円からはじめる限定金利提供中！】＜米ドル・ユーロ・豪ドル・NZ ... 米ドル・豪ドル・ユーロなど全9通貨の外国為替相場チャート表です。最新の為替レートや過去からの推移をご確認いただけます。実際の取引時に適用される為替レートは、 ... OANDA Rates™に基づいた、正確で信頼性の高い為替レートを提供しております。 ... OANDAの為替コンバーターを使用すると、最新の外国為替平均ビッド/アスクレートを ... Jun 2, 2023 ... USドル/円の為替レートの推移をグラフ及び時系列表にて掲載しています。 1USドル → 137.1942円 (参考: 1円 → 0.0073USドル) ※2023年5月の平均レート ...
Invoking: `Calculator` with `141.48 / 2`

> Entering new  chain...
141.48 / 2\`\`\`text
141.48 / 2
\`\`\`

...numexpr.evaluate("141.48 / 2")...

Answer: 70.74

> Finished chain.
> Answer: 70.74 最新のドル円為替レートを 2 で割った計算結果は、70.74 です。

> Finished chain.
> 最新のドル円為替レートを 2 で割った計算結果は、70.74 です。

```

## 長文の要約

```python
!pip install langchain
!pip install openai
!pip install tiktoken
!pip install unstructured
```

```python
import os

openai_api_key = "OPENAI_API_KEY"#@param {type:"string"}
os.environ["OPENAI_API_KEY"] = openai_api_key
```

```python
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

url = "changeme"
loader = UnstructuredURLLoader(urls=[url])
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=2000, chunk_overlap=0, model_name="gpt-3.5-turbo-0613"
)
docs = text_splitter.split_documents(data)
```

```python
len(docs)
```

```python
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0613")
summary_prompt_template = """Write a concise summary of the following:

{text}

CONCISE SUMMARY IN Japanse:"""

summary_prompt = PromptTemplate(
    template=summary_prompt_template, input_variables=["text"]
)
chain = load_summarize_chain(
    llm=llm,
    chain_type="map_reduce",
    map_prompt=summary_prompt,
    combine_prompt=summary_prompt,
    verbose=False,
)
summary = await chain.arun(docs)
display(summary)
```

```python
from IPython.display import display, Markdown

display(Markdown(summary))
```
