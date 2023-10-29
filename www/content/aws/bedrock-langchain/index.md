---
title: 'Amazon BedrockをLangChainから使う場合の簡単なサンプル'
date: '2023-10-29'
tags:
    - 'LangChain'
    - 'BedRock'
    - 'OpenAI'
thumbnail: 'aws/bedrock-langchain/images/thumbnail.png'
---

# Amazon BedrockをLangChainから使う場合の簡単なサンプル

Amazon BedrockをLangChainから使う場合の簡単なサンプルです。OpenAI API経由でGPT-3.5/4、Bedrock経由でAnthropic Claude2を呼び出します。現在のLangChain(0.0.323)限定かもしれませんが、OpenAIとBedrockを呼び分ける際に、返されるインスタンスの型が異なるという注意点があります。

## 前提条件

* langchain==0.0.323
* openai==0.27.8

## ソースコード

```python
from langchain.llms import Bedrock
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import AIMessage

gpt3_5 = ChatOpenAI(model="gpt-3.5-turbo-16k-0613", temperature=0)
gpt4 = ChatOpenAI(model="gpt-4-0613", temperature=0)
claude2 = Bedrock(
    credentials_profile_name="changeme",
    model_id="anthropic.claude-v2",
    model_kwargs={"temperature": 0, "max_tokens_to_sample": 4000},
)

prompt = ChatPromptTemplate.from_messages([("user", "{query}")])

for model_name, llm in zip(["gpt-3.5", "gpt-4", "claude2"], [gpt3_5, gpt4, claude2]):
    chain = prompt | llm
    res = chain.invoke({"query": "こんにちは"})

    # OpenAI APIの場合はAIMessageが返されますが、Bedrockの場合はstrが返されます。
    if type(res) == AIMessage:
        res_str = res.content
    else:
        res_str = res

    print(f"{model_name}: {res_str}")
```

```
gpt-3.5: こんにちは！何かお手伝いできますか？
gpt-4: こんにちは！何かお手伝いできることがありますか？
claude2:  はい、こんにちは。どうぞよろしくお願いします。
```

## 参考文献

* [LangChain - Bedrock](https://python.langchain.com/docs/integrations/llms/bedrock)
* [Amazon Bedrock](https://aws.amazon.com/jp/bedrock/)