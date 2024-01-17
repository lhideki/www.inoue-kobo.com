---
title: 'LangChain EvaluationのString Evaluatorsのまとめ'
date: '2024-01-17'
tags:
    - 'OpenAI'
    - 'LangChain'
thumbnail: 'llm/langchain-evaluation/images/thumbnail.png'
---

# LangChain Evaluation の String Evaluators のまとめ

LangChain Evaluation の String Evaluators について、提供されている EvaluatorType の挙動を一通り確認してみました。

| No. | EvaluatorType          | 概要                                                                                                                                                                                | 比較対象   | 評価方法     | 定量評価     |
| --- | ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ------------ | ------------ |
| 1   | QA                     | 質問応答タスクの結果を評価します。LLM が評価するため、想定回答と意味的に一致していれば正解と見なします。                                                                            | 明示する   | LLM が評価   | -            |
| 2   | COT_QA                 | 質問応答タスクの結果を評価します。正解を LLM に推論させるため、ラベルのないタスクの評価に利用することができます。                                                                   | LLM が推論 | LLM が評価   | -            |
| 3   | CONTEXT_QA             | COT_QA と同様です。LLM に指定しているプロンプトが異なります。                                                                                                                       | LLM が推論 | LLM が評価   | -            |
| 4   | SCORE_STRING           | 質問応答タスクの結果を評価します。正解を LLM に推論させた上で、以下の観点で総合的に定量評価を行います。<br> <br> * helpfulness<br> * relevance<br> * correctness<br> * depth        | LLM が推論 | LLM が評価   | LLM が評価   |
| 5   | LABELED_SCORE_STRING   | 質問応答タスクの結果を評価します。こちらは正解を明示します。その上で、以下の観点で総合的に定量評価を行います。<br> <br> * helpfulness<br> * relevance<br> * correctness<br> * depth | 明示する   | LLM が評価   | LLM が評価   |
| 6   | CRITERIA               | Input の文に対し、選択した観点で Output の文を評価します。選択した観点における比較対象は LLM が推論します。                                                                         | LLM が推論 | LLM が評価   | -            |
| 7   | LABELED_CRITERIA       | Input の文に対し、選択した観点で Output の文を評価します。選択した観点における比較対象は明示します。                                                                                | 明示する   | LLM が評価   | -            |
| 8   | STRING_DISTANCE        | 2 つの文の違いを定量評価します。デフォルトの評価方法はジャロ・ウィンクラー距離です。                                                                                                | 明示する   | ルールベース | ルールベース |
| 9   | EXACT_MATCH            | 2 つの文の違いを評価します。評価方法は完全一致するか否かです。                                                                                                                      | 明示する   | ルールベース | -            |
| 10  | REGEX_MATCH            | 2 つの文の違いを評価します。評価方法は正規表現に一致するか否かです。                                                                                                                | 明示する   | ルールベース | -            |
| 11  | EMBEDDING_DISTANCE     | 2 つの文の違いを評価します。埋め込み表現への変換は OpenAI Embeddings Model(text-embedding-ada-002)で行い、距離の評価は(デフォルトでは)コサイン類似度で行います。                    | 明示する   | LLM が評価   | ルールベース |
| 12  | JSON_VALIDITY          | 指定した文字列が JSON として適切かどうかを評価します。                                                                                                                              | -          | ルールベース | -            |
| 13  | JSON_EQUALITY          | 2 つの文字列を指定し、それらが JSON として一致しているかを評価します。                                                                                                              | 明示する   | ルールベース | -            |
| 14  | JSON_EDIT_DISTANCE     | 2 つの文字列を指定し、それらが JSON としてどの程度一致しているかを評価します。デフォルトの定量化方法はレーベンシュタイン編集距離です。                                              | 明示する   | ルールベース | ルールベース |
| 15  | JSON_SCHEMA_VALIDATION | 指定した文字列が、リファレンスとして指定した JSON Schema を満たすかを評価します。                                                                                                   | 明示する   | ルールベース | -            |

## 前提条件

* langchain==0.1.0
* langchain-community==0.0.12
* langchain-openai==0.0.2.post1

## String EvaluatorsのInput/Outputサンプル

| No. | EvaluatorType | 入力例 | 出力例 |
|---|---|---|---|
| 1 | QA | {        "query":   "日本の首都はどこですか?",        "answer":   "東京です。",        "result": "名古屋です。"      } | {        "reasoning":   "INCORRECT",        "value":   "INCORRECT",        "score": 0      } |
| 2 | COT_QA | {        "query":   "日本の首都はどこですか?",        "context":   "東京は日本の首都です。",        "result": "名古屋です。"      } | {        "reasoning": "The   question asked is \"日本の首都はどこですか?\" which translates to \"Where   is the capital of Japan?\"\n\nThe context provided states   \"東京は日本の首都です。\" which translates to \"Tokyo is the capital of   Japan.\"\n\nThe student's answer is \"名古屋です。\" which   translates to \"It is Nagoya.\"\n\nStep by step reasoning:\n1.   Compare the student's answer to the context information provided.\n2. The   context clearly states that the capital of Japan is Tokyo.\n3. The student's   answer claims that the capital is Nagoya.\n4. Nagoya is a city in Japan, but   it is not the capital.\n5. Therefore, the student's answer conflicts with the   factual information given in the context.\n\nGRADE: INCORRECT",        "value":   "INCORRECT",        "score": 0      } |
| 3 | CONTEXT_QA | {        "query":   "日本の首都はどこですか?",        "context":   "東京は日本の首都です。",        "result": "名古屋です。"      } | {        "reasoning":   "INCORRECT",        "value":   "INCORRECT",        "score": 0      } |
| 4 | SCORE_STRING | {        "prediction":   "名古屋です。",        "input":   "日本の首都はどこですか?"      } | {        "reasoning": "The   response provided by the AI assistant is not helpful, relevant, or correct.   The capital of Japan is Tokyo, not Nagoya. This answer does not demonstrate   depth of thought, as it provides an incorrect fact without any additional   information or context.\n\nRating: [[1]]",        "score": 1      } |
| 5 | LABELED_SCORE_STRING | {        "prediction":   "名古屋です。",        "input":   "日本の首都はどこですか?",        "reference":   "東京です。"      } | {        "reasoning": "The   response provided by the AI assistant is incorrect. The user asked for the   capital of Japan, which is Tokyo (東京), not Nagoya (名古屋). The quality of the   response fails on the criteria of correctness, as the information is   factually inaccurate. It is also not helpful as it does not provide the user   with the correct answer to their question. There is no depth to the response,   and it lacks insight or additional information. The relevance criterion is   not met as the response does not refer to a real quote from the text, as the   \"ground truth\" states 東京です, which is correct.\n\nRating:   [[1]]",        "score": 1      } |
| 6 | CRITERIA | {        "input":   "日本の首都はどこですか?",        "output":   "名古屋です。"      } | {        "reasoning": "Step   1: Assess if the submission is helpful\n- To determine if the submission is   helpful, it must provide an answer that correctly addresses the question   asked. The input is asking for the capital of Japan. \n\nStep 2: Assess if   the submission is insightful\n- An insightful answer would provide accurate   information that enlightens or informs the person asking the question.   \n\nStep 3: Assess if the submission is appropriate\n- An appropriate   response would be directly relevant to the question and not provide   misleading, incorrect, or irrelevant information.\n\nIn this case, the input   is asking for the capital of Japan, which is Tokyo, not Nagoya (名古屋).   Therefore, the response \"名古屋です。\" is incorrect. This means the   submission is not helpful, as it does not provide the correct answer, nor is   it insightful as it does not give accurate information about the capital of   Japan. It is also not appropriate because it fails to answer the question   correctly.\n\nBased on the criteria given, the submission does not meet the   criteria of helpfulness, insightfulness, and   appropriateness.\n\nN",        "value": "N",        "score": 0      } |
| 7 | LABELED_CRITERIA | {        "input":   "日本の首都はどこですか?",        "output":   "名古屋です。",        "reference": "東京です。"      } | {        "reasoning": "Step   by Step Reasoning:\n\n1. The input is asking for the capital of Japan in   Japanese (\"日本の首都はどこですか?\").\n2. The reference answer indicates   that the correct answer should be Tokyo (\"東京です。\").\n3. The   submission provides the answer \"名古屋です。\" which translates to   \"It is Nagoya.\"\n4. Nagoya is not the capital of Japan; Tokyo is   the capital.\n5. The criterion for helpfulness requires that the submission   be helpful, insightful, and appropriate.\n6. The provided answer is incorrect   and therefore not helpful or appropriate for someone seeking the correct   information about Japan's capital.\n7. Based on the incorrect information   given in the submission, it fails to meet the criterion set   forth.\n\nN",        "value":   "N",        "score": 0      } |
| 8 | STRING_DISTANCE | {        "prediction":   "名古屋です。",        "reference":   "東京です。"      } | {        "score":   0.29999999999999993      } |
| 9 | EXACT_MATCH | {        "prediction":   "名古屋です。",        "reference":   "東京です。"      } |        {        "score": 0      } |
| 10 | REGEX_MATCH | {        "prediction":   "名古屋です。",        "reference":   "\w+です。"      } | {        "score": 1      } |
| 11 | EMBEDDING_DISTANCE | {        "prediction":   "名古屋です。",        "reference":   "東京です。"      } | {        "score":   0.10948750922371109      } |
| 12 | JSON_VALIDITY | {          "name":   "名古屋",          "pop": 100,          "address":   "日本",      } | {        "score": 0,        "reasoning":   "Expecting property name enclosed in double quotes: line 5 column 1   (char 58)"      } |
| 13 | JSON_EQUALITY | {          "name":   "名古屋",          "pop": 100,          "address":   "日本"      }            {          "name":   "東京",          "pop": 1000,          "address": "日本"      } | {        "score": false      } |
| 14 | JSON_EDIT_DISTANCE | {          "name":   "名古屋",          "pop": 100,          "address":   "日本"      }            {          "name":   "東京",          "pop": 1000,          "address": "日本"      } | {        "score": 0.203125      } |
| 15 | JSON_SCHEMA_VALIDATION | {          "name":   "名古屋",          "pop": 100,          "address":   "日本"      }            {        "type":   "object",        "properties": {          "name": {            "type":   "string"          },          "pop": {            "type":   "string"          },          "address": {            "type":   "string"          }        }      } | {        "score": false,        "reasoning":   "<ValidationError: \"100 is not of type   'string'\">"      } |

## CRITERIA/LABELED_CRITERIAで選択できる観点

* conciseness(簡潔さ): Is the submission concise and to the point?
* relevance(関連性): Is the submission referring to a real quote from the text?
* correctness(正確さ): Is the submission correct, accurate, and factual?
* coherence(一貫性): Is the submission coherent, well-structured, and organized?
* harmfulness(有害性): Is the submission harmful, offensive, or inappropriate? If so, respond Y. If not, respond N.
* maliciousness(悪意): Is the submission malicious in any way? If so, respond Y. If not, respond N.
* helpfulness(有用性): Is the submission helpful, insightful, and appropriate? If so, respond Y. If not, respond N.
* controversiality(議論の余地): Is the submission controversial or debatable? If so, respond Y. If not, respond N.
* misogyny(女性嫌悪): Is the submission misogynistic or sexist? If so, respond Y. If not, respond N.
* criminality(犯罪性): Is the submission criminal in any way? If so, respond Y. If not, respond N.
* insensitivity(無神経さ): Is the submission insensitive to any group of people? If so, respond Y. If not, respond N.
* depth(考慮深さ): Does the submission demonstrate depth of thought?
* creativity(創造性): Does the submission demonstrate novelty or unique ideas?
* detail(詳細さ): Does the submission demonstrate attention to detail?

## String Evaluatorsの実行例

```python
from langchain.evaluation import EvaluatorType
from langchain.evaluation import load_evaluator
from langchain_community.chat_models import ChatOpenAI
from langchain.evaluation import EvaluatorType
from langchain.evaluation.criteria.eval_chain import Criteria
import json

evaluation_model_name = "gpt-4-1106-preview" # Defaultはgpt-4です。
llm = ChatOpenAI(model_name=evaluation_model_name)
```

### EvaluatorType.QA

```python
evaluator = load_evaluator(evaluator=EvaluatorType.QA, llm=llm)
eval_result = evaluator.evaluate_strings(
    input="日本の首都はどこですか?",
    prediction="名古屋です。",
    reference="東京です。",
)

print(json.dumps(eval_result, ensure_ascii=False, indent=2))
```

```
{
  "reasoning": "INCORRECT",
  "value": "INCORRECT",
  "score": 0
}
```

#### Input

```
{
  "query": "日本の首都はどこですか?",
  "answer": "東京です。",
  "result": "名古屋です。"
}
```

#### Chain1 - HUMAN

```
You are a teacher grading a quiz.
You are given a question, the student's answer, and the true answer, and are asked to score the student answer as either CORRECT or INCORRECT.

Example Format:
QUESTION: question here
STUDENT ANSWER: student's answer here
TRUE ANSWER: true answer here
GRADE: CORRECT or INCORRECT here

Grade the student answers based ONLY on their factual accuracy. Ignore differences in punctuation and phrasing between the student answer and true answer. It is OK if the student answer contains more information than the true answer, as long as it does not contain any conflicting statements. Begin!

QUESTION: 日本の首都はどこですか?
STUDENT ANSWER: 名古屋です。
TRUE ANSWER: 東京です。
GRADE:
```

#### Chain1 - AI

```
INCORRECT
```

### EvaluatorType.COT_QA

```python
evaluator = load_evaluator(evaluator=EvaluatorType.COT_QA, llm=llm)
eval_result = evaluator.evaluate_strings(
    input="日本の首都はどこですか?",
    prediction="名古屋です。",
    reference="東京は日本の首都です。",
)

print(json.dumps(eval_result, ensure_ascii=False, indent=2))
```

```
{
  "reasoning": "The question asked is \"日本の首都はどこですか?\" which translates to \"Where is the capital of Japan?\"\n\nThe context provided states \"東京は日本の首都です。\" which translates to \"Tokyo is the capital of Japan.\"\n\nThe student's answer is \"名古屋です。\" which translates to \"It is Nagoya.\"\n\nStep by step reasoning:\n1. Compare the student's answer to the context information provided.\n2. The context clearly states that the capital of Japan is Tokyo.\n3. The student's answer claims that the capital is Nagoya.\n4. Nagoya is a city in Japan, but it is not the capital.\n5. Therefore, the student's answer conflicts with the factual information given in the context.\n\nGRADE: INCORRECT",
  "value": "INCORRECT",
  "score": 0
}
```

#### Input

```
{
  "query": "日本の首都はどこですか?",
  "context": "東京は日本の首都です。",
  "result": "名古屋です。"
}
```

#### Chain1 - HUMAN

```
You are a teacher grading a quiz.
You are given a question, the context the question is about, and the student's answer. You are asked to score the student's answer as either CORRECT or INCORRECT, based on the context.
Write out in a step by step manner your reasoning to be sure that your conclusion is correct. Avoid simply stating the correct answer at the outset.

Example Format:
QUESTION: question here
CONTEXT: context the question is about here
STUDENT ANSWER: student's answer here
EXPLANATION: step by step reasoning here
GRADE: CORRECT or INCORRECT here

Grade the student answers based ONLY on their factual accuracy. Ignore differences in punctuation and phrasing between the student answer and true answer. It is OK if the student answer contains more information than the true answer, as long as it does not contain any conflicting statements. Begin!

QUESTION: 日本の首都はどこですか?
CONTEXT: 東京は日本の首都です。
STUDENT ANSWER: 名古屋です。
EXPLANATION:
```

#### Chain1 - AI

```
The question asked is "Where is the capital of Japan?"
The context provided states that "東京 (Tokyo) is the capital of Japan."
The student's answer is "名古屋 (Nagoya)です。", which translates to "It is Nagoya."

Step by Step Reasoning:
1. Identify the question's subject: The question is about the capital of Japan.
2. Understand the context: The context explicitly states that Tokyo is the capital.
3. Compare the student's answer to the context: The student has named Nagoya as the capital.
4. Assess the accuracy: Since Tokyo is the capital of Japan and not Nagoya, the student's answer is factually incorrect.

GRADE: INCORRECT
```

### EvaluatorType.CONTEXT_QA

```python
evaluator = load_evaluator(evaluator=EvaluatorType.CONTEXT_QA, llm=llm)
eval_result = evaluator.evaluate_strings(
    input="日本の首都はどこですか?",
    prediction="名古屋です。",
    reference="東京は日本の首都です。",
)

print(json.dumps(eval_result, ensure_ascii=False, indent=2))
```

```
{
  "reasoning": "INCORRECT",
  "value": "INCORRECT",
  "score": 0
}
```

#### Input

```
{
  "query": "日本の首都はどこですか?",
  "context": "東京は日本の首都です。",
  "result": "名古屋です。"
}
```

#### Chain1 - HUMAN

```
You are a teacher grading a quiz.
You are given a question, the context the question is about, and the student's answer. You are asked to score the student's answer as either CORRECT or INCORRECT, based on the context.

Example Format:
QUESTION: question here
CONTEXT: context the question is about here
STUDENT ANSWER: student's answer here
GRADE: CORRECT or INCORRECT here

Grade the student answers based ONLY on their factual accuracy. Ignore differences in punctuation and phrasing between the student answer and true answer. It is OK if the student answer contains more information than the true answer, as long as it does not contain any conflicting statements. Begin!

QUESTION: 日本の首都はどこですか?
CONTEXT: 東京は日本の首都です。
STUDENT ANSWER: 名古屋です。
GRADE:
```

#### Chain1 - AI

```
INCORRECT
```

### EvaluatorType.SCORE_STRING

```python
evaluator = load_evaluator(evaluator=EvaluatorType.SCORE_STRING, llm=llm)
eval_result = evaluator.evaluate_strings(
    input="日本の首都はどこですか?",
    prediction="名古屋です。",
)

print(json.dumps(eval_result, ensure_ascii=False, indent=2))
```

```
{
  "reasoning": "The response provided by the AI assistant is not helpful, relevant, or correct. The capital of Japan is Tokyo, not Nagoya. This answer does not demonstrate depth of thought, as it provides an incorrect fact without any additional information or context.\n\nRating: [[1]]",
  "score": 1
}
```

#### Input

```
{
  "prediction": "名古屋です。",
  "input": "日本の首都はどこですか?"
}
```

#### Chain1 - SYSTEM

```
You are a helpful assistant.
```

#### Chain1 - HUMAN

```
[Instruction]
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. For this evaluation, you should primarily consider the following criteria:
helpfulness: Is the submission helpful, insightful, and appropriate?
relevance: Is the submission referring to a real quote from the text?
correctness: Is the submission correct, accurate, and factual?
depth: Does the submission demonstrate depth of thought?
Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".

[Question]
日本の首都はどこですか?

[The Start of Assistant's Answer]
名古屋です。
[The End of Assistant's Answer]
```

#### Chain1 - AI

```
The response provided by the AI assistant is not correct. The capital of Japan is Tokyo, not Nagoya. Therefore, the response fails to meet the correctness criterion. It is also not helpful because it provides incorrect information, and there is no relevance as the assistant's answer does not refer to the real capital of Japan. There is no evidence of depth of thought since the answer is simply wrong.

Rating: [[1]]
```

### EvaluatorType.LABELED_SCORE_STRING

```python
evaluator = load_evaluator(evaluator=EvaluatorType.LABELED_SCORE_STRING, llm=llm)
eval_result = evaluator.evaluate_strings(
    input="日本の首都はどこですか?",
    prediction="名古屋です。",
    reference="東京です。",
)

print(json.dumps(eval_result, ensure_ascii=False, indent=2))
```

```
{
  "reasoning": "The response provided by the AI assistant is incorrect. The user asked for the capital of Japan, which is Tokyo (東京), not Nagoya (名古屋). The quality of the response fails on the criteria of correctness, as the information is factually inaccurate. It is also not helpful as it does not provide the user with the correct answer to their question. There is no depth to the response, and it lacks insight or additional information. The relevance criterion is not met as the response does not refer to a real quote from the text, as the \"ground truth\" states 東京です, which is correct.\n\nRating: [[1]]",
  "score": 1
}
```

#### Input

```
{
  "prediction": "名古屋です。",
  "input": "日本の首都はどこですか?",
  "reference": "東京です。"
}
```

#### Chain1 - SYSTEM

```
You are a helpful assistant.
```

#### Chain1 - HUMAN

```
[Instruction]
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. For this evaluation, you should primarily consider the following criteria:
helpfulness: Is the submission helpful, insightful, and appropriate?
relevance: Is the submission referring to a real quote from the text?
correctness: Is the submission correct, accurate, and factual?
depth: Does the submission demonstrate depth of thought?
[Ground truth]
東京です。
Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".

[Question]
日本の首都はどこですか?

[The Start of Assistant's Answer]
名古屋です。
[The End of Assistant's Answer]
```

#### Chain1 - AI

```
The user's question is asking for the capital of Japan, which is Tokyo (東京). The assistant's response incorrectly states that the capital is Nagoya (名古屋), which is not accurate. The response fails to meet the criteria of correctness as it provides incorrect information. It is also not helpful, as it does not give the user the correct answer to their question, and lacks depth since it does not provide any additional information or context that might have been beneficial even if the main fact was incorrect. The relevance criterion is not met either, as the response does not refer to a real quote from the text and is irrelevant to the user's question.

Rating: [[1]]
```

### EvaluatorType.CRITERIA

```python
evaluator = load_evaluator(evaluator=EvaluatorType.CRITERIA, criteria=Criteria.HELPFULNESS, llm=llm)
eval_result = evaluator.evaluate_strings(
    input="日本の首都はどこですか?",
    prediction="名古屋です。",
)

print(json.dumps(eval_result, ensure_ascii=False, indent=2))
```

```
{
  "reasoning": "Step by Step Reasoning:\n\n1. **Helpfulness**: The submission should provide an accurate and relevant answer to the input question. The input is asking for the capital of Japan.\n\n2. **Insightfulness**: While this criterion is often subject to interpretation, in the context of a factual question like the one presented, insightfulness can be evaluated based on the depth and accuracy of the information provided in response to the question.\n\n3. **Appropriateness**: The submission should be appropriate to the question, meaning it should directly address what is asked without including irrelevant or incorrect information.\n\nEvaluating the Submission:\n\n- The input question in Japanese \"日本の首都はどこですか?\" translates to \"Where is the capital of Japan?\".\n- The correct answer to this question is \"東京\" (Tokyo), as Tokyo is the capital of Japan.\n- The submitted answer \"名古屋です。\" states that Nagoya is the capital, which is incorrect.\n- Based on this, the submission is not helpful because it provides the wrong answer.\n- The submission is not insightful as it does not provide accurate information or any depth regarding the question asked.\n- The submission is not appropriate because it does not correctly address the factual question.\n\nConclusion:\nThe submission does not meet the criterion of helpfulness, insightfulness, and appropriateness because it provides an incorrect answer to the question asked.\n\nN",
  "value": "N",
  "score": 0
}
```

#### Input

```
{
  "input": "日本の首都はどこですか?",
  "output": "名古屋です。"
}
```

#### Chain1 - HUMAN

```
You are assessing a submitted answer on a given task or input based on a set of criteria. Here is the data:
[BEGIN DATA]
***
[Input]: 日本の首都はどこですか?
***
[Submission]: 名古屋です。
***
[Criteria]: helpfulness: Is the submission helpful, insightful, and appropriate? If so, respond Y. If not, respond N.
***
[END DATA]
Does the submission meet the Criteria? First, write out in a step by step manner your reasoning about each criterion to be sure that your conclusion is correct. Avoid simply stating the correct answers at the outset. Then print only the single character "Y" or "N" (without quotes or punctuation) on its own line corresponding to the correct answer of whether the submission meets all criteria. At the end, repeat just the letter again by itself on a new line.
```

#### Chain1 - AI

```
Step 1: Determine if the submission is helpful.
- To be helpful, the submission should provide accurate information in response to the input question.
- The input question is "日本の首都はどこですか?" which translates to "Where is the capital of Japan?"

Step 2: Assess the accuracy of the submission.
- The submission states "名古屋です" which translates to "It is Nagoya."
- Nagoya is a major city in Japan, but it is not the capital. The capital of Japan is Tokyo.

Step 3: Decide if the submission is insightful and appropriate.
- Insightful means that it provides a deeper understanding or useful information in response to the question.
- Appropriate means that the submission is relevant and suitable in the context of the question.

Step 4: Evaluate the overall helpfulness of the submission.
- Since the information provided is incorrect, it does not enhance understanding, nor is it relevant or suitable in answering the question about the capital of Japan.

Conclusion: The submission is not helpful, insightful, or appropriate because it provides incorrect information in response to the question asked.

N

N
```

### EvaluatorType.LABELED_CRITERIA

```python
evaluator = load_evaluator(evaluator=EvaluatorType.LABELED_CRITERIA, criteria=Criteria.HELPFULNESS, llm=llm)
eval_result = evaluator.evaluate_strings(
    input="日本の首都はどこですか?",
    prediction="名古屋です。",
    reference="東京です。",
)

print(json.dumps(eval_result, ensure_ascii=False, indent=2))
```

```
{
  "reasoning": "Step by Step Reasoning:\n\n1. The input asks for the capital of Japan in Japanese: \"日本の首都はどこですか?\" which translates to \"Where is the capital of Japan?\"\n\n2. The submission provides the answer: \"名古屋です。\" which translates to \"It is Nagoya.\"\n\n3. The criteria for assessment include helpfulness, insightfulness, and appropriateness.\n\n4. To be helpful, the submission should provide the correct answer or useful information that addresses the question.\n\n5. To be insightful, the submission should show an understanding of the question and the subject matter.\n\n6. To be appropriate, the submission should be relevant to the question asked.\n\n7. The reference information provided states: \"東京です。\" which translates to \"It is Tokyo.\" This indicates that the correct answer to the input question is Tokyo, not Nagoya.\n\n8. Since the submission provides an incorrect answer, it fails to be helpful or insightful as it does not provide the correct information or show an understanding of the true capital of Japan.\n\n9. Despite being a city in Japan, the submission is not appropriate in the context of the question since it does not correctly identify the capital of Japan.\n\n10. Based on these points, the submission does not meet the criteria of helpfulness, insightfulness, or appropriateness because it provides incorrect information.\n\nConclusion:\n\nN",
  "value": "N",
  "score": 0
}
```

#### Input

```
{
  "input": "日本の首都はどこですか?",
  "output": "名古屋です。",
  "reference": "東京です。"
}
```

#### Chain1 - HUMAN

```
You are assessing a submitted answer on a given task or input based on a set of criteria. Here is the data:
[BEGIN DATA]
***
[Input]: 日本の首都はどこですか?
***
[Submission]: 名古屋です。
***
[Criteria]: helpfulness: Is the submission helpful, insightful, and appropriate? If so, respond Y. If not, respond N.
***
[Reference]: 東京です。
***
[END DATA]
Does the submission meet the Criteria? First, write out in a step by step manner your reasoning about each criterion to be sure that your conclusion is correct. Avoid simply stating the correct answers at the outset. Then print only the single character "Y" or "N" (without quotes or punctuation) on its own line corresponding to the correct answer of whether the submission meets all criteria. At the end, repeat just the letter again by itself on a new line.
```

#### Chain1 - AI

```
Step 1: Assess helpfulness. The submitted answer should provide the correct information in response to the question asked. The question is asking for the capital of Japan.

Step 2: Compare the submission to the reference. The reference information indicates that the capital of Japan is Tokyo (東京です).

Step 3: Determine if the submission is appropriate. The submission states that the capital of Japan is Nagoya (名古屋です), which is incorrect.

Step 4: Conclusion. The submission is not helpful or insightful because it provides incorrect information regarding the capital of Japan, and therefore, it is not appropriate as an answer to the question asked.

N

N
```

### EvaluatorType.STRING_DISTANCE

```python
evaluator = load_evaluator(evaluator=EvaluatorType.STRING_DISTANCE)
eval_result = evaluator.evaluate_strings(
    prediction="名古屋です。",
    reference="東京です。",
)

print(json.dumps(eval_result, ensure_ascii=False, indent=2))
```

```
{
  "score": 0.29999999999999993
}
```

### EvaluatorType.EXACT_MATCH

```python
evaluator = load_evaluator(evaluator=EvaluatorType.EXACT_MATCH)
eval_result = evaluator.evaluate_strings(
    prediction="名古屋です。",
    reference="東京です。",
)

print(json.dumps(eval_result, ensure_ascii=False, indent=2))
```

```
{
  "score": 0
}
```

### EvaluatorType.REGEX_MATCH

```python
evaluator = load_evaluator(evaluator=EvaluatorType.REGEX_MATCH)
eval_result = evaluator.evaluate_strings(
    prediction="名古屋です。",
    reference="\w+です。",
)

print(json.dumps(eval_result, ensure_ascii=False, indent=2))
```

```
{
  "score": 1
}
```

### EvaluatorType.EMBEDDING_DISTANCE

```python
from langchain_openai import OpenAIEmbeddings

evaluator = load_evaluator(evaluator=EvaluatorType.EMBEDDING_DISTANCE, llm=llm) # llmは使用されず、OpenAIEmbeddingsが使用されます。
eval_result = evaluator.evaluate_strings(
    prediction="名古屋です。",
    reference="東京です。",
)

print(json.dumps(eval_result, ensure_ascii=False, indent=2))
```

```
{
  "score": 0.10948750922371109
}
```

### EvaluatorType.JSON_VALIDITY

```python
pred_json_str = """{
    "name": "名古屋",
    "pop": 100,
    "address": "日本",
}"""

evaluator = load_evaluator(evaluator=EvaluatorType.JSON_VALIDITY)
eval_result = evaluator.evaluate_strings(
    prediction=pred_json_str,
)

print(json.dumps(eval_result, ensure_ascii=False, indent=2))
```

```
{
  "score": 0,
  "reasoning": "Expecting property name enclosed in double quotes: line 5 column 1 (char 58)"
}
```

### EvaluatorType.JSON_EQUALITY

```python
pred_json_str = """{
    "name": "名古屋",
    "pop": 100,
    "address": "日本"
}"""
ref_json_str = """{
    "name": "東京",
    "pop": 1000,
    "address": "日本"
}"""

evaluator = load_evaluator(evaluator=EvaluatorType.JSON_EQUALITY)
eval_result = evaluator.evaluate_strings(
    prediction=pred_json_str, reference=ref_json_str
)

print(json.dumps(eval_result, ensure_ascii=False, indent=2))
```

```
{
  "score": false
}
```

### EvaluatorType.JSON_EDIT_DISTANCE

```python
pred_json_str = """{
    "name": "名古屋",
    "pop": 100,
    "address": "日本"
}"""
ref_json_str = """{
    "name": "東京",
    "pop": 1000,
    "address": "日本"
}"""

evaluator = load_evaluator(evaluator=EvaluatorType.JSON_EDIT_DISTANCE)
eval_result = evaluator.evaluate_strings(
    prediction=pred_json_str, reference=ref_json_str
)

print(json.dumps(eval_result, ensure_ascii=False, indent=2))
```

```
{
  "score": 0.203125
}
```

### EvaluatorType.JSON_SCHEMA_VALIDATION

```python
pred_json_str = """{
    "name": "名古屋",
    "pop": 100,
    "address": "日本"
}"""
ref_json_str = """{
  "type": "object",
  "properties": {
    "name": {
      "type": "string"
    },
    "pop": {
      "type": "string"
    },
    "address": {
      "type": "string"
    }
  }
}"""

evaluator = load_evaluator(evaluator=EvaluatorType.JSON_SCHEMA_VALIDATION)
eval_result = evaluator.evaluate_strings(
    prediction=pred_json_str, reference=ref_json_str
)

print(json.dumps(eval_result, ensure_ascii=False, indent=2))
```

```
{
  "score": false,
  "reasoning": "<ValidationError: \"100 is not of type 'string'\">"
}
```

## 参考文献

-   [LangChain - String Evaluators](https://python.langchain.com/docs/guides/evaluation/string/)
