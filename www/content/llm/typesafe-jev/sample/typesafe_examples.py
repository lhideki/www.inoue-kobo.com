# TypeSafe AI Jev: 桃太郎を使った利用例
# 実行すると4回のAPIリクエストを送信します。

from dotenv import load_dotenv

load_dotenv(".env")

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
