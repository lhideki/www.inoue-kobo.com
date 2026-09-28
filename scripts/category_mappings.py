from dataclasses import dataclass

# Shared by qiita_sync.py and zenn_sync.py so both post the same categories
# under the same directories. `emoji` is only used by zenn_sync.py (Zenn
# requires one per article) but lives here to keep a single source of truth.


@dataclass
class CategoryMapping:
    tag: str
    dir: str
    emoji: str


CATEGORY_MAPPINGS = [
    CategoryMapping('MachineLearning', 'www/content/ai_ml', '🤖'),
    CategoryMapping('AWS', 'www/content/aws', '☁️'),
    CategoryMapping('REST-API', 'www/content/restapi', '🔌'),
    CategoryMapping('Discord', 'www/content/discord', '🎮'),
    CategoryMapping('LLM', 'www/content/llm', '🧠'),
    CategoryMapping('プロジェクト管理', 'www/content/project-management', '📋'),
]
