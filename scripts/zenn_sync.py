import os
import re
import glob
import hashlib
import logging

import yaml

from qiita_sync import MyArticle
from category_mappings import CATEGORY_MAPPINGS

logging.basicConfig(level=logging.INFO)

# Zenn's GitHub integration has no posting API: it watches the connected
# repository's articles/ directory directly, so "syncing" here means
# regenerating that directory's Markdown files and letting the normal
# git commit/push make the change visible to Zenn.
ARTICLES_DIR = 'articles'
MAX_TOPICS = 5
SLUG_MIN_LEN = 12
SLUG_MAX_LEN = 50


def make_slug(name):
    # Zenn requires slugs of 12-50 chars from [a-z0-9_-]. Article directory
    # names are already close to that shape, so only pad/truncate with a
    # deterministic hash when they fall outside the allowed range, keeping
    # the same source article mapped to the same slug across runs.
    slug = re.sub(r'[^a-z0-9_-]', '-', name.lower()).strip('-')
    if len(slug) < SLUG_MIN_LEN:
        digest = hashlib.md5(name.encode()).hexdigest()
        pad = digest[:max(1, SLUG_MIN_LEN - len(slug) - 1)]
        slug = f'{slug}-{pad}'
    if len(slug) > SLUG_MAX_LEN:
        digest = hashlib.md5(name.encode()).hexdigest()[:6]
        slug = f'{slug[:SLUG_MAX_LEN - len(digest) - 1]}-{digest}'
    return slug


def parse_front_matter(filename):
    with open(filename) as f:
        lines = f.readlines()
    if not lines or lines[0].rstrip('\n') != '---':
        return {}
    for i, line in enumerate(lines[1:], start=1):
        if line.rstrip('\n') == '---':
            try:
                return yaml.safe_load(''.join(lines[1:i])) or {}
            except yaml.YAMLError:
                return {}
    return {}


class ZennArticle:
    def __init__(self, mapping, filename):
        self.mapping = mapping
        self.article = MyArticle(mapping.tag, filename)
        self.front_matter = parse_front_matter(filename)
        self.slug = make_slug(os.path.basename(os.path.dirname(filename)))

    def get_topics(self):
        # Zenn topics only accept alphanumerics, so tags like "Project
        # Management" or the Japanese category tag get their separators and
        # non-ASCII characters stripped; anything left empty is dropped.
        tags = self.front_matter.get('tags') or [self.mapping.tag]
        cleaned = []
        for tag in tags:
            normalized = re.sub(r'[^0-9a-zA-Z]', '', str(tag))
            if normalized and normalized not in cleaned:
                cleaned.append(normalized)
        if not cleaned:
            cleaned = ['tech']
        return cleaned[:MAX_TOPICS]

    def to_zenn_markdown(self):
        front_matter = {
            'title': self.article.get_title(),
            'emoji': self.mapping.emoji,
            'type': 'tech',
            'topics': self.get_topics(),
            'published': True,
        }
        header = yaml.dump(front_matter, allow_unicode=True, sort_keys=False)
        return f'---\n{header}---\n\n{self.article.to_posting_format()}'

    def get_output_path(self):
        return os.path.join(ARTICLES_DIR, f'{self.slug}.md')


def collect_articles():
    articles = []
    for mapping in CATEGORY_MAPPINGS:
        filenames = glob.glob(os.path.join(mapping.dir, '**', '*.md'), recursive=True)
        for filename in filenames:
            articles.append(ZennArticle(mapping, filename))
    return articles


def sync():
    os.makedirs(ARTICLES_DIR, exist_ok=True)

    managed_paths = set()
    changed = False

    for article in collect_articles():
        output_path = article.get_output_path()
        managed_paths.add(output_path)
        new_content = article.to_zenn_markdown()

        existing_content = None
        if os.path.exists(output_path):
            with open(output_path) as f:
                existing_content = f.read()

        if existing_content != new_content:
            logging.info(f'{output_path} will be written.')
            with open(output_path, 'w') as f:
                f.write(new_content)
            changed = True

    for filename in glob.glob(os.path.join(ARTICLES_DIR, '*.md')):
        if filename not in managed_paths:
            logging.info(f'{filename} will be removed.')
            os.remove(filename)
            changed = True

    return changed


def main():
    if sync():
        logging.info('Zenn articles were updated.')
    else:
        logging.info('No changes to Zenn articles.')


if __name__ == '__main__':
    main()
