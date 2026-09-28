import os
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Optional
import requests
import datetime
import re
import glob
import logging
import subprocess

logging.basicConfig(level=logging.INFO)
MAX_PAGES = 100
FQDN = 'https://www.inoue-kobo.com'
DOCUMENT_ROOT = 'www/content'


@dataclass
class CategoryMapping:
    tag: str
    dir: str


CATEGORY_MAPPINGS = [
    CategoryMapping('MachineLearning', 'www/content/ai_ml'),
    CategoryMapping('AWS', 'www/content/aws'),
    CategoryMapping('REST-API', 'www/content/restapi'),
    CategoryMapping('Discord', 'www/content/discord'),
    CategoryMapping('LLM', 'www/content/llm'),
    CategoryMapping('プロジェクト管理', 'www/content/project-management'),
]


class Article:
    def get_title(self):
        return self.title

    def get_body(self):
        return self.body

    def get_last_updated_date(self):
        return self.last_updated_date


class MyArticle(Article):
    def __init__(self, tag, filename):
        self.tag = tag
        self.filename = filename
        with open(filename) as f:
            lines = f.readlines()
        content_lines = self._remove_front_matter(lines)
        self.title = self._parse_title(content_lines)
        self.lines = content_lines
        self.body = ''.join(content_lines)
        self.last_updated_date = self._get_last_updated_date(filename)

    def _get_last_updated_date(self, filename):
        # actions/checkout resets every file's mtime to checkout time, so we
        # must use git's commit history (not the filesystem) to know when a
        # file actually last changed. Falls back to mtime outside of git
        # (e.g. shallow clones without history, or local uncommitted edits).
        try:
            result = subprocess.run(
                ['git', 'log', '-1', '--format=%aI', '--', filename],
                capture_output=True, text=True, check=True)
            iso_date = result.stdout.strip()
            if iso_date:
                return datetime.datetime.fromisoformat(iso_date)
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        return datetime.datetime.fromtimestamp(os.stat(filename).st_mtime)

    def _remove_front_matter(self, lines):
        if not lines or lines[0].rstrip('\n') != '---':
            return lines

        for i, line in enumerate(lines[1:], start=1):
            if line.rstrip('\n') == '---':
                return lines[i + 1:]

        # No closing marker found: treat as not having front matter at all,
        # rather than silently dropping the whole file's content.
        return lines

    def _parse_title(self, lines):
        for line in lines:
            matched = re.match(r'^#\s+(.+)', line)
            if matched:
                return matched.group(1).strip()
        return 'No Title'

    def get_filename(self):
        return self.filename

    def get_tag(self):
        return self.tag

    def to_posting_format(self):
        converted = []

        for line in self.lines:
            matched = re.match(r'^!\[(.*)\]\((.+)\)$', line)
            if not matched:
                converted.append(line)
                continue
            alt, relative_src = matched.group(1), matched.group(2)
            parent_dir = os.path.dirname(self.filename)[len(DOCUMENT_ROOT):]
            new_src = os.path.join(parent_dir, relative_src)
            converted.append(f'![{alt}]({FQDN}{new_src})\n')
        return ''.join(converted)


class QiitaArticle(Article):
    def __init__(self, source):
        self.source = source
        self.id = source['id']
        self.title = source['title']
        self.body = source['body']
        self.tags = source['tags']
        self.last_updated_date = self._parse_date(source['updated_at'])

    def get_id(self):
        return self.id

    def get_tags(self):
        return self.tags

    def _parse_date(self, datestr):
        colon_removed = datestr[0:22] + datestr[23:]
        return datetime.datetime.strptime(colon_removed, '%Y-%m-%dT%H:%M:%S%z')


class Articles(metaclass=ABCMeta):
    @abstractmethod
    def fetch(self):
        raise NotImplementedError

    def get(self, title):
        return self.articles[title]

    def list(self):
        return self.articles.values()

    def exist(self, title):
        return title in self.articles


class MyArticles(Articles):
    def __init__(self, tag, dir):
        self.tag = tag
        self.dir = dir
        self.articles = {}

    def fetch(self):
        filenames = glob.glob(os.path.join(self.dir, '**', '*.md'), recursive=True)
        for filename in filenames:
            article = MyArticle(self.tag, filename)
            self.articles[article.get_title()] = article


class QiitaArticles(Articles):
    def __init__(self, token, max_pages):
        self.token = token
        self.max_pages = max_pages
        self.fetch_url = 'https://qiita.com/api/v2/authenticated_user/items'
        self.post_url = 'https://qiita.com/api/v2/items'
        self.headers = {
            'Content-Type': 'application/json',
            'charset': 'utf-8',
            'Authorization': f'Bearer {self.token}'
        }
        self.articles = {}

    def post(self, my_article):
        params = {
            'title': my_article.get_title(),
            'body': my_article.to_posting_format(),
            'private': False,
            'tags': [{
                'name': my_article.get_tag(),
                'versions': []
            }]
        }
        res = requests.post(self.post_url, json=params, headers=self.headers)
        if res.status_code >= 300:
            raise RuntimeError(f'Qiita API error ({res.status_code}): {res.text}')

    def update(self, id, tags, my_article):
        params = {
            'title': my_article.get_title(),
            'body': my_article.to_posting_format(),
            'private': False,
            'tags': tags
        }
        res = requests.patch(f'{self.post_url}/{id}',
                              json=params, headers=self.headers)
        if res.status_code >= 300:
            raise RuntimeError(f'Qiita API error ({res.status_code}): {res.text}')

    def fetch(self):
        self.articles = {}

        for page in range(1, self.max_pages + 1):
            params = {
                'page': page,
                'per_page': 100
            }
            res = requests.get(self.fetch_url, params, headers=self.headers)
            if res.status_code >= 300:
                raise RuntimeError(f'Qiita API error ({res.status_code}): {res.text}')
            if len(res.json()) == 0:
                break
            for source in res.json():
                article = QiitaArticle(source)
                self.articles[article.get_title()] = article


@dataclass
class CheckResult:
    qiita_id: Optional[str]
    title: str
    is_changed: bool
    my_article: MyArticle
    qiita_article: Optional[QiitaArticle]


class UpdateChecker():
    def __init__(self, qiita_articles):
        self.qiita_articles = qiita_articles

    def check(self, my_articles):
        self.results = []

        for my_article in my_articles.list():
            title = my_article.get_title()
            if self.qiita_articles.exist(title):
                qiita_article = self.qiita_articles.get(title)
                is_changed = my_article.get_last_updated_date().timestamp() \
                    > qiita_article.get_last_updated_date().timestamp()
                result = CheckResult(
                    qiita_article.get_id(), title, is_changed, my_article, qiita_article)
            else:
                result = CheckResult(None, title, True, my_article, None)
            self.results.append(result)

    def get_results(self):
        return self.results


def main():
    token = os.environ.get('QIITA_TOKEN')
    if not token:
        logging.error('QIITA_TOKEN environment variable is not set.')
        raise SystemExit(1)

    qiita_articles = QiitaArticles(token, MAX_PAGES)
    qiita_articles.fetch()

    had_errors = False
    for mapping in CATEGORY_MAPPINGS:
        my_articles = MyArticles(mapping.tag, mapping.dir)
        my_articles.fetch()

        checker = UpdateChecker(qiita_articles)
        checker.check(my_articles)

        for result in checker.get_results():
            if not result.is_changed:
                continue
            try:
                if result.qiita_id:
                    logging.info(f'{result.title}({result.qiita_id}) will update.')
                    qiita_articles.update(
                        result.qiita_id, result.qiita_article.get_tags(), result.my_article)
                else:
                    logging.info(f'{result.title}(new) will post.')
                    qiita_articles.post(result.my_article)
            except RuntimeError as e:
                had_errors = True
                logging.error(f'Failed to sync "{result.title}": {e}')

    if had_errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
