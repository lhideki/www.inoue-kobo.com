import os
from abc import ABCMeta, abstractmethod
import requests
import datetime
import json
import re
import glob
import logging

logging.basicConfig(level=logging.INFO)
MAX_PAGES = 100
TOKEN = os.environ['QIITA_TOKEN']
FQDN = 'https://www.inoue-kobo.com'
DOCUMENT_ROOT = 'www/content'


class Article(metaclass=ABCMeta):
    def get_title(self):
        return self.title

    def get_body(self):
        return self.body

    def get_last_updated_date(self):
        return self.last_updated_date

    def to_posting_format(self):
        self.get_body()


class MyArticle(Article):
    def __init__(self, tag, filename):
        self.tag = tag
        self.filename = filename
        with open(filename) as f:
            lines = f.readlines()
            removed_meta_lines = self._remove_meta(lines)
            self.title = self._parse_title(removed_meta_lines)
            self.lines = removed_meta_lines
            self.body = ''.join(removed_meta_lines)
            self.last_updated_date = datetime.datetime.fromtimestamp(
                os.stat(filename).st_mtime)

    def _remove_meta(self, lines):
        new_lines = []
        is_in_meta = False
        for i, line in enumerate(lines):
            if i == 0 and re.match('^\-\-\-\\n$', line):
                is_in_meta = True
                continue
            if is_in_meta and re.match('^\-\-\-\\n$', line):
                is_in_meta = False
                continue
            if is_in_meta:
                continue

            new_lines.append(line)

        return new_lines

    def _parse_title(self, lines):
        for line in lines:
            if re.match('\# (.+)', line):
                return re.search('\# (.+)', line).group(1)
        return 'No Title'

    def get_filename(self):
        return self.filename

    def get_tag(self):
        return self.tag

    def to_posting_format(self):
        converted = []

        for line in self.lines:
            matched = re.match(r'^\!\[(.*)\]\((.+)\)$', line)
            if not matched:
                converted.append(line)
                continue
            alt = matched.group(1)
            relative_src = matched.group(2)
            parent_dir = os.path.dirname(self.filename)[len(DOCUMENT_ROOT):]
            new_src = os.path.join(parent_dir, relative_src)
            img = f'![{alt}]({FQDN}{new_src})\n'
            converted.append(img)
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
        dtime = datetime.datetime.strptime(
            colon_removed, '%Y-%m-%dT%H:%M:%S%z')

        return dtime


class Articles(metaclass=ABCMeta):
    def fetch(self):
        pass

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

        filenames = glob.glob(os.path.join(dir, '**', '*.md'), recursive=True)
        for filename in filenames:
            article = MyArticle(tag, filename)
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
            raise Exception(res)

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
            raise Exception(res)

    def fetch(self):
        self.articles = {}

        for i in range(1, self.max_pages + 1):
            params = {
                'page': i,
                'per_page': 100
            }
            res = requests.get(self.fetch_url, params, headers=self.headers)
            if res.status_code >= 300:
                raise Exception(res)
            if len(res.json()) == 0:
                break
            for source in res.json():
                article = QiitaArticle(source)
                self.articles[article.get_title()] = article


class UpdateChecker():
    def __init__(self, qiita_articles):
        self.qiita_articles = qiita_articles

    def check(self, my_articles):
        self.my_articles = my_articles
        self.results = []

        for my_article in self.my_articles.list():
            if self.qiita_articles.exist(my_article.get_title()):
                qiita_article = self.qiita_articles.get(my_article.get_title())
                result = [
                    qiita_article.get_id(),
                    my_article.get_title(),
                    my_article.get_last_updated_date().timestamp(
                    ) > qiita_article.get_last_updated_date().timestamp(),
                    my_article,
                    qiita_article
                ]
            else:
                result = [
                    None,
                    my_article.get_title(),
                    True,
                    my_article,
                    None
                ]
            self.results.append(result)

    def get_results(self):
        return self.results


if __name__ == '__main__':
    qiita_articles = QiitaArticles(TOKEN, MAX_PAGES)
    qiita_articles.fetch()

    my_dirs = [
        ['MachineLearning', 'www/content/ai_ml'],
        ['AWS', 'www/content/aws'],
        ['REST-API', 'www/content/restapi']
    ]
    for my_dir in my_dirs:
        my_articles = MyArticles(my_dir[0], my_dir[1])
        my_articles.fetch()

        checker = UpdateChecker(qiita_articles)
        checker.check(my_articles)

        results = checker.get_results()
        for result in results:
            id = result[0]
            is_changed = result[2]
            my_article = result[3]
            qiita_article = result[4]
            if is_changed:
                if id:
                    logging.info(
                        f'{my_article.get_title()}({id}) will update.')
                    qiita_articles.update(
                        id, qiita_article.get_tags(), my_article)
                else:
                    logging.info(f'{my_article.get_title()}(new) will post.')
                    qiita_articles.post(my_article)
            else:
                #logging.info(f'{my_article.get_title()} is not changed.')
                pass
