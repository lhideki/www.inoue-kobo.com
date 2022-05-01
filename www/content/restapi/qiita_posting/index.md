---
title: "GitHubにpushする際にQiitaにも投稿する"
date: "2018-11-25"
tags:
  - "Qiita"
  - "Git"
thumbnail: "restapi/qiita_posting/images/thumbnail.png"
---
# GitHubにpushする際にQiitaにも投稿する

## TL;DR

MarkdownドキュメントをGitHubにpushする際に、hooksを利用してQiitaにも自動的に投稿するようにします。

## 構築

### 概要

1. Qiitaのアクセストークンを発行する。
2. 投稿用のPythonプログラムを作成する。
3. gitでチェックアウトしたファイルの最終更新日を修正する。
4. gitのpre-pushに設置する。

### Qiitaのアクセストークンを発行する

QiitaのREST APIに投稿するためにアクセストークンを発行します。
以下のリンクから`個人用アクセストークン`を発行してください。

* [https://qiita.com/settings/applications](https://qiita.com/settings/applications)

トークンは`QIITA_TOKEN`という名前で環境変数に設定します。

### 投稿用のPythonプログラムを作成する

投稿用のPythonプログラムでは以下の処理を行います。

* ローカルのMarkdownドキュメントの一覧を取得する。
* Qiita上の記事一覧を取得する。
* Qiita上の記事とローカルのドキュメントの最終更新日を比較する。
* ローカル上にあって、Qiita上に存在しないドキュメントを新規に投稿する。
* Qiita上に存在するがローカル上のドキュメントの方が最終更新日が新しいQiita上の記事を更新する。

ここでは`scripts/article_manager.py`として以下のコードを保存します。

```python
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
FQDN = 'https://www.example.com'
DOCUMENT_ROOT = 'docs'


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
            self.title = self._parse_title(lines)
            self.lines = lines
            self.body = ''.join(lines)
            self.last_updated_date = datetime.datetime.fromtimestamp(
                os.stat(filename).st_mtime)

    def _parse_title(self, lines):
        return re.search('\# (.+)', lines[0]).group(1)

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
        ['MarchineLearning', 'docs/ai_ml'],
        ['AWS', 'docs/aws'],
        ['REST API', 'docs/restapi']
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
                    qiita_articles.update(id, qiita_article.get_tags(), my_article)
                else:
                    logging.info(f'{my_article.get_title()}(new) will post.')
                    qiita_articles.post(my_article)
            else:
                logging.info(f'{my_article.get_title()} is not changed.')

```

### gitでチェックアウトしたファイルの最終更新日を修正する

gitはチェックアウト(clone)したファイルの最終更新日は、チェックアウト日になっています。
ファイルの更新日(commitした日)ではないため、そのままではQiita上の記事の最終更新日と比較ができません。

解決策は色々ありますが、今回は以下の記事にある公式が紹介しているPerlスクリプトを使用します。

* [checkoutしたファイルのmtimeを、そのファイルがcommitされた時刻に合わせたい ― svnとgitの場合](http://d.hatena.ne.jp/hirose31/20090106/1231171068)

`scripts/git-set-file-times.pl`という名前のファイルを作成し、以下のPerlコードを記載します。

```perl
#!/usr/bin/perl

my %attributions;
my @files;

open IN, "git ls-tree -r --full-name HEAD |" or die;
while (<IN>) {
	if (/^\S+\s+blob \S+\s+(\S+)$/) {
		push(@files, $1);
		$attributions{$1} = -1;
	}
}
close IN;

my $remaining = $#files + 1;

open IN, "git log -r --root --raw --no-abbrev --pretty=format:%h~%an~%ad~ |" or die;
while (<IN>) {
	if (/^([^:~]+)~(.*)~([^~]+)~$/) {
		($commit, $author, $date) = ($1, $2, $3);
	} elsif (/^:\S+\s+1\S+\s+\S+\s+\S+\s+\S\s+(.*)$/) {
		if ($attributions{$1} == -1) {
			$attributions{$1} = "$author, $date ($commit)";
			$remaining--;
			if ($remaining <= 0) {
				break;
			}
		}
	}
}
close IN;

for $f (@files) {
	print "$f	$attributions{$f}\n";
}
```

### gitのpre-pushに設置する

git cloneした直下に以下のファイルを作成します。既にhooksを利用している場合は、内容を追記してください。

* .git/hooks/pre-push

記載する内容は以下の通りです。

```sh
#!/usr/bin/env bash

echo 'mtime updating.'
perl scripts/git-set-file-times.pl
echo 'check update for qiita posted.'
python scripts/article_manager.py
```

## 運用

`article_manager.py`の`my_dirs`に設定したローカルディレクトリ内の`index.md`について、
GitHubにpushする際に自動的にQiita上に投稿されるようになります。

### Qiitaのタグに関する注意事項

Qiitaのタグは記事を一意に識別するための識別子の一部になっています。
`id`が一致してもタグが一致してないと別記事となり、記事の更新ができないため注意してください。

投稿も同様で、既存のタグを設定しないと`404 NotFound`になります。