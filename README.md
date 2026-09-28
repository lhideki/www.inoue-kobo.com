# www.inoue-kobo.com

[www.inoue-kobo.com](https://www.inoue-kobo.com/) のソースリポジトリです。[Hugo](https://gohugo.io/) で構築した静的サイトを AWS(S3 + CloudFront)にホスティングしており、記事の一部は GitHub Actions 経由で [Qiita](https://qiita.com/) にも自動投稿されます。

## 構成

```
.
├── www/                    # Hugo サイト本体
│   ├── config.toml         # Hugo設定
│   ├── content/            # 記事(カテゴリごとのMarkdown)
│   ├── layouts/            # レイアウトのカスタマイズ
│   ├── static/              # 静的ファイル
│   └── themes/mainroad/    # テーマ(git submodule)
├── scripts/
│   ├── qiita_sync.py       # www/content配下の記事とQiitaを同期するスクリプト
│   └── requirements.txt
└── .github/workflows/
    ├── main.yml            # Hugoビルド & S3/CloudFrontへのデプロイ
    └── qiita-post.yml      # Qiitaへの自動投稿
```

## セットアップ

テーマを git submodule で取り込んでいるため、クローン時は `--recurse-submodules` を付けるか、後から初期化してください。

```bash
git clone --recurse-submodules <このリポジトリのURL>
# もしくはクローン後に
git submodule update --init --recursive
```

[Hugo](https://gohugo.io/installation/)(extended版, 0.138.0系で動作確認)をインストールしてください。

## ローカルでの確認

```bash
cd www
hugo server
```

`http://localhost:1313/` でプレビューできます。

## 記事の追加・更新

`www/content/<カテゴリ>/` 配下に Markdown ファイルを追加・編集します。ファイル冒頭の `# タイトル` がそのまま記事タイトルとして扱われます(Qiita連携も含む)。

## デプロイ

`master` ブランチへの push で `.github/workflows/main.yml` が実行され、以下を行います。

1. S3上のトピックファイルを `www/content` に同期
2. `hugo --environment production` でビルド
3. ビルド結果を S3 にアップロードし、CloudFront のキャッシュを無効化

毎週火曜9:00(JST)にも定期実行されます(`workflow_dispatch` での手動実行も可能)。

必要な GitHub Secrets:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `HOMEPAGE_S3_BUCKET`

## Qiitaへの自動投稿

`www/content/**` の変更(`about.md` / `schedule.md` / `trainings.md` / `training/` / `webservice/` を除く)を含む push で `.github/workflows/qiita-post.yml` が実行され、`scripts/qiita_sync.py` が以下を行います。

- `www/content/{ai_ml,aws,restapi,discord,llm}` 配下の記事をタイトルでQiita側の記事と突き合わせ
- gitのコミット日時がQiita側の最終更新日時より新しい記事のみ新規投稿 / 更新
- 画像の相対パスは `https://www.inoue-kobo.com` を付与した絶対URLに変換して投稿

必要な GitHub Secrets:

- `QIITA_TOKEN`

手動で実行する場合:

```bash
export QIITA_TOKEN=xxxx
pip install -r scripts/requirements.txt
python scripts/qiita_sync.py
```
