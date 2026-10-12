import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from article_urls import article_base_url, resolve_article_urls
from qiita_sync import MyArticle, QiitaArticles, main, sync_selected_article

BASE = 'https://www.inoue-kobo.com/llm/example/'


class ArticleURLTests(unittest.TestCase):
    def convert(self, value):
        return resolve_article_urls(value, BASE)

    def test_html_keeps_formatting_and_resolves_multiline_attributes(self):
        original = '''<figure class="sample">
  <a href='images/full.png?x=1&amp;y=2#view'>
    <IMG alt='example src="do-not-change"'
         src="./images/small.png" width="900" loading="lazy">
  </a>
</figure>
'''
        expected = original.replace("href='images/", "href='" + BASE + 'images/')
        expected = expected.replace('src="./images/', 'src="' + BASE + 'images/')
        self.assertEqual(self.convert(original), expected)

    def test_inline_html_and_unquoted_attributes(self):
        self.assertEqual(self.convert('Text <img src=images/a.png> and <a href=../other/>link</a>.'),
                         f'Text <img src={BASE}images/a.png> and <a href=https://www.inoue-kobo.com/llm/other/>link</a>.')

    def test_markdown_images_in_paragraphs_lists_and_links(self):
        for original in ['![alt](images/a.png)\n',
                         'Before ![alt](<images/a.png> "title") after.\n',
                         '> - ![alt](images/a.png)\n',
                         '[![alt](images/a.png)](https://example.com)\n']:
            with self.subTest(original=original):
                self.assertEqual(self.convert(original), original.replace('images/a.png', BASE + 'images/a.png'))

    def test_absolute_urls_and_page_references_unchanged(self):
        for url in ['https://cdn.example/a.png', 'http://example/a.png', '//cdn.example/a.png',
                    'data:image/png;base64,abc', '#here', '?size=large', 'mailto:test@example.com']:
            original = f'<a href="{url}">link</a><img src="{url}">\n'
            self.assertEqual(self.convert(original), original)
        self.assertEqual(self.convert('![a](https://example.com/a.png)'), '![a](https://example.com/a.png)')

    def test_root_relative_and_parent_paths(self):
        self.assertEqual(self.convert('![a](/shared/a.png) ![b](../b.png?x=1#v)'),
                         '![a](https://www.inoue-kobo.com/shared/a.png) ![b](https://www.inoue-kobo.com/llm/b.png?x=1#v)')

    def test_code_examples_comments_and_escaped_images_unchanged(self):
        cases = ['```html\n<img src="images/a.png">\n![a](images/a.png)\n```\n',
                 '~~~md\n![a](images/a.png)\n~~~\n',
                 '    <img src="images/a.png">\n    ![a](images/a.png)\n',
                 '`<img src="images/a.png">` and ``![a](images/a.png)``',
                 '<!-- <img src="images/a.png"> -->\n',
                 '<pre><code><img src="images/a.png"></code></pre>\n',
                 'Text <code><img src="images/a.png"></code> text',
                 'Text <code>![a](images/a.png)</code> text',
                 r'\![a](images/a.png)']
        for original in cases:
            with self.subTest(original=original):
                self.assertEqual(self.convert(original), original)

    def test_code_followed_by_real_image(self):
        original = '```\n![a](images/code.png)\n```\n\n![b](images/real.png)\n'
        self.assertEqual(self.convert(original), original.replace('images/real.png', BASE + 'images/real.png'))

    def test_prose_and_qiita_extensions_unchanged(self):
        original = '# 日本語\n\n:::note info\n本文。 $h=0$\n:::\n\n| A | B |\n|---|---|\n| 1 | 2 |\n'
        self.assertEqual(self.convert(original), original)

    def test_article_base_and_outside_root(self):
        self.assertEqual(article_base_url('www/content/llm/example/index.md', 'www/content', 'https://www.inoue-kobo.com'), BASE)
        with self.assertRaises(ValueError):
            article_base_url('/tmp/outside.md', 'www/content', 'https://www.inoue-kobo.com')


class SelectedArticleTests(unittest.TestCase):
    item_id = 'ee3d2dfcab6f30b69bed'

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.filename = root / 'index.md'
        self.filename.write_text('# Example\n\n<img src="images/a.png">\n')
        self.mapping = Mock(tag='LLM', dir=str(root))
        self.mapping_patch = patch('qiita_sync.CATEGORY_MAPPINGS', [self.mapping])
        self.mapping_patch.start()
        self.addCleanup(self.mapping_patch.stop)
        self.body_patch = patch.object(MyArticle, 'to_posting_format', return_value='converted body')
        self.body_patch.start()
        self.addCleanup(self.body_patch.stop)
        self.source = {'id': self.item_id, 'title': 'Example', 'body': 'old body',
                       'private': False, 'tags': [{'name': 'LLM', 'versions': []}]}
        self.response = Mock()
        self.response.json.return_value = self.source
        self.client = QiitaArticles('test-token', 1)

    @patch('qiita_sync.requests.patch')
    @patch('qiita_sync.requests.post')
    @patch('qiita_sync.requests.get')
    def test_only_gets_and_updates_explicit_item(self, get, post, patch_request):
        get.return_value = self.response
        patch_request.return_value.status_code = 200
        sync_selected_article(self.client, str(self.filename), self.item_id)
        get.assert_called_once_with(f'https://qiita.com/api/v2/items/{self.item_id}',
                                    headers=self.client.headers, timeout=30)
        post.assert_not_called()
        patch_request.assert_called_once_with(f'https://qiita.com/api/v2/items/{self.item_id}',
            json={'title': 'Example', 'body': 'converted body', 'private': False, 'tags': self.source['tags']},
            headers=self.client.headers)

    @patch('qiita_sync.requests.get')
    def test_unchanged_body_skips_patch(self, get):
        self.source['body'] = 'converted body'
        get.return_value = self.response
        with patch.object(self.client, 'update') as update:
            sync_selected_article(self.client, str(self.filename), self.item_id)
            update.assert_not_called()

    @patch('qiita_sync.requests.get')
    def test_wrong_item_or_private_fails_closed(self, get):
        get.return_value = self.response
        for key, value in [('title', 'Different article'), ('id', '0' * 20), ('private', True), ('private', None)]:
            previous = self.source[key]
            self.source[key] = value
            with self.subTest(key=key), patch.object(self.client, 'update') as update:
                with self.assertRaises(ValueError):
                    sync_selected_article(self.client, str(self.filename), self.item_id)
                update.assert_not_called()
            self.source[key] = previous

    @patch('qiita_sync.requests.get')
    def test_invalid_path_id_and_symlink_cannot_call_api(self, get):
        outside = Path(self.temp.name).parent / 'not-an-article.md'
        link = Path(self.temp.name) / 'link.md'
        link.symlink_to('/etc/hosts')
        for filename, item_id in [(self.filename, 'bad-id'), (outside, self.item_id), (link, self.item_id)]:
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                sync_selected_article(self.client, str(filename), item_id)
        get.assert_not_called()

    @patch.dict(os.environ, {'QIITA_TOKEN': 'test-token'})
    @patch.object(QiitaArticles, 'fetch')
    @patch('qiita_sync.sync_selected_article')
    def test_cli_selected_mode_does_not_fetch_all_items(self, selected, fetch):
        main(['--article', str(self.filename), '--item-id', self.item_id])
        selected.assert_called_once()
        fetch.assert_not_called()

    @patch.dict(os.environ, {'QIITA_TOKEN': 'test-token'})
    @patch('qiita_sync.CATEGORY_MAPPINGS', [])
    @patch.object(QiitaArticles, 'fetch')
    @patch('qiita_sync.sync_selected_article')
    def test_default_cli_keeps_normal_sync(self, selected, fetch):
        main([])
        fetch.assert_called_once()
        selected.assert_not_called()

    def test_cli_requires_both_selectors(self):
        for args in [['--article', 'index.md'], ['--item-id', self.item_id]]:
            with self.subTest(args=args), self.assertRaises(SystemExit) as error:
                main(args)
            self.assertEqual(error.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
