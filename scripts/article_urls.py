"""Resolve article image URLs without re-rendering Markdown or code examples."""
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
import re
from urllib.parse import urljoin, urlsplit

from markdown_it import MarkdownIt
from markdown_it.rules_inline import html_inline, image


# Match whole attributes so text such as alt='example src="..."' stays intact.
ATTRIBUTE = re.compile(r'''([^\s=<>/]+)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))''')


def article_base_url(filename, document_root, origin):
    relative = Path(filename).resolve().parent.relative_to(Path(document_root).resolve())
    path = relative.as_posix()
    return origin.rstrip('/') + '/' + (path + '/' if path != '.' else '')


def resolve_url(value, base):
    decoded = unescape(value).strip()
    if not decoded or decoded.startswith(('#', '?', '//')) or urlsplit(decoded).scheme:
        return value
    # Resolve the original spelling to preserve entities, escapes and query strings.
    return urljoin(base, value)


class HTMLURLs(HTMLParser):
    def __init__(self, source, base, allowed=None):
        super().__init__(convert_charrefs=False)
        self.base = base
        self.allowed = allowed
        self.offsets = [0]
        for match in re.finditer('\n', source):
            self.offsets.append(match.end())
        self.patches = []
        self.literal = []
        self.literal_start = None
        self.protected = []
        self.feed(source)
        self.close()
        if self.literal:
            self.protected.append((self.literal_start, len(source)))

    def position(self):
        line, column = self.getpos()
        return self.offsets[line - 1] + column

    def handle_starttag(self, tag, attrs):
        start = self.position()
        if self.allowed is not None and start not in self.allowed:
            return
        if tag in ('pre', 'code', 'script', 'style'):
            if not self.literal:
                self.literal_start = start
            self.literal.append(tag)
        if self.literal:
            return
        attribute = {'img': 'src', 'a': 'href'}.get(tag)
        if not attribute:
            return
        raw = self.get_starttag_text()
        for match in ATTRIBUTE.finditer(raw):
            if match.group(1).lower() != attribute:
                continue
            group = next(n for n in (2, 3, 4) if match.group(n) is not None)
            value = match.group(group)
            resolved = resolve_url(value, self.base)
            if resolved != value:
                self.patches.append((start + match.start(group), start + match.end(group), resolved))

    def handle_endtag(self, tag):
        if self.allowed is not None and self.position() not in self.allowed:
            return
        if self.literal and tag == self.literal[-1]:
            self.literal.pop()
            if not self.literal:
                self.protected.append((self.literal_start, self.position()))


def resolve_article_urls(source, base):
    """Only edit URL spans in parsed HTML and inline Markdown images.

    Block maps keep fenced/indented code untouched. Inline rules identify actual
    image and HTML syntax, so backticks and escaped examples remain unchanged.
    The original source (including Qiita extensions) is never re-rendered.
    """
    parser = MarkdownIt('commonmark')
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    tokens = parser.parse(source)
    patches = []
    seen = set()
    for token in tokens:
        if token.type not in ('inline', 'html_block') or not token.map:
            continue
        bounds = tuple(token.map)
        if bounds in seen:
            continue
        seen.add(bounds)
        begin, end = offsets[bounds[0]], offsets[bounds[1]]
        segment = source[begin:end]
        local = []
        allowed = None
        if token.type == 'inline':
            allowed = set()

            def capture_html(state, silent):
                start = state.pos
                result = html_inline(state, silent)
                if result and not silent and state.src == segment:
                    allowed.add(start)
                return result

            def capture_image(state, silent):
                start = state.pos
                result = image(state, silent)
                if not result or silent or state.src != segment:
                    return result
                finish = state.pos
                state.pos = start
                label_end = state.md.helpers.parseLinkLabel(state, start + 1, False)
                state.pos = finish
                pos = label_end + 1
                if pos >= len(segment) or segment[pos] != '(':
                    return result  # Reference-style images are left as authored.
                pos += 1
                while pos < len(segment) and segment[pos] in ' \t\n':
                    pos += 1
                destination = state.md.helpers.parseLinkDestination(segment, pos, finish)
                if destination.ok:
                    stop = destination.pos
                    if segment[pos] == '<':
                        pos, stop = pos + 1, stop - 1
                    value = segment[pos:stop]
                    resolved = resolve_url(value, base)
                    if value != resolved:
                        local.append((pos, stop, resolved))
                return result

            inline_parser = MarkdownIt('commonmark')
            inline_parser.inline.ruler.at('html_inline', capture_html)
            inline_parser.inline.ruler.at('image', capture_image)
            inline_parser.parseInline(segment)
        html = HTMLURLs(segment, base, allowed)
        local = [p for p in local if not any(a <= p[0] < b for a, b in html.protected)]
        local.extend(html.patches)
        patches.extend((begin + a, begin + b, replacement) for a, b, replacement in local)
    for start, end, replacement in sorted(patches, reverse=True):
        source = source[:start] + replacement + source[end:]
    return source
