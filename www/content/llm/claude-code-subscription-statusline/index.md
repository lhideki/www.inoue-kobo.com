---
title: 'Claude Codeのステータスラインに契約アカウントと利用枠の残量を表示する'
date: '2026-09-10'
thumbnail: 'llm/claude-code-subscription-statusline/images/thumbnail.png'
tags:
    - 'Claude Code'
    - 'LLM'
---

# Claude Codeのステータスラインに契約アカウントと利用枠の残量を表示する

Claude Codeをサブスクリプションで利用する場合、5時間・週間の利用枠の残量を確認しながら作業を進めたい場面があります。また、仕事用と個人用など複数のアカウントを使い分けていると、どの契約アカウントでログインしているかも確認したくなります。ということで、契約アカウントのメールアドレスと利用枠の残量を、Claude Codeのステータスラインに表示する方法を紹介します。スクリプトを保存し、設定ファイルに追記することで、画面下部に以下のように表示できます。

```text
user@example.com │ Opus │ 5h:残76.5%(2h0m) │ 7d:残58.8%(72h0m)
```

メールアドレスと数値はサンプルです。表示内容は以下のとおりです。

| 表示 | 意味 |
| --- | --- |
| `user@example.com` | Claude Codeのログイン情報に保存された契約アカウントのメールアドレス |
| `Opus` | そのセッションで利用中のモデル |
| `5h:残76.5%` | 5時間の利用枠の残り割合。使用率は23.5% |
| `7d:残58.8%` | 週間の利用枠の残り割合。使用率は41.2% |
| `(2h0m)` / `(72h0m)` | それぞれの利用枠がリセットされるまでの時間。`h`は時間、`m`は分 |

利用枠は、使用率を100%から引いた「残り割合」で表示します。アカウントのメールアドレスも並ぶため、どのアカウントで利用しているかを確認しやすくなります。

## 前提条件

この記事は、次の環境を対象にしています。

* ターミナルでClaude Codeを利用していること。本文のコマンド例はmacOSを使用しています。
* Claude Pro、Max、Teamのいずれかの契約アカウントで、Claude Codeにログイン済みであること。
* Node.js 24系がインストールされ、ターミナルで`node`を実行できること。
* Claude Codeの設定を標準の場所(`~/.claude/`と`~/.claude.json`)に保存していること。環境変数`CLAUDE_CONFIG_DIR`による保存先の変更は、この手順の対象外です。

Node.jsが未導入の場合は、[公式ダウンロードページ](https://nodejs.org/en/download)から、利用するOSに合った24系のインストール方法を選択してください。インストール後はターミナルを開き直します。追加のnpmパッケージは不要です。

以下のコマンドで、バージョンを確認できます。

```bash
node --version
claude --version
```

今回の環境はNode.js v24.18.1、Claude Code 2.1.236です。掲載スクリプトは、サンプルの入力データを使って表示を確認しています。

## スクリプトを保存する

まず、スクリプトを保存します。ターミナルで以下のコマンドを実行してください。

```bash
mkdir -p ~/.claude
nano ~/.claude/statusline-subscription.mjs
```

以下のコードを全文貼り付けてください。

同じ内容のファイルを[statusline-subscription.mjs](./statusline-subscription.mjs)から保存することもできます。保存先とファイル名は、上記と同じにしてください。

```javascript
#!/usr/bin/env node

// Claude Code statusline:  <account> │ <model> │ 5h:残…% │ 7d:残…% [│ Opus7d:残…%]
//
// レート制限は、まず Claude Code が stdin(ステータスラインへの入力)で渡してくる
// `rate_limits` を使う(使用率の取得に追加のネットワーク通信・キーチェーン参照は不要)。
// stdin に無い場合(セッション開始直後など)のみ OAuth usage API にフォールバックする。

import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

const SEP = " │ ";
const CONFIG_DIR = process.env.CLAUDE_CONFIG_DIR || homedir();
const CLAUDE_JSON = join(CONFIG_DIR, ".claude.json");
const CACHE_PATH = join(CONFIG_DIR, "cache", "claude-code-usage-statusline.json");
const CACHE_TTL_MS = 60_000;

// ---------------------------------------------------------------------------
// stdin(ステータスラインへの入力)
// ---------------------------------------------------------------------------

function readStdinJson() {
  try {
    const input = readFileSync(0, "utf8");
    return input ? JSON.parse(input) : {};
  } catch {
    return {};
  }
}

// ---------------------------------------------------------------------------
// ログイン中の Anthropic アカウント(~/.claude.json の oauthAccount)
// ---------------------------------------------------------------------------

function getAccountLabel() {
  try {
    const account = JSON.parse(readFileSync(CLAUDE_JSON, "utf8"))?.oauthAccount;
    if (!account) return "";
    return account.emailAddress || account.displayName || account.fullName || "";
  } catch {
    return "";
  }
}

// ---------------------------------------------------------------------------
// 表示整形
// ---------------------------------------------------------------------------

function formatResetIn(resetsAt) {
  if (resetsAt == null) return "";

  // stdin は epoch 秒(数値)、usage API は ISO 文字列で返してくる。
  const ms =
    typeof resetsAt === "number"
      ? resetsAt < 1e12
        ? resetsAt * 1000
        : resetsAt
      : new Date(resetsAt).getTime();

  const diff_ms = ms - Date.now();
  if (!Number.isFinite(diff_ms) || diff_ms <= 0) return "reset soon";

  const total_minutes = Math.ceil(diff_ms / 60_000);
  const hours = Math.floor(total_minutes / 60);
  const minutes = total_minutes % 60;

  return hours > 0 ? `${hours}h${minutes}m` : `${minutes}m`;
}

// used_percentage は 0〜100。残り割合とリセットまでの時間を返す。
function formatUsage(label, used_percentage, resets_at) {
  if (typeof used_percentage !== "number") return `${label}:n/a`;

  const used = Math.max(0, Math.min(100, used_percentage));
  const remain = Math.max(0, 100 - used);
  const reset_in = formatResetIn(resets_at);

  return `${label}:残${remain.toFixed(1)}%${reset_in ? `(${reset_in})` : ""}`;
}

// ---------------------------------------------------------------------------
// レート制限 — stdin 優先
// ---------------------------------------------------------------------------

function usageFromStdin(status) {
  const rl = status?.rate_limits;
  if (!rl) return null;

  const parts = [];
  if (rl.five_hour) {
    parts.push(formatUsage("5h", rl.five_hour.used_percentage, rl.five_hour.resets_at));
  }
  if (rl.seven_day) {
    parts.push(formatUsage("7d", rl.seven_day.used_percentage, rl.seven_day.resets_at));
  }
  return parts.length ? parts : null;
}

// ---------------------------------------------------------------------------
// レート制限 — OAuth usage API へのフォールバック
// ---------------------------------------------------------------------------

function getOauthToken() {
  try {
    const raw = execFileSync(
      "security",
      ["find-generic-password", "-s", "Claude Code-credentials", "-w"],
      { encoding: "utf8" }
    ).trim();
    return JSON.parse(raw)?.claudeAiOauth?.accessToken ?? null;
  } catch {
    return null;
  }
}

function readCache() {
  try {
    if (!existsSync(CACHE_PATH)) return null;
    const cached = JSON.parse(readFileSync(CACHE_PATH, "utf8"));
    if (Date.now() - cached.cached_at > CACHE_TTL_MS) return null;
    return cached.data;
  } catch {
    return null;
  }
}

function writeCache(data) {
  try {
    mkdirSync(dirname(CACHE_PATH), { recursive: true });
    writeFileSync(CACHE_PATH, JSON.stringify({ cached_at: Date.now(), data }), "utf8");
  } catch {
    // ignore
  }
}

async function fetchUsage(token) {
  const cached = readCache();
  if (cached) return cached;

  const response = await fetch("https://api.anthropic.com/api/oauth/usage", {
    method: "GET",
    headers: {
      Accept: "application/json, text/plain, */*",
      "Content-Type": "application/json",
      "User-Agent": "claude-code/2.0",
      Authorization: `Bearer ${token}`,
      "anthropic-beta": "oauth-2025-04-20",
    },
  });

  if (!response.ok) throw new Error(`Usage API failed: ${response.status}`);

  const data = await response.json();
  writeCache(data);
  return data;
}

// usage API の *.utilization は 0〜100。
async function usageFromApi() {
  const token = getOauthToken();
  if (!token) return null;

  const usage = await fetchUsage(token);
  const parts = [];
  const push = (label, bucket) => {
    if (bucket && typeof bucket.utilization === "number") {
      parts.push(formatUsage(label, bucket.utilization, bucket.resets_at));
    }
  };

  push("5h", usage.five_hour);
  push("7d", usage.seven_day);
  push("Opus7d", usage.seven_day_opus);

  return parts.length ? parts : null;
}

// ---------------------------------------------------------------------------

async function main() {
  const status = readStdinJson();
  const model = status?.model?.display_name ?? status?.model?.id ?? "";
  const account = getAccountLabel();

  const head = [account, model].filter(Boolean);

  let usageParts = usageFromStdin(status);
  if (!usageParts) {
    try {
      usageParts = await usageFromApi();
    } catch {
      usageParts = ["usage:error"];
    }
  }
  if (!usageParts) usageParts = ["usage:n/a"];

  console.log([...head, ...usageParts].join(SEP));
}

main();

```

## settings.jsonに設定を追加する

設定ファイルは`~/.claude/settings.json`です。以下の内容を記載してください。

すでに設定がある場合は、既存の項目を残して、最も外側の`{ }`の中に`statusLine`を追加してください。直前の項目との間にはカンマが必要です。すでに`statusLine`がある場合は、その項目だけを変更します。

```json
{
  "statusLine": {
    "type": "command",
    "command": "node \"$HOME/.claude/statusline-subscription.mjs\""
  }
}
```

## 表示を確認する

Claude Codeを起動し、通常の会話を1回行って、応答後に画面下部を確認してください。

```bash
claude
```

表示されるアカウントをClaude Codeの`/status`で確認し、利用枠は`/usage`と見比べます。たとえば`/usage`で5時間枠の使用率が23.5%であれば、ステータスラインは`5h:残76.5%`となります。更新タイミングによって数値に差が出る場合があります。

## 表示されない場合の確認事項

| 状況 | 確認すること |
| --- | --- |
| ステータスライン全体が表示されない | `settings.json`の構文、`statusLine`の綴り、スクリプトの保存先を確認します。 |
| `node: command not found`と表示される | ターミナルで`command -v node`を実行し、設定JSONの`command`内の`node`を、表示された実行ファイルの絶対パスに置き換えます。 |
| アカウント部分だけ表示されない | `/status`でログイン状態を確認します。`~/.claude.json`からアカウント情報を取得できない場合、この部分は省略されます。 |
| 想定と違うアカウントが表示される | `/status`でログインアカウントを確認し、必要なら`/login`で利用したいアカウントにログインし直します。複数セッションを起動している場合は、切り替え後に対象セッションを起動し直してください。 |
| `usage:n/a`と表示される | まだ利用枠の情報を取得できていません。Pro/Max/Teamのアカウントでログインしていることを確認し、通常の会話を1回行ってから再確認します。 |
| `usage:error`と表示される | 補完用の通信などでエラーが発生しています。ネットワークとログイン状態を確認してください。Claude Codeから利用枠が渡されるようになれば、そちらを優先して表示します。 |
| `5h:n/a`や片方の利用枠だけが表示される | Claude Codeから渡された情報が一部不足しています。次の応答後に再確認します。 |
| リセットまでの時間が動かない | この設定では秒刻みの更新を行いません。Claude Codeがスクリプトを再実行したタイミングで再計算します。 |

## 補足

Linuxでは、Node.jsとテキストエディターが利用できれば、本文と同じ保存先と設定JSONを使用できます。Windowsでは、ユーザーフォルダー内の`.claude`にスクリプトと`settings.json`を保存します。たとえばユーザーフォルダーが`C:/Users/username`の場合、設定は以下のようになります。`username`は自分のユーザーフォルダー名に置き換えてください。

```json
{
  "statusLine": {
    "type": "command",
    "command": "node \"C:/Users/username/.claude/statusline-subscription.mjs\""
  }
}
```

Windowsの`command`に記載するパスには、バックスラッシュではなく`/`を使用します。ファイルの作成・編集には任意のテキストエディターを使用し、UTF-8で保存してください。[公式ドキュメント: Windows configuration](https://code.claude.com/docs/en/statusline#windows-configuration)

アカウント情報の読み込みもNode.jsでホームディレクトリを求めるため、macOS以外でも、ホームディレクトリの`.claude.json`に同じ形式のログイン情報が保存されていれば表示できます。Windows・Linuxの実機での動作は、本記事では未検証です。

利用枠は、Claude Codeが標準入力で渡す`rate_limits`から取得します。そこから5時間・週間の表示項目を作れない場合は、macOSのキーチェーンに保存されたClaude Codeの認証情報を使って、OAuthのusageエンドポイントから補完します。補完した使用量データは60秒間キャッシュするため、`/usage`の表示と一時的に差が出る場合があります。

WindowsやLinuxでは、掲載スクリプトのキーチェーンによる補完は利用できません。そのため、セッション開始直後などに`usage:n/a`となる場合がありますが、通常の会話を行い、Claude Codeから`rate_limits`が渡されるようになれば表示できます。

契約アカウントの表示は、ローカルに保存されたログイン情報に基づきます。契約プラン名や請求先組織を判定するものではありません。別セッションでアカウントを切り替えた場合は、`/status`でも対象セッションのログイン状態を確認してください。

表示を元に戻す場合は、`settings.json`から今回追加した`statusLine`項目を削除するか、以前の`statusLine`の内容に戻してください。

## 参考文献

* [Customize your status line - Claude Code Docs](https://code.claude.com/docs/en/statusline)
* [Claude Code CHANGELOG](https://github.com/anthropics/claude-code/blob/main/CHANGELOG.md)
