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
