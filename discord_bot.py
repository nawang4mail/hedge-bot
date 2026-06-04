"""
Hedge Bot — Discord LLM Chat Bot
=================================
Runs as a standalone process alongside the FastAPI server.

Features
--------
  !insider <question>   — ask the LLM about the insider currently loaded
                          (uses the most recent watchlist entries as context)
  !watchlist            — list your currently watched insiders
  !alert                — manually trigger a watchlist scan right now
  !help                 — show commands

Requirements
------------
  pip install discord.py httpx

Setup
-----
1. Go to https://discord.com/developers/applications
2. Create a new application → Bot → copy the TOKEN
3. Enable "Message Content Intent" under Bot → Privileged Gateway Intents
4. Invite with: https://discord.com/api/oauth2/authorize?client_id=YOUR_APP_ID&permissions=2048&scope=bot
5. Add the token to your Discord settings in the UI or via:
       POST http://localhost:8000/discord/settings
       {"webhook_url": "...", "bot_token": "YOUR_TOKEN_HERE"}
6. Run:  python discord_bot.py
"""
from __future__ import annotations

import asyncio
import os
import json
import sys
from pathlib import Path

import httpx

# ── Config ─────────────────────────────────────────────────────────────────────

API_BASE   = os.getenv("HEDGE_BOT_API", "http://localhost:8000")
CONN_FILE  = Path(__file__).parent / "connections.json"
BOT_PREFIX = "!"
MAX_TRADES = 100   # rows sent to LLM per query


# ── Credential loader ──────────────────────────────────────────────────────────

def _get_token() -> str:
    # 1. Environment variable (preferred for CI/prod)
    if os.getenv("DISCORD_BOT_TOKEN"):
        return os.environ["DISCORD_BOT_TOKEN"]
    # 2. connections.json (saved via UI)
    if CONN_FILE.exists():
        data = json.loads(CONN_FILE.read_text())
        token = data.get("discord", {}).get("bot_token", "")
        if token:
            return token
    return ""


# ── API helpers ────────────────────────────────────────────────────────────────

async def _api_get(path: str) -> dict | list:
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get(f"{API_BASE}{path}")
        r.raise_for_status()
        return r.json()


async def _api_post(path: str, body: dict) -> dict:
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(f"{API_BASE}{path}", json=body)
        r.raise_for_status()
        return r.json()


async def _get_watchlist() -> list[dict]:
    try:
        return await _api_get("/insider/watchlist")  # type: ignore[return-value]
    except Exception:
        return []


async def _get_trades_for_cik(cik: str) -> list[dict]:
    try:
        data = await _api_get(f"/insider/person/{cik}")
        return data.get("trades", [])[:MAX_TRADES]  # type: ignore[union-attr]
    except Exception:
        return []


async def _ask_llm(question: str, rows: list[dict], context: str) -> str:
    try:
        result = await _api_post("/insider/chat", {
            "question": question,
            "rows":     rows,
            "context":  context,
        })
        return result.get("answer", "No response from LLM.")
    except httpx.ConnectError:
        return "❌ Cannot reach the API server at `{API_BASE}`. Is it running?"
    except Exception as exc:
        return f"❌ Error: {exc}"


# ── Discord bot ────────────────────────────────────────────────────────────────

def run_bot(token: str):
    try:
        import discord
    except ImportError:
        print("discord.py not installed. Run: pip install discord.py")
        sys.exit(1)

    intents = discord.Intents.default()
    intents.message_content = True
    client  = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        print(f"[discord_bot] Logged in as {client.user} (ID: {client.user.id})")
        print(f"[discord_bot] API target: {API_BASE}")
        print("[discord_bot] Ready. Commands: !insider, !watchlist, !alert, !help")

    @client.event
    async def on_message(message: discord.Message):
        if message.author.bot:
            return
        content = message.content.strip()
        if not content.startswith(BOT_PREFIX):
            return

        parts   = content[len(BOT_PREFIX):].split(maxsplit=1)
        command = parts[0].lower() if parts else ""
        arg     = parts[1] if len(parts) > 1 else ""

        # ── !help ──────────────────────────────────────────────────────────────
        if command == "help":
            await message.channel.send(
                "**Hedge Bot — Insider Commands**\n"
                "`!insider <question>` — ask the LLM about your top watched insider\n"
                "`!watchlist` — show your watched insiders\n"
                "`!alert` — run a watchlist scan and send any new trade alerts\n"
                "`!help` — show this message"
            )

        # ── !watchlist ─────────────────────────────────────────────────────────
        elif command == "watchlist":
            async with message.channel.typing():
                wl = await _get_watchlist()
            if not wl:
                await message.channel.send("Your watchlist is empty. Add insiders from the Hedge Bot UI.")
                return
            lines = [f"**Watched Insiders ({len(wl)})**"]
            for i, w in enumerate(wl, 1):
                lines.append(f"{i}. {w.get('name_clean') or w.get('cik')} `CIK: {w.get('cik')}`")
            await message.channel.send("\n".join(lines))

        # ── !insider <question> ───────────────────────────────────────────────
        elif command == "insider":
            if not arg:
                await message.channel.send(
                    "Usage: `!insider <your question>`\n"
                    "Example: `!insider Has Tim Cook been a net buyer recently?`"
                )
                return

            async with message.channel.typing():
                # Use first watchlist entry as context, or ask user to specify
                wl = await _get_watchlist()
                if not wl:
                    await message.channel.send(
                        "Your watchlist is empty — add an insider in the UI first, "
                        "or the question will be answered with no trade context."
                    )
                    rows    = []
                    context = "No insider data loaded"
                else:
                    target  = wl[0]   # most recently watched
                    name    = target.get("name_clean") or target.get("cik", "")
                    rows    = await _get_trades_for_cik(target["cik"])
                    context = f"{name} Form 4 history ({len(rows)} trades)"
                    await message.channel.send(
                        f"🔍 Querying LLM about **{name}** ({len(rows)} trades)…"
                    )

                answer = await _ask_llm(arg, rows, context)

            # Discord has a 2000-char message limit — split if needed
            if len(answer) <= 1900:
                await message.channel.send(f"**Answer:**\n{answer}")
            else:
                chunks = [answer[i:i+1900] for i in range(0, len(answer), 1900)]
                for i, chunk in enumerate(chunks):
                    prefix = "**Answer (cont.):**\n" if i > 0 else "**Answer:**\n"
                    await message.channel.send(prefix + chunk)

        # ── !alert ─────────────────────────────────────────────────────────────
        elif command == "alert":
            async with message.channel.typing():
                try:
                    await _api_post("/discord/alert/watchlist", {})
                    await message.channel.send(
                        "✅ Watchlist scan triggered. Any new open-market trades "
                        "will appear as Discord embeds in the configured alert channel."
                    )
                except Exception as exc:
                    await message.channel.send(f"❌ Scan failed: {exc}")

        else:
            await message.channel.send(
                f"Unknown command `!{command}`. Type `!help` for available commands."
            )

    client.run(token)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    token = _get_token()
    if not token:
        print(
            "No bot token found.\n"
            "Either:\n"
            "  1. Set DISCORD_BOT_TOKEN env var\n"
            "  2. Save it via the UI: POST /discord/settings {\"webhook_url\":\"...\",\"bot_token\":\"...\"}\n"
        )
        sys.exit(1)

    print(f"[discord_bot] Starting… API = {API_BASE}")
    run_bot(token)
