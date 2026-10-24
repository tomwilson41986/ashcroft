"""Telegram's Bot API, the little of it the Professor uses: a message or a photo to a channel, a single-use invite
link, and the removal of a lapsed member (ban, then unban so they can rejoin with a new link). Plain requests; the
token from TELEGRAM_BOT_TOKEN. Every call returns Telegram's reply as a dict, or raises with its description."""
from __future__ import annotations

import os

import requests

API = "https://api.telegram.org/bot{token}/{method}"
TIMEOUT = 30


def _token() -> str:
    t = os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not t:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
    return t


def call(method: str, **params) -> dict:
    files = params.pop("_files", None)
    r = requests.post(API.format(token=_token(), method=method), data=params, files=files, timeout=TIMEOUT)
    try:
        body = r.json()
    except ValueError:
        raise RuntimeError(f"telegram {method}: HTTP {r.status_code} {r.text[:200]}")
    if not body.get("ok"):
        raise RuntimeError(f"telegram {method}: {body.get('description', body)}")
    return body["result"]


def send_message(chat_id: str, text: str, parse_mode: str = "HTML", silent: bool = False) -> dict:
    return call("sendMessage", chat_id=chat_id, text=text[:4096], parse_mode=parse_mode, disable_web_page_preview=True,
                disable_notification=silent)


def send_photo(chat_id: str, png: bytes, caption: str, parse_mode: str = "HTML") -> dict:
    return call("sendPhoto", chat_id=chat_id, caption=caption[:1024], parse_mode=parse_mode,
                _files={"photo": ("card.png", png, "image/png")})


def invite_link(chat_id: str, expire_seconds: int = 3600, name: str = "") -> str:
    import time
    r = call("createChatInviteLink", chat_id=chat_id, member_limit=1, expire_date=int(time.time()) + expire_seconds,
             name=name[:32])
    return r["invite_link"]


def kick(chat_id: str, user_id: int) -> None:
    call("banChatMember", chat_id=chat_id, user_id=user_id)
    call("unbanChatMember", chat_id=chat_id, user_id=user_id, only_if_banned=True)


def me() -> dict:
    return call("getMe")
