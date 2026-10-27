"""Where the tips go, and the record that they went: the free Telegram channel (Phase 1; X and Instagram follow in
Phase 2, the members channel in Phase 3), each post keyed and logged in the day's posts.json so a run that starts
again never posts twice, and a channel's failure never stops the others.

Nothing is posted unless PROFESSOR_LIVE=yes and the run is not a dry run; a dry run writes every post's text and card
to out/professor/<day>/ instead, which is what the owner reads first.
"""
from __future__ import annotations

import io
import logging
import os
from datetime import datetime, timezone

from professor import store
from professor.render import day_card, save, tip_card
from professor.selection import Tip
from professor.writeup import compose

log = logging.getLogger(__name__)


def live() -> bool:
    return os.getenv("PROFESSOR_LIVE", "").lower() == "yes" and not store.dry()


def channels(cfg: dict, test: bool = False) -> dict:
    """The channel ids from the environment; the test channel stands in for every one with --test-channels."""
    if test:
        t = os.getenv("TELEGRAM_TEST_CHANNEL_ID", "")
        return {"telegram_free": t, "telegram_members": t}
    return {"telegram_free": os.getenv("TELEGRAM_FREE_CHANNEL_ID", ""),
            "telegram_members": os.getenv("TELEGRAM_MEMBERS_CHANNEL_ID", "")}


def _png(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def post_tip(tip: Tip, cfg: dict, chans: dict, paid_open: bool = False) -> dict:
    """One tip to its channels: the members channel first when paid is open (every tip), then the free channel
    (the free tips). A withheld tip is written to the record and posted nowhere."""
    day = tip.cand.day
    text = compose(tip, cfg)
    sheet = text["sheet"]
    out = store.out_dir(day)
    card = tip_card(sheet, cfg)
    png_path = save(card, out / "cards" / f"{tip.slug}.png")
    store.put_bytes(f"professor/cards/{day}/{tip.slug}.png", png_path.read_bytes(), "image/png")
    (out / "writeups").mkdir(exist_ok=True)
    (out / "writeups" / f"{tip.slug}.md").write_text(f"# {text['headline']}\n\n{text['body']}\n\n---\nX: {text['x_text']}\n")
    results = {"slug": tip.slug, "withheld": tip.withheld, "posted": []}
    if tip.withheld:
        store.record_post(day, f"withheld:{tip.slug}", {"tier": tip.tier, "rank": tip.rank})
        return results
    targets = []
    if paid_open and chans.get("telegram_members"):
        targets.append(("telegram_members", "members"))
    if tip.tier == "free" and chans.get("telegram_free"):
        targets.append(("telegram_free", "free"))
    posts = store.post_log(day)
    for chan, kind in targets:
        key = f"telegram:{kind}:{tip.slug}"
        if key in posts:
            results["posted"].append({"key": key, "skipped": "already posted"})
            continue
        caption = text["telegram"]
        if not live():
            results["posted"].append({"key": key, "dry_run": True, "chat_id": chans[chan], "caption_chars": len(caption)})
            continue
        try:
            from professor import telegram
            reply = telegram.send_photo(chans[chan], _png(card), caption[:1024])
            if len(caption) > 1024:
                telegram.send_message(chans[chan], text["body"][: 4096], silent=True)
            store.record_post(day, key, {"message_id": reply.get("message_id"), "chat_id": chans[chan]})
            tip.published_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
            results["posted"].append({"key": key, "message_id": reply.get("message_id")})
        except Exception as e:  # noqa: BLE001
            log.warning("post %s: %s", key, str(e)[:160])
            results["posted"].append({"key": key, "error": str(e)[:160]})
    return results


def post_day_card(tips: list[Tip], cfg: dict, chans: dict, title: str, kind: str = "free") -> dict:
    """The day's card (the free tips at 10:00 UK): one photo to the free channel, keyed by day and kind."""
    shown = [t for t in tips if not t.withheld and (kind != "free" or t.tier == "free")]
    if not shown:
        return {"skipped": "no tips to show"}
    day = shown[0].cand.day
    sheets = [compose(t, cfg)["sheet"] for t in shown]
    img = day_card(sheets, cfg, title)
    path = save(img, store.out_dir(day) / "cards" / f"day_{kind}.png")
    store.put_bytes(f"professor/cards/{day}/day_{kind}.png", path.read_bytes(), "image/png")
    key = f"telegram:{kind}:day-card"
    if key in store.post_log(day):
        return {"key": key, "skipped": "already posted"}
    caption = f"<b>{title}</b>\n" + "\n".join(f"{s['off_uk']} {s['venue']}: {s['runner']} {s['advised_fraction']}" for s in sheets) \
              + f"\n\n{cfg['compliance']['position']}\n{cfg['compliance']['age']}"
    chan = chans.get("telegram_free", "")
    if not live() or not chan:
        return {"key": key, "dry_run": True, "caption": caption}
    try:
        from professor import telegram
        reply = telegram.send_photo(chan, _png(img), caption[:1024])
        store.record_post(day, key, {"message_id": reply.get("message_id"), "chat_id": chan})
        return {"key": key, "message_id": reply.get("message_id")}
    except Exception as e:  # noqa: BLE001
        return {"key": key, "error": str(e)[:160]}
