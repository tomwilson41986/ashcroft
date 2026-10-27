"""The Professor's cards (Pillow): a tip card (1080 x 1080) and the day's card (1080 x 1350), the compliance footer
on both. The fonts are Pillow's bundled DejaVu where the OFL fonts in professor/assets/fonts are absent, so a card
always renders; the brand fonts can be dropped in later without a code change.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ASSETS = Path(__file__).resolve().parent / "assets" / "fonts"
INK, PAPER, ACCENT, MUTED = (24, 28, 36), (248, 246, 240), (184, 134, 11), (110, 110, 110)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for name in (("Bold" if bold else "Regular"),):
        for p in (ASSETS / f"Professor-{name}.ttf", ASSETS / f"{name}.ttf"):
            if p.exists():
                return ImageFont.truetype(str(p), size)
    try:
        import PIL
        dv = Path(PIL.__file__).resolve().parent / "fonts" / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
        if dv.exists():
            return ImageFont.truetype(str(dv), size)
    except Exception:  # noqa: BLE001
        pass
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
              else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, width: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        trial = (line + " " + word).strip()
        if draw.textlength(trial, font=font) <= width:
            line = trial
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def _footer(draw: ImageDraw.ImageDraw, W: int, H: int, cfg: dict) -> None:
    f = _font(26)
    y = H - 150
    for line in (cfg["compliance"]["position"], cfg["compliance"]["age"]):
        for l in _wrap(draw, line, f, W - 120):
            draw.text((60, y), l, font=f, fill=MUTED)
            y += 32


def tip_card(sheet: dict, cfg: dict, size: tuple[int, int] = (1080, 1080)) -> Image.Image:
    W, H = size
    img = Image.new("RGB", size, PAPER)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 14], fill=ACCENT)
    d.text((60, 50), "THE PROFESSOR", font=_font(34, True), fill=ACCENT)
    d.text((60, 100), f"{sheet['off_uk']} {sheet['venue']}  ·  {sheet['day']}", font=_font(34), fill=MUTED)
    y = 180
    for l in _wrap(d, sheet["runner"].upper(), _font(84, True), W - 120):
        d.text((60, y), l, font=_font(84, True), fill=INK)
        y += 96
    y += 20
    d.text((60, y), sheet["advised_fraction"], font=_font(150, True), fill=ACCENT)
    y += 170
    line = (f"{sheet['n_bookmakers']} bookmakers: {sheet['bookmakers']}" if sheet["price_source"] == "books"
            else f"Betfair exchange, GBP{sheet['exchange_size']} at the price")
    for l in _wrap(d, line, _font(34), W - 120):
        d.text((60, y), l, font=_font(34), fill=INK)
        y += 42
    y += 24
    for l in _wrap(d, f"Model price {sheet['model_price']}, expected SP {sheet['expected_bsp']}: "
                      f"the price is {sheet['overlay_pct']}% over the expected close.", _font(32), W - 120):
        d.text((60, y), l, font=_font(32), fill=INK)
        y += 40
    _footer(d, W, H, cfg)
    return img


def day_card(sheets: list[dict], cfg: dict, title: str, size: tuple[int, int] = (1080, 1350)) -> Image.Image:
    W, H = size
    img = Image.new("RGB", size, PAPER)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 14], fill=ACCENT)
    d.text((60, 50), "THE PROFESSOR", font=_font(34, True), fill=ACCENT)
    d.text((60, 100), title, font=_font(40, True), fill=INK)
    y = 190
    for s in sheets:
        d.text((60, y), f"{s['off_uk']} {s['venue']}", font=_font(30), fill=MUTED)
        y += 36
        d.text((60, y), s["runner"], font=_font(52, True), fill=INK)
        d.text((W - 60 - d.textlength(s["advised_fraction"], font=_font(52, True)), y), s["advised_fraction"],
               font=_font(52, True), fill=ACCENT)
        y += 62
        sub = (f"{s['n_bookmakers']} bookmakers" if s["price_source"] == "books" else "exchange") + \
              f" · expected SP {s['expected_bsp']}"
        d.text((60, y), sub, font=_font(28), fill=MUTED)
        y += 60
        if y > H - 260:
            break
    _footer(d, W, H, cfg)
    return img


def save(img: Image.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "PNG", optimize=True)
    return path
