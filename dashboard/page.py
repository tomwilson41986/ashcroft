"""The dashboard page: one self-contained HTML file, the figures embedded and refreshed from ``traders.json`` beside it.

Opened from where it is hosted, the page reads ``traders.json`` again every 60 seconds and updates the figures in
place (a cell that changes flashes once). Opened as a file, it shows the figures it was built with and says so.
"""
from __future__ import annotations

import json

TEMPLATE = r"""<!doctype html>
<html lang="en-GB">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Ashcroft Traders</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&family=Geist+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
  --paper:#0B0C0E; --paper-2:#15171B; --ink:#E6E8EB; --ink-2:#9AA0A8; --ink-3:#5E646C;
  --rule:#23272D; --rule-strong:#E6E8EB; --accent:#F2B33D; --accent-ink:#F5C566; --accent-wash:#2A2213;
  --pos:#3FCF8E; --neg:#FF7066; --cat-mute:#33373D;
  --horse:#6AA2E8; --greyhound:#DE6E28; --football:#7BC154; --cricket:#CFB8EF;
  --font-ui:"Geist",ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
  --font-mono:"Geist Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  --s-1:4px; --s-2:8px; --s-3:12px; --s-4:16px; --s-5:24px; --s-6:32px; --s-7:48px; --s-8:64px;
  color-scheme: dark;
}
:root[data-theme="light"] {
  --paper:#FFFFFF; --paper-2:#F3F4F6; --ink:#0D1014; --ink-2:#4D5560; --ink-3:#8A919B;
  --rule:#E2E5E9; --rule-strong:#0D1014; --accent:#B87800; --accent-ink:#8F5D00; --accent-wash:#FBF3E3;
  --pos:#12784C; --neg:#B4372B; --cat-mute:#D3D7DC;
  --horse:#3F7FCF; --greyhound:#C4531F; --football:#306406; --cricket:#7C298E;
  color-scheme: light;
}
@media (prefers-color-scheme: light) {
  :root:not([data-theme="dark"]) {
    --paper:#FFFFFF; --paper-2:#F3F4F6; --ink:#0D1014; --ink-2:#4D5560; --ink-3:#8A919B;
    --rule:#E2E5E9; --rule-strong:#0D1014; --accent:#B87800; --accent-ink:#8F5D00; --accent-wash:#FBF3E3;
    --pos:#12784C; --neg:#B4372B; --cat-mute:#D3D7DC;
    --horse:#3F7FCF; --greyhound:#C4531F; --football:#306406; --cricket:#7C298E;
    color-scheme: light;
  }
}
* { box-sizing: border-box; }
html, body { margin: 0; background: var(--paper); color: var(--ink); }
body { font: 400 14px/1.5 var(--font-ui); -webkit-font-smoothing: antialiased; padding-bottom: 40px; }
::selection { background: var(--accent); color: #0B0C0E; }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.num, .mono { font-family: var(--font-mono); font-variant-numeric: tabular-nums lining-nums; }
.wrap { max-width: 1280px; margin: 0 auto; padding: 0 16px; }

header.top { display: flex; align-items: baseline; justify-content: space-between; gap: var(--s-4);
  padding: var(--s-5) 0 var(--s-4); border-bottom: 1px solid var(--rule); }
.brand { font: 700 15px/1 var(--font-ui); letter-spacing: -0.01em; }
.brand sup { font-size: 0.65em; }
.brand span { color: var(--ink-2); font-weight: 500; margin-left: var(--s-2); }
.clock { color: var(--ink-2); font-size: 12px; text-align: right; }
.clock b { color: var(--ink); font-weight: 500; }
.theme { all: unset; cursor: pointer; color: var(--ink-2); font-size: 12px; padding: 12px 0 12px var(--s-3); }
.theme:hover { color: var(--ink); }

nav.sports { display: flex; gap: 0; border-bottom: 1px solid var(--rule); overflow-x: auto; scrollbar-width: none; }
nav.sports button { all: unset; cursor: pointer; padding: 14px var(--s-4) 12px; min-height: 44px; box-sizing: border-box;
  color: var(--ink-2); font-weight: 500; white-space: nowrap; border-bottom: 2px solid transparent; margin-bottom: -1px; }
nav.sports button:hover { color: var(--ink); }
nav.sports button[aria-selected="true"] { color: var(--ink); border-bottom-color: var(--accent); }
nav.sports button .tab-pnl { font-family: var(--font-mono); font-size: 12px; margin-left: var(--s-2); color: var(--ink-3); }
nav.sports button:focus-visible { outline-offset: -2px; }

section.sport-head { padding: var(--s-6) 0 var(--s-4); display: grid; grid-template-columns: 1fr auto; gap: var(--s-4);
  align-items: end; }
.sport-head h1 { margin: 0; font: 600 28px/1.1 var(--font-ui); letter-spacing: -0.02em; }
.sport-head p { margin: var(--s-1) 0 0; color: var(--ink-2); max-width: 70ch; }
dl.stats { display: flex; flex-wrap: wrap; gap: var(--s-4) var(--s-6); margin: 0; padding: var(--s-4) 0;
  border-block: 1px solid var(--rule); }
.stat dt { font: 600 11px/1 var(--font-ui); letter-spacing: 0.11em; text-transform: uppercase; color: var(--ink-2); }
.stat dd { margin: var(--s-2) 0 0; font: 500 22px/1 var(--font-mono); font-variant-numeric: tabular-nums; }
.stat dd small { font: 400 12px/1 var(--font-ui); color: var(--ink-2); margin-left: 4px; }
.stats-label { font-size: 12px; color: var(--ink-3); margin: var(--s-2) 0 0; }

/* the trader seats: a ruled ledger, one row a trader */
.board { margin-top: var(--s-6); }
.board-head, .seat { display: grid; align-items: center; gap: var(--s-3);
  grid-template-columns: minmax(250px, 2.2fr) repeat(6, minmax(92px, 1fr)) 132px; }
.board-head { font: 600 11px/1.2 var(--font-ui); letter-spacing: 0.11em; text-transform: uppercase; color: var(--ink-2);
  padding-bottom: var(--s-2); border-bottom: 1px solid var(--rule-strong); }
.board-head > div:not(:first-child) { text-align: right; }
.board-head .unit { display: block; text-transform: none; letter-spacing: 0; font-weight: 400; color: var(--ink-3); margin-top: 2px; }
.seat { padding: var(--s-4) 0; border-bottom: 1px solid var(--rule); }
.seat .who { display: grid; grid-template-columns: 52px 1fr; gap: var(--s-3); align-items: center; min-width: 0; }
.seat .name { font-weight: 600; font-size: 15px; display: flex; align-items: center; gap: var(--s-2); flex-wrap: wrap; }
.mode { font: 600 10px/1 var(--font-ui); letter-spacing: 0.11em; text-transform: uppercase; padding: 3px 5px;
  border: 1px solid var(--ink-3); color: var(--ink-2); border-radius: 2px; }
.mode.live { border-color: var(--accent); color: var(--accent-ink); }
.seat .what { color: var(--ink-2); font-size: 12px; line-height: 1.4; margin-top: 2px; }
.seat .seen { color: var(--ink-3); font-size: 11px; margin-top: 4px; }
.seat .seen .dot { display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: var(--ink-3);
  margin-right: 5px; vertical-align: 1px; }
.seat .seen .dot.on { background: var(--pos); }
.cell { text-align: right; font-family: var(--font-mono); font-variant-numeric: tabular-nums; font-size: 15px;
  transition: background-color 600ms ease-out; padding: 2px 0; }
.cell .sub { display: block; font-size: 11px; color: var(--ink-3); margin-top: 2px; line-height: 1.35; }
.cell .lbl { display: none; }
.cell .sub b { font-weight: 400; white-space: nowrap; }
.pos { color: var(--pos); } .neg { color: var(--neg); } .na { color: var(--ink-3); }
.flash-pos { background: color-mix(in srgb, var(--pos) 20%, transparent); transition: none; }
.flash-neg { background: color-mix(in srgb, var(--neg) 20%, transparent); transition: none; }
.spark { display: block; width: 132px; height: 36px; }
.spark-cap { font-size: 11px; color: var(--ink-3); text-align: right; }
.seat .empty-note { grid-column: 2 / -1; color: var(--ink-3); font-size: 13px; }
.seat.nodata .who { opacity: 0.75; }
.notes { grid-column: 1 / -1; font-size: 12px; color: var(--ink-3); padding-left: 64px; overflow-wrap: anywhere; min-width: 0; }
.notes p { margin: 2px 0 0; }
.err { color: var(--neg); }
.flag { font: 600 10px/1 var(--font-ui); letter-spacing: 0.08em; text-transform: uppercase; padding: 3px 6px; border-radius: 2px;
  border: 1px solid currentColor; }
.flag.green { color: var(--pos); } .flag.red { color: var(--neg); } .flag.amber { color: var(--accent-ink); } .flag.grey { color: var(--ink-3); }

.empty { padding: var(--s-7) 0; border-bottom: 1px solid var(--rule); }
.empty h2 { margin: 0; font: 600 18px/1.3 var(--font-ui); }
.empty p { margin: var(--s-2) 0 0; color: var(--ink-2); max-width: 64ch; }

.method { margin-top: var(--s-7); border-top: 2px solid var(--rule-strong); padding-top: var(--s-3); max-width: 80ch;
  color: var(--ink-2); font-size: 13px; }
.method h2 { color: var(--ink); font: 600 15px/1.3 var(--font-ui); margin: 0 0 var(--s-2); }
.method p { margin: var(--s-2) 0; }

footer.status { position: fixed; inset: auto 0 0 0; background: var(--paper-2); border-top: 1px solid var(--rule);
  font: 400 11px/1 var(--font-mono); color: var(--ink-2); }
footer.status .wrap { display: flex; gap: var(--s-4); justify-content: space-between; padding-block: 9px; overflow: hidden;
  white-space: nowrap; }
footer.status .live-dot { color: var(--pos); }
footer.status .stale { color: var(--accent-ink); }

@media (max-width: 980px) {
  .board-head { display: none; }
  .seat { grid-template-columns: repeat(3, 1fr); row-gap: var(--s-3); }
  .seat .who { grid-column: 1 / -1; }
  .cell { text-align: left; }
  .cell .lbl { display: block; font: 600 10px/1.2 var(--font-ui); letter-spacing: 0.11em; text-transform: uppercase;
    color: var(--ink-2); margin-bottom: 4px; }
  .sparkbox { grid-column: 1 / -1; }
  .spark { width: 100%; }
  .spark-cap { text-align: left; }
  .notes { padding-left: 0; }
  .seat .empty-note { grid-column: 1 / -1; }
  section.sport-head { grid-template-columns: 1fr; }
}
@media (max-width: 420px) {
  .seat { grid-template-columns: repeat(2, 1fr); }
  .stat dd { font-size: 19px; }
  .sport-head h1 { font-size: 24px; }
}
@media (prefers-reduced-motion: reduce) { .cell { transition: none; } }
</style>
</head>
<body>
<div class="wrap">
  <header class="top">
    <div class="brand">Ashcroft<span>Traders</span></div>
    <div class="clock"><span id="when"></span><button class="theme" id="theme" type="button" aria-label="Switch colour theme">Theme</button></div>
  </header>
  <nav class="sports" role="tablist" aria-label="Sport" id="tabs"></nav>
  <main id="main"></main>
  <section class="method">
    <h2>How the figures are made</h2>
    <p><b>P&amp;L</b> is settled P&amp;L after commission, read from each trader's own ledger the way its own settle step reads it. A market still to run is not in it. <b>Volume</b> is the money matched on the trader's orders that day (backs; lays for the early lay). Paper traders place nothing: their figures are what the rule would have made.</p>
    <p><b>Forecast</b> follows the owner's day report: never the raw expected value. The record is every fully settled day before today; the expected value still to run is scaled by the record's settled result over its expected value. A trader with no expected value in its ledger is forecast from its record's mean a market. The range is the 10th-90th percentile of the record's misses. <b>Month forecast</b> adds the record's mean day for each day left, if the trader trades every one of them. Fewer than 3 settled days: no forecast.</p>
    <p id="srcline"></p>
  </section>
</div>
<footer class="status"><div class="wrap"><span id="feed"></span><span id="gen"></span></div></footer>
<script id="data" type="application/json">__DATA__</script>
<script>
(function () {
  "use strict";
  var REFRESH_MS = 60000;
  var data = JSON.parse(document.getElementById("data").textContent);
  var live = false, lastFetch = null, sel = null;

  // ---- theme ----
  var root = document.documentElement;
  try { var th = localStorage.getItem("r2-theme"); if (th) root.setAttribute("data-theme", th); } catch (e) {}
  document.getElementById("theme").addEventListener("click", function () {
    var cur = root.getAttribute("data-theme") ||
      (window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
    var next = cur === "light" ? "dark" : "light";
    root.setAttribute("data-theme", next);
    try { localStorage.setItem("r2-theme", next); } catch (e) {}
  });

  // ---- formatting ----
  var MINUS = "−", DASH = "–";
  function gbp(v, signed) {
    if (v === null || v === undefined || !isFinite(v)) return null;
    var a = Math.abs(v), s = a >= 1000 ? a.toLocaleString("en-GB", {maximumFractionDigits: 0})
      : a.toLocaleString("en-GB", {minimumFractionDigits: 2, maximumFractionDigits: 2});
    var sign = v < -0.004 ? MINUS : (signed && v > 0.004 ? "+" : "");
    return sign + "£" + s;
  }
  function cls(v) { return v === null || v === undefined ? "na" : v > 0.004 ? "pos" : v < -0.004 ? "neg" : ""; }
  function esc(s) { return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
    return {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]; }); }
  function money(v, signed) { var t = gbp(v, signed); return t === null ? '<span class="na">' + DASH + '</span>' : t; }
  function dm(day) {
    var d = new Date(day + "T12:00:00Z");
    return isNaN(d) ? day : d.toLocaleDateString("en-GB", {day: "numeric", month: "short", timeZone: "UTC"});
  }
  function ukTime(iso) {
    if (!iso) return null;
    var d = new Date(iso); if (isNaN(d)) return null;
    return d.toLocaleTimeString("en-GB", {hour: "2-digit", minute: "2-digit", timeZone: "Europe/London"});
  }

  // ---- avatars: a sport glyph on a 24 grid, the ring solid amber for real money, dashed for paper ----
  var GLYPH = {
    horse: '<path d="M8 20.5v-4.2c0-2.6.9-4.6 2.6-6.4L11.4 5l2.3 2.1 2.7-.6c2.1.9 3.4 3.1 4.1 6.1l-1.1 1.9-3.1-.9-1.8 2.6v4.3"/><path d="M11.4 5 10 3.5"/><circle cx="15.6" cy="9.6" r="1" class="datum"/>',
    greyhound: '<path d="M2.8 13.6 10.2 11c.9-2.9 3-4.8 5.6-5l.9-2.2 1 2.6c1.6 1 2.6 2.9 2.6 5.2v8.9"/><path d="M2.8 13.6l.9 1.7 6.3-.6 3.4 5.8"/><path d="M16.7 3.8 14.6 8"/><circle cx="15.4" cy="10.1" r="1" class="datum"/>',
    football: '<circle cx="12" cy="12" r="8.6"/><path d="M12 8.3l3.5 2.5-1.3 4.1H9.8l-1.3-4.1z" class="datum"/><path d="M12 8.3V3.4M15.5 10.8l4.6-1.5M14.2 14.9l2.9 4M9.8 14.9l-2.9 4M8.5 10.8 3.9 9.3"/>',
    cricket: '<path d="M4.6 17.8l8.9-8.9 2.5 2.5-8.9 8.9a1.8 1.8 0 0 1-2.5 0 1.8 1.8 0 0 1 0-2.5z"/><path d="M14.8 10.2l4.3-4.3"/><circle cx="17.6" cy="17.4" r="2.2" class="datum"/>'
  };
  function avatar(t) {
    var tone = "var(--" + (t.glyph || "horse") + ")";
    var ring = t.mode === "live"
      ? '<circle cx="26" cy="26" r="24.5" fill="none" stroke="var(--accent)" stroke-width="2"/>'
      : '<circle cx="26" cy="26" r="24.5" fill="none" stroke="var(--ink-3)" stroke-width="1.5" stroke-dasharray="3 3"/>';
    return '<svg width="52" height="52" viewBox="0 0 52 52" role="img" aria-label="' + esc(t.name) + ' avatar">' +
      '<circle cx="26" cy="26" r="22" fill="var(--paper-2)"/>' + ring +
      '<g transform="translate(11 10) scale(1.2)" fill="none" stroke="' + tone + '" stroke-width="1.5" ' +
      'stroke-linecap="square" stroke-linejoin="miter">' + GLYPH[t.glyph || "horse"].replace(/class="datum"/g,
      'fill="' + tone + '" stroke="none"') + '</g>' +
      '<rect x="31" y="35" width="21" height="14" rx="2" fill="var(--paper)" stroke="var(--rule)"/>' +
      '<text x="41.5" y="45.3" text-anchor="middle" font-family="Geist Mono, monospace" font-size="9" ' +
      'font-weight="500" fill="var(--ink)">' + esc(t.initials) + '</text></svg>';
  }

  // ---- the 30-day sparkline: daily P&L as bars from zero ----
  function spark(daily) {
    var pts = (daily || []).filter(function (d) { return d.pnl !== null; });
    if (!pts.length) return '<div class="spark-cap">no settled days</div>';
    var W = 132, H = 36, n = pts.length, bw = Math.max(1.5, Math.min(6, (W - n) / n));
    var mx = Math.max.apply(null, pts.map(function (d) { return Math.abs(d.pnl); })) || 1;
    var mid = H / 2, out = "";
    pts.forEach(function (d, i) {
      var h = Math.max(1, Math.abs(d.pnl) / mx * (mid - 2)), x = W - (n - i) * (bw + 1);
      var y = d.pnl >= 0 ? mid - h : mid;
      out += '<rect x="' + x.toFixed(1) + '" y="' + y.toFixed(1) + '" width="' + bw.toFixed(1) + '" height="' +
        h.toFixed(1) + '" fill="var(--' + (d.pnl >= 0 ? "pos" : "neg") + ')"><title>' + d.day + ": " +
        esc(gbp(d.pnl, true)) + '</title></rect>';
    });
    return '<svg class="spark" viewBox="0 0 ' + W + ' ' + H + '" preserveAspectRatio="none" role="img" aria-label="Daily P&amp;L, last 30 days">' +
      '<line x1="0" x2="' + W + '" y1="' + mid + '" y2="' + mid + '" stroke="var(--rule)" stroke-width="1"/>' + out +
      '</svg><div class="spark-cap">' + n + ' settled day' + (n === 1 ? "" : "s") + ', last 30</div>';
  }

  function range(lo, hi) {
    if (lo === null || hi === null || lo === undefined || hi === undefined) return "";
    return '<span class="sub"><b>' + esc(gbp(lo, true)) + "</b> to <b>" + esc(gbp(hi, true)) + '</b></span>';
  }
  function cell(key, label, value, signed, sub) {
    return '<div class="cell" data-k="' + key + '" data-v="' + (value === null || value === undefined ? "" : value) +
      '"><span class="lbl">' + label + '</span><span class="' + (signed ? cls(value) : "") + '">' +
      money(value, signed) + '</span>' + (sub || "") + '</div>';
  }

  function edgeLine(t) {
    var e = t.edge;
    if (!e || !t.has_data) return "";
    var pct = function (v) { return v === null || v === undefined ? DASH : (v > 0 ? "+" : v < 0 ? MINUS : "") + Math.abs(v * 100).toFixed(1) + "%"; };
    return '<div class="seen"><span class="flag ' + esc(e.flag) + '" title="' + esc(e.flag_reason) + '">' + esc(e.flag_label) +
      '</span> CLV 7 days <span class="mono">' + pct(e.clv_7d) + '</span> \u00b7 record <span class="mono">' + pct(e.clv_record) +
      '</span>' + (e.ev_today !== null ? ' \u00b7 EV today <span class="mono">' + esc(gbp(e.ev_today, true)) + '</span>' : "") + '</div>';
  }

  function seat(t) {
    var d = t.today, m = t.month, a = t.all_time;
    var seen = ukTime(d.last_order_utc), fresh = d.last_order_utc &&
      (Date.now() - new Date(d.last_order_utc).getTime()) < 20 * 60000;
    var who = '<div class="who">' + avatar(t) + '<div><div class="name">' + esc(t.name) +
      '<span class="mode ' + (t.mode === "live" ? "live" : "") + '">' + (t.mode === "live" ? "Real money" : "Paper") +
      '</span></div><div class="what">' + esc(t.what) + '</div>' + edgeLine(t) +
      '<div class="seen"><span class="dot' + (fresh ? " on" : "") + '"></span>' +
      (seen ? "last ledger row " + seen + " UK" : (d.traded ? "ledger today" : "no ledger today")) +
      (d.traded ? " · " + d.settled + " of " + d.markets + " markets settled" : "") + '</div></div></div>';
    var notes = (t.notes || []).map(function (n) { return "<p>" + esc(n) + "</p>"; }).join("");
    if (t.error) notes += '<p class="err">Not read: ' + esc(t.error) + '</p>';
    if (a.open_days && a.open_days.length) notes += "<p>Days not yet fully settled: " + esc(a.open_days.join(", ")) + "</p>";
    notes += '<p class="mono">Source: ' + esc(t.source) + '</p>';
    if (!t.has_data) {
      return '<div class="seat nodata" data-id="' + esc(t.id) + '">' + who +
        '<div class="empty-note">No data found: no ledger in S3 for this trader yet.</div>' +
        '<div class="notes">' + notes + '</div></div>';
    }
    var fsub = d.forecast !== null ? range(d.forecast_lo, d.forecast_hi) :
      '<span class="sub">' + (d.traded ? "too few settled days" : "not trading today") + '</span>';
    if (d.forecast_7 !== null && d.forecast_7 !== undefined) fsub += '<span class="sub">last 7 days: <b>' + esc(gbp(d.forecast_7, true)) + '</b></span>';
    var vsub = d.open_liability ? '<span class="sub">open liability ' + esc(gbp(d.open_liability)) + '</span>' : "";
    var msub = m.forecast !== null ? range(m.forecast_lo, m.forecast_hi) : '<span class="sub">too few settled days</span>';
    var psub = '<span class="sub">' + m.days + ' day' + (m.days === 1 ? "" : "s") + ' traded</span>';
    var asub = '<span class="sub">' + a.days + ' days, ' + a.days_up + ' up' + (a.first_day ? ', from <b>' + esc(dm(a.first_day)) + '</b>' : "") + '</span>';
    return '<div class="seat" data-id="' + esc(t.id) + '">' + who +
      cell("tp", "Today P&amp;L", d.pnl, true, "") +
      cell("tv", "Today volume", d.volume, false, vsub) +
      cell("tf", "Today forecast", d.forecast, true, fsub) +
      cell("mp", "Month P&amp;L", m.pnl, true, psub) +
      cell("mf", "Month forecast", m.forecast, true, msub) +
      cell("ap", "All time P&amp;L", a.pnl, true, asub) +
      '<div class="sparkbox">' + spark(t.daily) + '</div>' +
      '<div class="notes">' + notes + '</div></div>';
  }

  function sportPnl(s) { return s.live_totals ? s.live_totals.today_pnl : null; }

  function renderTabs() {
    var tabs = document.getElementById("tabs");
    tabs.innerHTML = data.sports.map(function (s) {
      var p = sportPnl(s);
      return '<button role="tab" type="button" id="tab-' + s.id + '" aria-controls="main" aria-selected="' +
        (s.id === sel) + '" data-s="' + s.id + '" tabindex="' + (s.id === sel ? 0 : -1) + '">' + esc(s.label) +
        '<span class="tab-pnl">' + (p === null ? "" : esc(gbp(p, true))) + '</span></button>';
    }).join("");
    Array.prototype.forEach.call(tabs.querySelectorAll("button"), function (b) {
      b.addEventListener("click", function () { choose(b.getAttribute("data-s")); });
      b.addEventListener("keydown", function (e) {
        if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
        var ids = data.sports.map(function (s) { return s.id; }), i = ids.indexOf(sel);
        choose(ids[(i + (e.key === "ArrowRight" ? 1 : ids.length - 1)) % ids.length]);
        document.getElementById("tab-" + sel).focus();
      });
    });
  }

  function renderMain(prev) {
    var s = data.sports.filter(function (x) { return x.id === sel; })[0];
    var main = document.getElementById("main");
    main.setAttribute("aria-labelledby", "tab-" + s.id);
    var live = s.traders.filter(function (t) { return t.mode === "live" && t.has_data; });
    var lt = s.live_totals || {};
    var head = '<section class="sport-head"><div><h1>' + esc(s.label) + '</h1><p>' +
      (s.traders.length ? s.traders.length + " trader" + (s.traders.length === 1 ? "" : "s") + ", " +
        live.length + " on real money" : "No traders") + '</p></div></section>';
    if (!s.traders.length) {
      main.innerHTML = head + '<div class="empty"><h2>No data found</h2><p>No ' + esc(s.label.toLowerCase()) +
        ' trader is running: there is no ledger for one in S3. A trader appears here once it writes its first ledger.</p></div>';
      return;
    }
    var stats = live.length ? '<dl class="stats">' +
      [["Today P&amp;L", lt.today_pnl, true], ["Today volume", lt.today_volume, false],
       ["Today forecast", lt.today_forecast, true], ["Month P&amp;L", lt.month_pnl, true],
       ["Month forecast", lt.month_forecast, true], ["All time P&amp;L", lt.all_time_pnl, true]]
        .map(function (x) { return '<div class="stat"><dt>' + x[0] + '</dt><dd class="' + (x[2] ? cls(x[1]) : "") +
          '">' + money(x[1], x[2]) + '</dd></div>'; }).join("") +
      '</dl><p class="stats-label">Real-money traders only, summed. Paper traders are not in these totals.</p>' : "";
    var board = '<div class="board" role="table" aria-label="' + esc(s.label) + ' traders"><div class="board-head" role="row">' +
      '<div>Trader</div><div>Today P&amp;L<span class="unit">settled</span></div><div>Volume<span class="unit">matched today</span></div>' +
      '<div>Forecast<span class="unit">today</span></div><div>Month P&amp;L<span class="unit">to date</span></div>' +
      '<div>Forecast<span class="unit">month</span></div><div>All time<span class="unit">P&amp;L</span></div>' +
      '<div>Daily P&amp;L<span class="unit">last 30 days</span></div></div>' +
      s.traders.slice().sort(function (a, b) {
        return (b.mode === "live") - (a.mode === "live") || (b.has_data - a.has_data);
      }).map(seat).join("") + '</div>';
    main.innerHTML = head + stats + board;
    if (prev) flash(prev);
  }

  function snapshot() {
    var out = {};
    Array.prototype.forEach.call(document.querySelectorAll(".seat[data-id] .cell[data-k]"), function (c) {
      out[c.closest(".seat").getAttribute("data-id") + "/" + c.getAttribute("data-k")] = c.getAttribute("data-v");
    });
    return out;
  }
  function flash(prev) {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    Array.prototype.forEach.call(document.querySelectorAll(".seat[data-id] .cell[data-k]"), function (c) {
      var k = c.closest(".seat").getAttribute("data-id") + "/" + c.getAttribute("data-k");
      var was = prev[k], now = c.getAttribute("data-v");
      if (was === undefined || was === now || was === "" || now === "") return;
      c.classList.add(parseFloat(now) > parseFloat(was) ? "flash-pos" : "flash-neg");
      setTimeout(function () { c.classList.remove("flash-pos", "flash-neg"); }, 50);
    });
  }

  function renderStatus() {
    var gen = new Date(data.generated_utc);
    var age = Math.round((Date.now() - gen.getTime()) / 60000);
    document.getElementById("when").innerHTML = "<b>" + esc(data.today) + "</b> UK trading day";
    document.getElementById("gen").textContent = "built " + data.generated_utc.replace("T", " ").replace("Z", "") +
      " UTC (" + (age <= 0 ? "just now" : age + " min ago") + ")";
    var f = document.getElementById("feed");
    f.innerHTML = live ? '<span class="live-dot">●</span> live: re-read every 60 s' +
      (lastFetch ? ", last " + lastFetch.toLocaleTimeString("en-GB", {timeZone: "Europe/London"}) + " UK" : "")
      : '<span class="stale">○</span> figures as built: open it where traders.json is served to update live';
    if (age > 30) f.innerHTML += ' <span class="stale">· the data is ' + age + ' min old</span>';
    document.getElementById("srcline").innerHTML = "<b>Source:</b> " + esc(data.source) +
      (data.errors && data.errors.length ? ' <span class="err">' + data.errors.length + " trader(s) not read this build.</span>" : "");
  }

  function choose(id) {
    sel = id;
    try { history.replaceState(null, "", "#" + id); } catch (e) {}
    renderTabs(); renderMain(null);
  }

  function refresh() {
    fetch("traders.json?t=" + Date.now(), {cache: "no-store"}).then(function (r) {
      if (!r.ok) throw new Error(r.status);
      return r.json();
    }).then(function (j) {
      if (!j || !j.sports) throw new Error("bad");
      var prev = snapshot();
      data = j; live = true; lastFetch = new Date();
      renderTabs(); renderMain(prev); renderStatus();
    }).catch(function () { live = false; renderStatus(); });
  }

  var h = (location.hash || "").slice(1);
  sel = data.sports.some(function (s) { return s.id === h; }) ? h : data.sports[0].id;
  renderTabs(); renderMain(null); renderStatus();
  if (location.protocol !== "file:") { refresh(); setInterval(refresh, REFRESH_MS); }
  setInterval(renderStatus, 30000);
})();
</script>
</body>
</html>
"""


def render(data: dict) -> str:
    """The page with ``data`` embedded (``</`` escaped so the JSON cannot close its script tag)."""
    blob = json.dumps(data, separators=(",", ":"), default=str).replace("</", "<\\/")
    return TEMPLATE.replace("__DATA__", blob)
