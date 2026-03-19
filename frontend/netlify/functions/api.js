// Netlify Function — proxies S3 reads for the Ashcroft dashboard.
// Routes:
//   /api/bets/all         → dashboard/bets.json
//   /api/bets/{date}      → filters bets.json by date
//   /api/bets/edge-stats  → computed from bets.json
//   /api/pnl/daily        → dashboard/daily_pnl.json
//   /api/pnl/summary      → computed from bets.json + daily_pnl.json
//   /api/odds/latest/{d}  → dashboard/odds/{date}.json

const { S3Client, GetObjectCommand } = require("@aws-sdk/client-s3");

const BUCKET = process.env.ULTRA_BETTING_S3_BUCKET || "ashcroft";
const s3 = new S3Client({ region: process.env.AWS_DEFAULT_REGION || "us-east-1" });

async function readS3Json(key) {
  try {
    const resp = await s3.send(new GetObjectCommand({ Bucket: BUCKET, Key: key }));
    const body = await resp.Body.transformToString();
    return JSON.parse(body);
  } catch (e) {
    if (e.name === "NoSuchKey") return null;
    console.error(`S3 read failed for ${key}:`, e.message);
    return null;
  }
}

function json(data, status = 200) {
  return {
    statusCode: status,
    headers: {
      "Content-Type": "application/json",
      "Access-Control-Allow-Origin": "*",
      "Cache-Control": "public, max-age=30",
    },
    body: JSON.stringify(data),
  };
}

// Edge stats computed from bets
function computeEdgeStats(bets) {
  const settled = bets.filter(b => b.status === "WON" || b.status === "LOST");
  const buckets = { "20%+": [], "15-20%": [], "10-15%": [], "5-10%": [], "<5%": [] };

  for (const b of settled) {
    const e = b.edge_pct || 0;
    if (e >= 20) buckets["20%+"].push(b);
    else if (e >= 15) buckets["15-20%"].push(b);
    else if (e >= 10) buckets["10-15%"].push(b);
    else if (e >= 5) buckets["5-10%"].push(b);
    else buckets["<5%"].push(b);
  }

  return Object.entries(buckets)
    .filter(([, items]) => items.length > 0)
    .map(([bucket, items]) => ({
      bucket,
      total: items.length,
      won: items.filter(b => b.status === "WON").length,
      pnl: Math.round(items.reduce((s, b) => s + (b.net_pnl || 0), 0) * 100) / 100,
    }));
}

// P&L summary computed from bets + daily_pnl
function computeSummary(bets, dailyPnl) {
  const settled = bets.filter(b => b.status === "WON" || b.status === "LOST");
  if (!settled.length) return { total_bets: 0, wins: 0, net_pnl: 0, roi: 0, bank: 1000, avg_edge: 0 };

  const wins = settled.filter(b => b.status === "WON").length;
  const netPnl = settled.reduce((s, b) => s + (b.net_pnl || 0), 0);
  const staked = settled.reduce((s, b) => s + (b.stake || 0), 0);
  const edges = settled.map(b => b.edge_pct || 0);
  const bank = dailyPnl && dailyPnl.length ? dailyPnl[dailyPnl.length - 1].bank_end : 1000;

  return {
    total_bets: settled.length,
    wins,
    strike_rate: Math.round(wins / settled.length * 1000) / 10,
    net_pnl: Math.round(netPnl * 100) / 100,
    roi: staked ? Math.round(netPnl / staked * 10000) / 100 : 0,
    avg_edge: Math.round(edges.reduce((a, b) => a + b, 0) / edges.length * 10) / 10,
    bank: Math.round(bank * 100) / 100,
  };
}

exports.handler = async (event) => {
  const path = event.path.replace("/.netlify/functions/api", "").replace("/api", "");

  // /pnl/daily
  if (path === "/pnl/daily") {
    const data = await readS3Json("dashboard/daily_pnl.json");
    return json(data || []);
  }

  // /pnl/summary
  if (path === "/pnl/summary") {
    const [bets, pnl] = await Promise.all([
      readS3Json("dashboard/bets.json"),
      readS3Json("dashboard/daily_pnl.json"),
    ]);
    return json(computeSummary(bets || [], pnl || []));
  }

  // /bets/all
  if (path === "/bets/all") {
    const data = await readS3Json("dashboard/bets.json");
    return json(data || []);
  }

  // /bets/edge-stats
  if (path === "/bets/edge-stats") {
    const bets = await readS3Json("dashboard/bets.json");
    return json(computeEdgeStats(bets || []));
  }

  // /bets/live/today
  if (path === "/bets/live/today") {
    const bets = await readS3Json("dashboard/bets.json");
    const today = new Date().toISOString().slice(0, 10);
    return json((bets || []).filter(b => b.race_date === today && b.status === "PENDING"));
  }

  // /bets/{date}
  const betsMatch = path.match(/^\/bets\/(\d{4}-\d{2}-\d{2})$/);
  if (betsMatch) {
    const bets = await readS3Json("dashboard/bets.json");
    return json((bets || []).filter(b => b.race_date === betsMatch[1]));
  }

  // /odds/latest/{date}
  const oddsMatch = path.match(/^\/odds\/latest\/(\d{4}-\d{2}-\d{2})$/);
  if (oddsMatch) {
    const data = await readS3Json(`dashboard/odds/${oddsMatch[1]}.json`);
    return json(data || []);
  }

  return json({ error: "Not found" }, 404);
};
