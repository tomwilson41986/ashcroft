/* Ashcroft — shared JS */

async function apiFetch(path) {
    try {
        const resp = await fetch(API_BASE + path);
        if (!resp.ok) { console.error(`API ${path}: ${resp.status}`); return null; }
        return await resp.json();
    } catch (err) {
        console.error(`API ${path}: ${err.message}`);
        return null;
    }
}

function fmtOdds(val) { return val != null ? val.toFixed(2) : '-'; }
function fmtPnl(val) { return val != null ? (val >= 0 ? '+' : '') + '£' + val.toFixed(2) : '-'; }
function normaliseName(name) { return (name || '').trim().toLowerCase().replace(/\s*\([a-z]{2,3}\)\s*$/, ''); }

function ordinal(n) {
    if (!n) return '-';
    const s = ['th', 'st', 'nd', 'rd'];
    const v = n % 100;
    return n + (s[(v - 20) % 10] || s[v] || s[0]);
}

function statusBadge(status) {
    const cls = status === 'WON' ? 'badge-won' : status === 'LOST' ? 'badge-lost' : 'badge-pending';
    return `<span class="badge ${cls}">${status}</span>`;
}

function typeBadge(type) {
    const cls = type === 'BACK' ? 'badge-back' : 'badge-lay';
    return `<span class="badge ${cls}">${type}</span>`;
}

function colorClass(val, zero) {
    if (val > (zero || 0)) return 'text-green';
    if (val < (zero || 0)) return 'text-red';
    return '';
}

function chartOptions(prefix) {
    return {
        responsive: true,
        plugins: {
            legend: { labels: { color: '#64748b', font: { size: 11 } } }
        },
        scales: {
            x: { ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { color: '#f1f5f9' } },
            y: { ticks: { color: '#94a3b8', font: { size: 10 }, callback: v => prefix + v }, grid: { color: '#f1f5f9' } }
        }
    };
}

function renderBetsTable(bets) {
    if (!bets || bets.length === 0) return '<div class="empty-state"><p>No bets found.</p></div>';

    let html = `<table><thead><tr>
        <th>Time</th><th>Track</th><th>Horse</th>
        <th class="text-right">Pred BFSP</th><th class="text-right">Back</th>
        <th class="text-right">Edge</th><th class="text-right">Stake</th>
        <th>Status</th><th class="text-right">P&amp;L</th>
    </tr></thead><tbody>`;

    for (const b of bets) {
        html += `<tr>
            <td>${b.race_time || '-'}</td>
            <td>${b.track || '-'}</td>
            <td class="font-medium">${b.horse_name}</td>
            <td class="text-right">${b.predicted_bfsp ? b.predicted_bfsp.toFixed(2) : '-'}</td>
            <td class="text-right">${b.betfair_back ? b.betfair_back.toFixed(2) : '-'}</td>
            <td class="text-right ${b.edge_pct >= 15 ? 'text-green font-bold' : ''}">${b.edge_pct ? '+' + b.edge_pct.toFixed(1) + '%' : '-'}</td>
            <td class="text-right">£${b.stake ? b.stake.toFixed(2) : '-'}</td>
            <td>${statusBadge(b.status)}</td>
            <td class="text-right font-medium ${colorClass(b.net_pnl)}">
                ${b.net_pnl != null ? fmtPnl(b.net_pnl) : '-'}
            </td>
        </tr>`;
    }
    return html + '</tbody></table>';
}

/* Betting strategies */
const STRATEGIES = [
    { id: 'all',           label: 'All Predictions' },
    { id: 'all_edge',      label: 'All Predictions (Edge Only)' },
    { id: 'top1',          label: 'Top Ranked' },
    { id: 'top1_edge',     label: 'Top Ranked (Edge Only)' },
    { id: 'top3',          label: 'Top 3 Ranked' },
    { id: 'top3_edge',     label: 'Top 3 Ranked (Edge Only)' },
];

function applyStrategy(bets, strategyId) {
    if (!bets) return [];
    switch (strategyId) {
        case 'all':        return bets;
        case 'all_edge':   return bets.filter(b => b.edge_pct > 0);
        case 'top1':       return bets.filter(b => b.rank === 1);
        case 'top1_edge':  return bets.filter(b => b.rank === 1 && b.edge_pct > 0);
        case 'top3':       return bets.filter(b => b.rank >= 1 && b.rank <= 3);
        case 'top3_edge':  return bets.filter(b => b.rank >= 1 && b.rank <= 3 && b.edge_pct > 0);
        default:           return bets;
    }
}

function computeSummaryFromBets(bets) {
    const settled = bets.filter(b => b.status === 'WON' || b.status === 'LOST');
    if (!settled.length) return { total_bets: 0, wins: 0, strike_rate: 0, net_pnl: 0, roi: 0, avg_edge: 0, bank: 1000 };
    const wins = settled.filter(b => b.status === 'WON').length;
    const netPnl = settled.reduce((s, b) => s + (b.net_pnl || 0), 0);
    const staked = settled.reduce((s, b) => s + (b.stake || 0), 0);
    const edges = settled.map(b => b.edge_pct || 0);
    return {
        total_bets: settled.length,
        wins,
        strike_rate: Math.round(wins / settled.length * 1000) / 10,
        net_pnl: Math.round(netPnl * 100) / 100,
        roi: staked ? Math.round(netPnl / staked * 10000) / 100 : 0,
        avg_edge: Math.round(edges.reduce((a, b) => a + b, 0) / edges.length * 10) / 10,
        bank: Math.round((1000 + netPnl) * 100) / 100,
    };
}

function computeDailyPnlFromBets(bets, stake) {
    stake = stake || 10;
    const settled = bets.filter(b => b.status === 'WON' || b.status === 'LOST');
    const byDate = {};
    for (const b of settled) {
        const d = b.race_date;
        if (!byDate[d]) byDate[d] = [];
        byDate[d].push(b);
    }
    const dates = Object.keys(byDate).sort();
    let cumPnl = 0, bank = 1000;
    return dates.map(d => {
        const dayBets = byDate[d];
        const winners = dayBets.filter(b => b.status === 'WON').length;
        const losers = dayBets.length - winners;
        const netPnl = dayBets.reduce((s, b) => s + (b.net_pnl || 0), 0);
        const staked = dayBets.reduce((s, b) => s + (b.stake || 0), 0);
        cumPnl += netPnl;
        const bankStart = bank;
        bank += netPnl;
        return {
            date: d, num_bets: dayBets.length, winners, losers, voids: 0,
            total_staked: Math.round(staked * 100) / 100,
            net_pnl: Math.round(netPnl * 100) / 100,
            roi_pct: staked ? Math.round(netPnl / staked * 10000) / 100 : 0,
            cumulative_pnl: Math.round(cumPnl * 100) / 100,
            bank_start: Math.round(bankStart * 100) / 100,
            bank_end: Math.round(bank * 100) / 100,
        };
    });
}

function computeEdgeStatsFromBets(bets) {
    const settled = bets.filter(b => b.status === 'WON' || b.status === 'LOST');
    const buckets = { '20%+': [], '15-20%': [], '10-15%': [], '5-10%': [], '<5%': [] };
    for (const b of settled) {
        const e = b.edge_pct || 0;
        if (e >= 20) buckets['20%+'].push(b);
        else if (e >= 15) buckets['15-20%'].push(b);
        else if (e >= 10) buckets['10-15%'].push(b);
        else if (e >= 5) buckets['5-10%'].push(b);
        else buckets['<5%'].push(b);
    }
    return Object.entries(buckets)
        .filter(([, items]) => items.length > 0)
        .map(([bucket, items]) => ({
            bucket,
            total: items.length,
            won: items.filter(b => b.status === 'WON').length,
            pnl: Math.round(items.reduce((s, b) => s + (b.net_pnl || 0), 0) * 100) / 100,
        }));
}

function renderStrategySelect(id, onChange) {
    let html = `<select id="${id}" class="input">`;
    for (const s of STRATEGIES) {
        html += `<option value="${s.id}">${s.label}</option>`;
    }
    html += '</select>';
    return html;
}

/* Sidebar renderer */
function renderSidebar(activePage) {
    return `
    <aside class="sidebar">
        <a href="/" class="sidebar-brand"><span>A</span>shcroft</a>

        <div class="sidebar-section">
            <div class="sidebar-section-title">Overview</div>
            <a href="/" class="sidebar-link ${activePage === 'dashboard' ? 'active' : ''}">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-4 0h4"/></svg>
                Dashboard
            </a>
        </div>

        <div class="sidebar-section">
            <div class="sidebar-section-title">Markets</div>
            <a href="/races.html" class="sidebar-link ${activePage === 'races' ? 'active' : ''}">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"/></svg>
                Live Races
            </a>
            <a href="/bets.html" class="sidebar-link ${activePage === 'bets' ? 'active' : ''}">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"/></svg>
                Bet Tracker
            </a>
        </div>

        <div class="sidebar-section">
            <div class="sidebar-section-title">Trading</div>
            <a href="/paper-trades.html" class="sidebar-link ${activePage === 'paper-trades' ? 'active' : ''}">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z"/></svg>
                Paper Trading
            </a>
        </div>

        <div class="sidebar-section">
            <div class="sidebar-section-title">Analytics</div>
            <a href="/performance.html" class="sidebar-link ${activePage === 'performance' ? 'active' : ''}">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/></svg>
                Performance
            </a>
        </div>
    </aside>`;
}
