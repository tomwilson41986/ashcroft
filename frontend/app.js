/* Ashcroft — shared JS helpers */

async function apiFetch(path) {
    try {
        const resp = await fetch(API_BASE + path);
        if (!resp.ok) {
            console.error(`API ${path}: ${resp.status}`);
            return null;
        }
        return await resp.json();
    } catch (err) {
        console.error(`API ${path}: ${err.message}`);
        return null;
    }
}

function fmtOdds(val) {
    return val != null ? val.toFixed(2) : '-';
}

function normaliseName(name) {
    return (name || '').trim().toLowerCase().replace(/\s*\([a-z]{2,3}\)\s*$/, '');
}

function statusBadge(status) {
    const cls = status === 'WON' ? 'badge-won' :
                status === 'LOST' ? 'badge-lost' : 'badge-pending';
    return `<span class="px-2 py-0.5 rounded text-xs font-medium ${cls}">${status}</span>`;
}

function ordinal(n) {
    if (!n) return '-';
    const s = ['th', 'st', 'nd', 'rd'];
    const v = n % 100;
    return n + (s[(v - 20) % 10] || s[v] || s[0]);
}

function setStatWithColor(id, text, isPositive) {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = text;
    el.classList.remove('text-green', 'text-red');
    el.classList.add(isPositive ? 'text-green' : 'text-red');
}

function chartOptions(prefix) {
    return {
        responsive: true,
        plugins: { legend: { labels: { color: '#94a3b8' } } },
        scales: {
            x: { ticks: { color: '#64748b' }, grid: { color: '#1e293b' } },
            y: { ticks: { color: '#64748b', callback: v => prefix + v }, grid: { color: '#1e293b' } }
        }
    };
}

function renderBetsTable(bets) {
    if (!bets || bets.length === 0) return '<p class="text-muted text-center py-8">No bets.</p>';

    let html = `<table><thead><tr>
        <th>Time</th><th>Track</th><th>Horse</th>
        <th class="text-right">Pred BFSP</th><th class="text-right">Back</th>
        <th class="text-right">Edge</th><th class="text-right">Stake</th>
        <th>Status</th><th class="text-right">P&amp;L</th>
    </tr></thead><tbody>`;

    for (const b of bets) {
        html += `<tr>
            <td>${b.race_time}</td>
            <td>${b.track}</td>
            <td class="font-medium">${b.horse_name}</td>
            <td class="text-right">${b.predicted_bfsp ? b.predicted_bfsp.toFixed(2) : '-'}</td>
            <td class="text-right">${b.betfair_back ? b.betfair_back.toFixed(2) : '-'}</td>
            <td class="text-right ${b.edge_pct >= 15 ? 'text-green' : b.edge_pct >= 10 ? 'text-yellow' : ''}">${b.edge_pct ? '+' + b.edge_pct.toFixed(1) + '%' : '-'}</td>
            <td class="text-right">£${b.stake ? b.stake.toFixed(2) : '-'}</td>
            <td>${statusBadge(b.status)}</td>
            <td class="text-right ${b.net_pnl > 0 ? 'text-green' : b.net_pnl < 0 ? 'text-red' : ''}">
                ${b.net_pnl != null ? '£' + (b.net_pnl >= 0 ? '+' : '') + b.net_pnl.toFixed(2) : '-'}
            </td>
        </tr>`;
    }

    return html + '</tbody></table>';
}
