/* Ashcroft Web Dashboard — minimal JS helpers */

// Format numbers with sign
function formatPnl(value) {
    const sign = value >= 0 ? '+' : '';
    return sign + '£' + value.toFixed(2);
}

// Auto-scroll to first bet row
document.addEventListener('DOMContentLoaded', function() {
    const betRow = document.querySelector('.bet-row');
    if (betRow) {
        betRow.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
});
