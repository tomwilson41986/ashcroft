"""The Phase 0 model condition (the owner's decision, 4 Nov 2026): beat the same-rows baseline, not a fixed number.

The build spec's fixed bar (log loss under 0.4379, the published logistic on required run rate) is cleared by our own
same-rows baseline as well, so it says nothing about the model. The condition now is:

- the calibrated model's log loss is below the baseline's on the same held-out rows, and the 95% interval of the
  difference, from resampling whole matches, lies wholly below zero;
- ECE under 0.131 (the spec's own figure, kept).

The fixed bar is still reported, for reference.
"""

from __future__ import annotations

import pandas as pd

from cricket.model.metrics import paired_bootstrap, score

ECE_MAX = 0.131
FIXED_BAR = 0.4379


def model_gate(test: pd.DataFrame, model: str = "p_cal", base: str = "p_base", group: str = "match_id") -> dict:
    bs = paired_bootstrap(test.bat_won, test[model], test[base], test[group])
    ece = score(test.bat_won, test[model])["ece"]
    checks = {"beats the same-rows baseline (95% interval of the difference below 0)": bs["hi"] < 0,
              f"ECE below {ECE_MAX}": ece < ECE_MAX}
    return {"bootstrap": bs, "ece": ece, "checks": checks, "pass": all(checks.values())}


def gate_lines(g: dict, what: str) -> list[str]:
    b = g["bootstrap"]
    out = [f"Phase 0 model condition ({what}): **{'PASS' if g['pass'] else 'NOT MET'}**", ""]
    out += [f"- {k}: **{'yes' if v else 'no'}**" for k, v in g["checks"].items()]
    out.append(f"- Log loss, calibrated model minus baseline: {b['diff']:+.4f} (95% interval {b['lo']:+.4f} to "
               f"{b['hi']:+.4f}; {b['groups']:,} matches resampled, {b['rows']:,} rows)")
    return out
