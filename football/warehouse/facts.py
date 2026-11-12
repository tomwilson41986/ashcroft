"""The warehouse's headline numbers, from the tables as they stand: what docs/football_data_platform.md cites.

    python -m football.warehouse facts [--out facts.json] [--root <local folder>]

Every figure is one query over the catalog's views (football.warehouse.catalog), so a number in the document can be
traced to its query here and re-run on any copy of the warehouse (the capture bucket in the workflow, a local folder
on a laptop). A table that is not built is left out, not guessed at.
"""

from __future__ import annotations


def _round(v, n: int = 4):
    return None if v is None else round(float(v), n)


def facts(store) -> dict:
    """{section: figures}; sections whose tables are missing are absent."""
    from football.warehouse import catalog
    con = catalog.connect(store)
    have = set(catalog.tables(store))
    out: dict = {}

    def one(sql: str) -> tuple:
        return con.sql(sql).fetchone()

    def rows(sql: str) -> list[tuple]:
        return con.sql(sql).fetchall()

    out["rows"] = {v: one(f"SELECT count(*) FROM {v}")[0] for v in sorted(have)}

    if "tick" in have:
        r = one("""SELECT count(*), sum(pre_off::INT), sum(in_play::INT), sum((NOT pre_off AND NOT in_play)::INT)
                   FROM tick""")
        out["ticks"] = dict(zip(("all", "pre_off", "in_play", "after_the_off_not_in_play"), r))
    if "market" in have:
        r = one("""SELECT count(*), sum((in_play_pt IS NOT NULL)::INT), sum((in_play_pt IS NULL)::INT),
                          sum((n_inplay_turns > 1)::INT), sum((market_time <> market_time_pre)::INT),
                          sum((market_time_pre <> market_time_first)::INT), sum(n_ltp_invalid),
                          sum((in_play_pt IS NULL AND n_ticks > n_ticks_pre)::INT)
                   FROM market""")
        out["markets"] = dict(zip(("all", "turned_in_play", "never_in_play", "turned_in_play_more_than_once",
                                   "kickoff_time_rewritten_at_or_after_the_off", "rescheduled_before_the_off",
                                   "price_updates_outside_betfairs_range",
                                   "never_in_play_but_traded_after_the_scheduled_off"), r))
        out["markets_by_type"] = dict(rows("SELECT market_type, count(*) FROM market GROUP BY 1 ORDER BY 1"))
    if "event" in have:
        r = one("""SELECT count(*), sum(went_in_play::INT), sum((result IS NOT NULL)::INT), sum(void::INT),
                          sum((close_legs = 3)::INT),
                          sum((mo_market_id IS NOT NULL AND ou_market_id IS NOT NULL
                               AND ah_market_id IS NOT NULL)::INT),
                          sum((country_code IS NULL)::INT), sum((kickoff_shift_s <> 0)::INT),
                          sum((inplay_turns > 1)::INT), min(kickoff), max(kickoff),
                          quantile_cont(kickoff_delay_s, 0.5), quantile_cont(kickoff_delay_s, 0.9),
                          quantile_cont(kickoff_delay_s, 0.99)
                   FROM event""")
        out["events"] = dict(zip(("all", "went_in_play", "with_a_result", "void", "full_closing_1x2",
                                  "all_three_markets", "no_country_code", "rescheduled_before_the_off",
                                  "turned_in_play_more_than_once"), r[:9]))
        out["events"].update({"first_kickoff": str(r[9]), "last_kickoff": str(r[10]),
                              "kickoff_delay_s": {"median": _round(r[11], 0), "p90": _round(r[12], 0),
                                                  "p99": _round(r[13], 0)}})
        out["events_by_category"] = dict(rows("SELECT category, count(*) FROM event GROUP BY 1 ORDER BY 2 DESC"))
        r = one("""SELECT count(*), avg(p_home), avg((result = 'H')::INT), avg(p_draw), avg((result = 'D')::INT),
                          avg(p_away), avg((result = 'A')::INT)
                   FROM event WHERE close_legs = 3 AND result IS NOT NULL AND close_age_s < 600""")
        out["closing_1x2_against_results"] = {
            "matches_with_a_fresh_full_closing_book": r[0],
            **{k: _round(v) for k, v in zip(("p_home", "home_rate", "p_draw", "draw_rate", "p_away", "away_rate"),
                                            r[1:])}}
    if "checkpoint" in have:
        out["match_odds_home_by_checkpoint"] = {
            int(cp): {"rows": n, "book_complete": _round(c), "schedule_known": _round(k)}
            for cp, n, c, k in rows("""SELECT cp_min, count(*), avg(book_complete::INT), avg(schedule_known::INT)
                                       FROM checkpoint WHERE market_type = 'MATCH_ODDS' AND role = 'home'
                                       GROUP BY 1 ORDER BY 1 DESC""")}
        r = one("SELECT count(*), sum((NOT schedule_known)::INT) FROM checkpoint WHERE cp_min > 0")
        out["checkpoint_rows_before_the_close"] = {"rows": r[0], "schedule_not_known": r[1]}
    if "implied" in have:
        r = one("""SELECT count(*), sum((total IS NOT NULL AND supremacy IS NOT NULL)::INT), avg(total),
                          avg(supremacy), quantile_cont(rho, 0.5),
                          sum((rho IS NULL AND total IS NOT NULL AND supremacy IS NOT NULL
                               AND p_draw IS NOT NULL)::INT), avg(p_over)
                   FROM implied WHERE cp_min = 0""")
        out["implied_at_the_close"] = {"events": r[0], "with_goal_expectancies": r[1], "total_mean": _round(r[2]),
                                       "supremacy_mean": _round(r[3]), "rho_median": _round(r[4]),
                                       "books_no_one_market_could_make": r[5], "p_over_mean": _round(r[6])}
        fresh = "mo_age_s < 900 AND ou_age_s < 900 AND ah_age_s < 900 AND sup_gap IS NOT NULL"
        r = one(f"""SELECT count(*), stddev(sup_gap), quantile_cont(sup_gap, 0.5) FROM implied
                    WHERE cp_min = 60 AND {fresh}""")
        out["handicap_against_1x2_at_t60_fresh_books"] = {
            "rows": r[0], "sd_goals": _round(r[1]), "median_goals": _round(r[2]),
            "median_by_line": {str(line): {"rows": n, "median_goals": _round(m)} for line, n, m in rows(
                f"""SELECT ah_line, count(*), quantile_cont(sup_gap, 0.5) FROM implied
                    WHERE cp_min = 60 AND {fresh} AND ah_line IN (-0.5, 0.0, 0.5) GROUP BY 1 ORDER BY 1""")}}
    if "move" in have:
        r = one("""SELECT count(*), sum(cp_fresh::INT), sum(usable::INT), sum((NOT schedule_known)::INT)
                   FROM move WHERE market_type = 'MATCH_ODDS' AND cp_min IN (360, 120, 60, 15)""")
        out["move_rows_the_model_reads"] = dict(zip(("rows", "cp_fresh", "usable", "schedule_not_known"), r))
        out["match_odds_home_at_t60_by_tier"] = {
            t: {"rows": n, "usable": u} for t, n, u in rows(
                """SELECT tier, count(*), sum(usable::INT) FROM move
                   WHERE market_type = 'MATCH_ODDS' AND role = 'home' AND cp_min = 60 GROUP BY 1 ORDER BY 1""")}
    if "timeline" in have:
        r = one("""SELECT count(*), sum(usable::INT), sum((usable AND (consistent OR reconciled))::INT),
                          sum((usable AND consistent)::INT), sum(n_goals), sum(n_unplaced) FROM timeline""")
        out["inferred_timelines"] = dict(zip(("matches", "usable", "usable_and_score_agrees_with_the_settlement",
                                              "usable_and_the_prices_alone_agree", "goals",
                                              "goals_without_a_minute"), r))
    if "inplay" in have and "inplay_minute" in have:
        r = one("""SELECT count(*), sum(filled::INT), count(perf_home), avg(perf_home), avg(perf_away),
                          stddev(perf_home) FROM inplay""")
        out["in_play"] = {"matches": r[0], "with_a_summary": r[1], "with_a_performance": r[2],
                          "perf_home_mean": _round(r[3]), "perf_away_mean": _round(r[4]), "perf_sd": _round(r[5])}
        r = one("SELECT count(*), sum(ok::INT), sum(score_ok::INT) FROM inplay_minute")
        out["in_play"].update({"minute_rows": r[0], "minutes_ok": r[1], "minutes_score_ok": r[2]})
        if "event" in have:
            out["in_play"]["chances_against_results_by_minute"] = {
                int(m): {"rows": n, "p_home": _round(a), "home_rate": _round(b), "p_draw": _round(c),
                         "draw_rate": _round(d)}
                for m, n, a, b, c, d in rows(
                    """SELECT mi.minute, count(*), avg(mi.p_home), avg((e.result = 'H')::INT), avg(mi.p_draw),
                              avg((e.result = 'D')::INT)
                       FROM inplay_minute mi JOIN event e USING (event_id)
                       WHERE mi.ok AND e.result IS NOT NULL AND mi.minute IN (15, 30, 75, 90, 105)
                       GROUP BY 1 ORDER BY 1""")}
    if "team_rating" in have:
        r = one("SELECT count(*), sum((n_prior >= 1)::INT), sum(rated::INT), max(pool_teams) FROM team_rating")
        out["ratings"] = dict(zip(("team_match_rows", "with_a_rating_of_their_own", "closes_that_rate",
                                   "largest_pool_teams"), r))
    if "rating_gap" in have:
        r = one("SELECT count(*), avg(linked::INT) FROM rating_gap")
        out.setdefault("ratings", {}).update({"gap_rows": r[0], "gap_rows_linked_share": _round(r[1])})
    if "team_summary" in have:
        r = one("SELECT count(*), sum(ranked::INT), max(pool_size) FROM team_summary")
        out["club"] = {"teams": r[0], "ranked": r[1], "largest_pool_teams": r[2],
                       "ranked_by_category": dict(rows("""SELECT category, count(*) FROM team_summary WHERE ranked
                                                          GROUP BY 1 ORDER BY 2 DESC""")),
                       "strongest_senior_sides": [f"{n} {s:.2f}" for n, s in rows(
                           """SELECT team_name, strength FROM team_summary WHERE ranked AND category = 'senior'
                              ORDER BY strength DESC, team_id LIMIT 12""")]}
    if "country_strength" in have:
        out.setdefault("club", {})["strongest_countries_senior_top5"] = [f"{c} {s:.2f}" for c, s in rows(
            """SELECT country_code, strength_top5 FROM country_strength
               WHERE category = 'senior' AND strength_top5 IS NOT NULL ORDER BY strength_top5 DESC LIMIT 10""")]
    if "team_match" in have:
        r = one("""SELECT count(*), avg(pts_vs_xpts), stddev(pts_vs_xpts) / sqrt(count(*)), stddev(pts_vs_xpts),
                          stddev(perf), stddev(luck)
                   FROM team_match WHERE points IS NOT NULL AND xpts_pre IS NOT NULL""")
        out.setdefault("club", {})["points_against_expectation"] = {
            "team_matches": r[0], "mean": _round(r[1]), "se": _round(r[2]), "sd": _round(r[3], 3),
            "sd_performance": _round(r[4], 3), "sd_luck": _round(r[5], 3)}
        out["club"]["what_forecasts_a_teams_later_points"] = _points_forecast(con)
        # the performance measure by the pre-match expectation, on the months its line was not fitted on: raw is
        # xpts_inplay - xpts_pre, which leans against strong sides; perf is measured from the fitted line
        out["club"]["performance_by_expected_points_even_months"] = {
            f"{lo:g} to {lo + 0.5:g}": {"team_matches": n, "raw": _round(raw), "perf": _round(perf),
                                       "perf_se": _round(se)}
            for lo, n, raw, perf, se in rows(
                """SELECT floor(xpts_pre / 0.5) * 0.5, count(*), avg(xpts_inplay - xpts_pre), avg(perf),
                          stddev(perf) / sqrt(count(*))
                   FROM team_match WHERE perf IS NOT NULL AND month(kickoff) % 2 = 0 GROUP BY 1 ORDER BY 1""")}
    if "link" in have:
        r = one("SELECT count(*), sum(linked::INT) FROM link")
        out["link"] = {"events": r[0], "linked": r[1], "by_method": dict(rows(
            "SELECT method, count(*) FROM link WHERE linked GROUP BY 1 ORDER BY 2 DESC"))}
        r = one("""SELECT sum((result_agrees IS NOT NULL)::INT), sum(result_agrees::INT),
                          sum((over25_agrees IS NOT NULL)::INT), sum(over25_agrees::INT) FROM link WHERE linked""")
        out["link"]["settlement_against_the_fixtures_score"] = dict(zip(
            ("result_known", "result_agrees", "over25_known", "over25_agrees"), r))
    if "goal_check" in have:
        r = one("""SELECT count(*), sum(score_agrees::INT), sum((score_agrees IS NOT NULL)::INT),
                          sum(usable::INT), sum((usable AND score_agrees)::INT) FROM goal_check""")
        out["inferred_goals_against_real_ones"] = dict(zip(
            ("matches", "score_agrees", "score_known", "usable", "usable_and_score_agrees"), r))
    con.close()
    return out


def _points_forecast(con) -> dict:
    """For teams with 30 or more matches: the first half of a team's matches against the second. What goes with
    its later points: its points so far, or what the market expected of it?"""
    import numpy as np
    d = con.sql("""SELECT team_id, kickoff, event_id, points, xpts_pre, perf, luck FROM team_match
                   WHERE points IS NOT NULL AND xpts_pre IS NOT NULL""").df()
    d = d.sort_values(["team_id", "kickoff", "event_id"])
    d["k"] = d.groupby("team_id").cumcount()
    d["n"] = d.groupby("team_id")["k"].transform("size")
    d = d[d["n"] >= 30]
    if d["team_id"].nunique() < 50:
        return {"teams": int(d["team_id"].nunique())}
    d["half"] = (d["k"] >= d["n"] / 2).astype(int)
    h = d.pivot_table(index="team_id", columns="half", values=["points", "xpts_pre", "perf", "luck"], aggfunc="mean")
    X = np.c_[np.ones(len(h)), h["points"][0], h["xpts_pre"][0]]
    y = h["points"][1].to_numpy()
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return {"teams": int(len(h)),
            "corr_points_so_far_with_later_points": _round(h["points"][0].corr(h["points"][1]), 3),
            "corr_expected_points_so_far_with_later_points": _round(h["xpts_pre"][0].corr(h["points"][1]), 3),
            "later_points_on_both": {"points_so_far": _round(b[1], 3), "expected_points_so_far": _round(b[2], 3)},
            "performance_first_half_with_second": _round(h["perf"][0].corr(h["perf"][1]), 3),
            "luck_first_half_with_second": _round(h["luck"][0].corr(h["luck"][1]), 3)}
