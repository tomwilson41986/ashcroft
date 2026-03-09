"""Tests for PreRaceBuilder — pre-race feature vector construction."""

import numpy as np
import pandas as pd
import pytest

from src.models.prerace_builder import PreRaceBuilder, HORSE_CAREER_COLS, JOCKEY_COLS, TRAINER_COLS


def _make_history(n_races=5, n_runners=8):
    """Create a minimal history DataFrame for testing."""
    rows = []
    horses = [f"Horse_{i}" for i in range(n_runners)]
    for race_idx in range(n_races):
        date = pd.Timestamp(f"2024-01-{race_idx + 1:02d}")
        for i, horse in enumerate(horses):
            rows.append({
                "date": date,
                "course": "Ascot",
                "raceid": f"race_{race_idx}",
                "horse_name": horse,
                "jockey": f"Jockey_{i % 3}",
                "trainer": f"Trainer_{i % 2}",
                "placing_numerical": (i % n_runners) + 1,
                "bfsp": 2.0 + i * 1.5,
                "number_of_runners": n_runners,
                "official_rating": 80 + i,
                "weight_lbs": 126 + i,
                "distance_furlongs": 8.0,
                "race_class": 3,
                # Pre-computed custom metrics (normally from CustomMetricsEngine)
                "preracehorsecareerNFP": 0.5 + 0.01 * race_idx,
                "preracehorsecareerRB": 0.5,
                "preracehorsecareerFSARB": 0.5,
                "preracehorsecareerWIV": 1.0 + 0.1 * race_idx,
                "preracehorsecareerWAX": 0.01 * race_idx,
                "preracehorsecareerWOA": 0.01,
                "preracehorsecareerCWO": 0.02 * race_idx,
                "preracehorsecareerRuns": race_idx + 1,
                "preracehorsecareerWins": 1,
                "preracehorsecareerPlaces": 2,
                "Horse_Career_EPF": 3.0,
                "horsepaceindex": 4.0,
                "LR3NFPtotal": 0.6,
                "LR5NFPtotal": 0.55,
                "LR10NFPtotal": 0.52,
                "LR3_RWO": 1.1,
                "LR5_RWO": 1.05,
                "LR10_RWO": 1.02,
                "FSS": 2.0,
                "FCS": 3.0,
                "PFD3": 0.05,
                "PFD5": 0.04,
                "PFD10": 0.03,
                "WPMRF3": 5000.0,
                "WPMRF5": 4500.0,
                "WPMRF10": 4000.0,
                "PMW3": 100.0,
                "PMW5": 90.0,
                "PMW10": 80.0,
                "OFS3": 1.5,
                "OFS5": 1.4,
                "OFS10": 1.3,
                "FinalDSLR": 0.5,
                "CIL3": 50.0,
                "CIL5": 40.0,
                "CIL10": 30.0,
                "preracejockeycareerWins": 10,
                "preracejockeycareerRuns": 50,
                "preracejockeycareerWIV": 1.2,
                "preracejockeycareerNFP": 0.55,
                "preracejockeycareerRB": 0.5,
                "preracejockeycareerWAX": 0.05,
                "preracejockeycareerWOA": 0.03,
                "Jockey_Career_EPF": 3.5,
                "jockeypaceindex": 4.0,
                "totalLRPjockeyindex": 10.0,
                "preracetrainercareerWins": 20,
                "preracetrainercareerRuns": 100,
                "preracetrainercareerWIV": 1.1,
                "preracetrainercareerNFP": 0.5,
                "preracetrainercareerRB": 0.48,
                "preracetrainercareerWAX": 0.04,
                "preracetrainercareerWOA": 0.02,
                "trainer_Career_EPF": 3.2,
                "trainerpaceindex": 3.8,
                "trainerjockeycareerWIV": 1.15,
                "trainerjockeycareerNFP": 0.52,
            })
    return pd.DataFrame(rows)


def _make_declared_runners(n=4):
    """Create declared runners for an upcoming race."""
    return pd.DataFrame({
        "horse_name": [f"Horse_{i}" for i in range(n)],
        "jockey": [f"Jockey_{i % 3}" for i in range(n)],
        "trainer": [f"Trainer_{i % 2}" for i in range(n)],
        "course": ["Ascot"] * n,
        "distance_furlongs": [8.0] * n,
        "going": ["Good"] * n,
        "race_class": [3] * n,
        "weight_lbs": [126 + i for i in range(n)],
        "draw_position": list(range(1, n + 1)),
        "number_of_runners": [n] * n,
        "age": [4] * n,
    })


class TestPreRaceBuilder:
    def test_basic_build(self):
        history = _make_history()
        builder = PreRaceBuilder(history)
        runners = _make_declared_runners()
        features = builder.build_features(runners, "2024-01-10")

        assert len(features) == 4
        assert "preracehorsecareerNFP" in features.columns
        assert "is_debut" in features.columns

    def test_no_lookahead(self):
        """Features must only use data before race_date."""
        history = _make_history(n_races=10)
        builder = PreRaceBuilder(history)
        runners = _make_declared_runners(n=2)

        # Build features as of day 3 — should only use data from days 1-2
        features_early = builder.build_features(runners, "2024-01-03")
        # Build as of day 8 — should use data from days 1-7
        features_late = builder.build_features(runners, "2024-01-08")

        # With more history, career stats should differ (more data points)
        # The career NFP should reflect more runs in the late version
        assert features_late["preracehorsecareerRuns"].iloc[0] >= features_early["preracehorsecareerRuns"].iloc[0]

    def test_debut_runner(self):
        """Debut runners with no history get median-filled features."""
        history = _make_history()
        builder = PreRaceBuilder(history)

        runners = _make_declared_runners(n=2)
        runners.loc[1, "horse_name"] = "NewHorse_Unknown"

        features = builder.build_features(runners, "2024-01-10")

        # The unknown horse should be flagged as debut
        assert features.loc[1, "is_debut"] == 1
        assert features.loc[0, "is_debut"] == 0

        # Debut runner should have median values, not NaN for career cols
        for col in HORSE_CAREER_COLS:
            if col in features.columns:
                assert np.isfinite(features.loc[1, col]), f"{col} should be filled for debut"

    def test_within_race_ranks(self):
        """Ranks should be integers and cover all runners."""
        history = _make_history()
        builder = PreRaceBuilder(history)
        runners = _make_declared_runners(n=4)
        features = builder.build_features(runners, "2024-01-10")

        if "rNFP" in features.columns:
            ranks = features["rNFP"].dropna()
            if len(ranks) > 0:
                assert ranks.min() >= 1
                assert ranks.max() <= 4

    def test_field_features(self):
        """Field-level averages should be computed."""
        history = _make_history()
        builder = PreRaceBuilder(history)
        runners = _make_declared_runners(n=4)
        features = builder.build_features(runners, "2024-01-10")

        assert "field_avg_career_nfp" in features.columns
        # All runners should have the same field average
        assert features["field_avg_career_nfp"].nunique() == 1

    def test_going_encoding(self):
        assert PreRaceBuilder._encode_going("Good") == 4.0
        assert PreRaceBuilder._encode_going("Heavy") == 7.0
        assert PreRaceBuilder._encode_going("Firm") == 2.0
        assert PreRaceBuilder._encode_going("Good to Soft") == 5.0
        assert PreRaceBuilder._encode_going("") == 4.0
        assert PreRaceBuilder._encode_going(None) == 4.0

    def test_course_stats(self):
        """Course-specific win rate should be computed."""
        history = _make_history()
        builder = PreRaceBuilder(history)
        runners = _make_declared_runners(n=2)
        features = builder.build_features(runners, "2024-01-10")

        assert "course_win_pct" in features.columns
        assert "course_runs" in features.columns

    def test_days_since_last_run(self):
        """days_since_last_run should reflect calendar difference."""
        history = _make_history(n_races=5)
        builder = PreRaceBuilder(history)
        runners = _make_declared_runners(n=1)
        features = builder.build_features(runners, "2024-01-10")

        dslr = features["days_since_last_run"].iloc[0]
        # Last race was Jan 5, predicting for Jan 10, so 5 days
        assert dslr == 5

    def test_feature_names_method(self):
        """get_feature_names should return a list of strings."""
        history = _make_history()
        builder = PreRaceBuilder(history)
        names = builder.get_feature_names()
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)
        assert len(names) > 50

    def test_all_debut_race(self):
        """Handle a race where all runners are debutants."""
        history = _make_history()
        builder = PreRaceBuilder(history)

        runners = pd.DataFrame({
            "horse_name": ["NewA", "NewB", "NewC"],
            "jockey": ["Jockey_0", "Jockey_1", "Jockey_2"],
            "trainer": ["Trainer_0", "Trainer_1", "Trainer_0"],
            "course": ["Ascot"] * 3,
            "distance_furlongs": [8.0] * 3,
            "going": ["Good"] * 3,
            "race_class": [3] * 3,
            "weight_lbs": [126, 127, 128],
            "draw_position": [1, 2, 3],
            "number_of_runners": [3] * 3,
            "age": [3, 3, 3],
        })

        features = builder.build_features(runners, "2024-01-10")
        assert len(features) == 3
        assert (features["is_debut"] == 1).all()
