"""
Agent-based race simulation (Monte-Carlo) for feature generation.

    from model.abm import simulate_race, add_abm_features, ABM_FEATURES

See RESEARCH_FRAMEWORK.md §3 for the design and its data constraints.
"""

from model.abm.agents import AbilityConfig, FieldParams, build_field
from model.abm.calibrate import DEFAULT_PRIORS, PatternTargets, abc_smc, calibrate_interactions, observed_patterns, pattern_loss, simulated_patterns
from model.abm.energy import EnergyConfig
from model.abm.features import ABM_FEATURES, add_abm_features, load_abm_features, merge_abm_features, simulate_race, summarise
from model.abm.simulate import RaceSimulator, SimConfig, SimResult
from model.abm.track import Track

__all__ = [
    "AbilityConfig", "FieldParams", "build_field", "EnergyConfig", "Track",
    "RaceSimulator", "SimConfig", "SimResult", "ABM_FEATURES", "add_abm_features",
    "load_abm_features", "merge_abm_features", "simulate_race", "summarise",
    "DEFAULT_PRIORS", "PatternTargets", "abc_smc", "calibrate_interactions",
    "observed_patterns", "pattern_loss", "simulated_patterns",
]
