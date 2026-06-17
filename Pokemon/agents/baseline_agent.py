"""Baseline agent stub for competition submission."""

from __future__ import annotations

import random
from typing import Any


def random_agent(obs_dict: dict) -> list[int]:
    """Select random legal options (same idea as sample notebook)."""
    from cg.api import to_observation_class

    obs = to_observation_class(obs_dict)
    n = len(obs.select.option)
    k = obs.select.maxCount
    return random.sample(list(range(n)), k)


def select_action(obs_dict: dict, your_deck: list[int] | None = None) -> list[int]:
    """Entry point called by submission harness."""
    _ = your_deck
    return random_agent(obs_dict)


def run_self_check() -> None:
    """Smoke test when cg-lib is available."""
    from src.utils import setup_cg_lib

    setup_cg_lib()
    from cg.game import battle_finish, battle_select, battle_start

    from src.config import SAMPLE_DECK

    obs, start = battle_start(SAMPLE_DECK, SAMPLE_DECK)
    if start.errorPlayer >= 0:
        raise RuntimeError(f"Deck error: player={start.errorPlayer} type={start.errorType}")

    steps = 0
    while obs["current"]["result"] < 0 and steps < 500:
        obs = battle_select(select_action(obs, SAMPLE_DECK))
        steps += 1

    battle_finish()
    print(f"Self-check finished in {steps} steps, result={obs['current']['result']}")


if __name__ == "__main__":
    run_self_check()
