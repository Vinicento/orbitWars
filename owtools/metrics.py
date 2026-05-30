from __future__ import annotations

from pathlib import Path

import pandas as pd

# Orbit Wars observation indexes
PID = 0
OWNER = 1
SHIPS = 5
PROD = 6
F_OWNER = 1
F_SHIPS = 6


def observation_from_step(step):
    return step[0].observation


def rows_from_env(env) -> pd.DataFrame:
    """Return one row per turn and player."""
    rows = []
    player_count = len(env.steps[-1])

    for turn, step in enumerate(env.steps):
        obs = observation_from_step(step)
        planets = obs["planets"]
        fleets = obs.get("fleets", [])

        for player in range(player_count):
            owned_planets = [p for p in planets if p[OWNER] == player]
            owned_fleets = [f for f in fleets if f[F_OWNER] == player]
            planet_ships = sum(p[SHIPS] for p in owned_planets)
            fleet_ships = sum(f[F_SHIPS] for f in owned_fleets)
            production = sum(p[PROD] for p in owned_planets)

            rows.append(
                {
                    "turn": turn,
                    "player": player,
                    "planet_count": len(owned_planets),
                    "fleet_count": len(owned_fleets),
                    "planet_ships": planet_ships,
                    "fleet_ships": fleet_ships,
                    "total_ships": planet_ships + fleet_ships,
                    "production": production,
                }
            )

    return pd.DataFrame(rows)


def final_rows_from_env(env, seed: int) -> pd.DataFrame:
    """Return final score table, one row per player."""
    final_step = env.steps[-1]
    per_turn = rows_from_env(env)
    last = per_turn[per_turn["turn"] == per_turn["turn"].max()].copy()

    rewards = [state.reward for state in final_step]
    statuses = [state.status for state in final_step]

    last["seed"] = seed
    last["reward"] = last["player"].map(dict(enumerate(rewards)))
    last["status"] = last["player"].map(dict(enumerate(statuses)))
    last["rank"] = last["reward"].rank(method="min", ascending=False).astype(int)
    return last


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
