"""
Orbit Wars - Nearest Planet Sniper Agent

A simple agent that captures the nearest unowned planet when it has
enough ships to guarantee the takeover.

Strategy:
  For each planet we own, find the closest planet we don't own.
  If we have more ships than the target's garrison, send exactly
  enough to capture it (garrison + 1). Otherwise, wait and accumulate.

Key concepts demonstrated:
  - Parsing the observation (planets, player ID)
  - Computing angles with atan2 for fleet direction
  - Sending moves as [from_planet_id, angle, num_ships]
"""

import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet


def angle(start_planet, target_planet):
    dx = target_planet.x - start_planet.x
    dy = target_planet.y - start_planet.y

    angle = math.atan2(dy, dx)

    return angle


def dist(start_planet, target_planet):
    return math.dist((start_planet.x, start_planet.y), (target_planet.x, target_planet.y))


def agent(obs):
    moves = []

    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets

    planets = [Planet(*p) for p in raw_planets]
    p_neutral = [p for p in planets if p.owner == -1]
    p_enemy = [p for p in planets if p.owner != player and p.owner != -1]
    p_mine = [p for p in planets if p.owner == player]

    if len(p_mine) == 0:
        return []

    targets = p_neutral if len(p_neutral) > 0 else p_enemy

    for p_n in targets:
        rec_fleet = p_n.ships + 1

        nearest_planet = None
        min_dist = float("inf")

        for p_m in p_mine:
            dist_between = dist(p_n, p_m)

            if dist_between < min_dist:
                min_dist = dist_between
                nearest_planet = p_m

        if nearest_planet is None:
            continue

        if nearest_planet.ships <= rec_fleet + 3:
            continue

        moves.append([nearest_planet.id, angle(nearest_planet, p_n), rec_fleet])
    #  if(p_mine[0].ships > rec_fleet):
    #      moves.append([p_mine[0].id, angle(p_mine[0], p), rec_fleet])

    if len(p_mine) == 0:
        return moves

    if len(p_neutral) == 0:
        return moves

    # moves.append([my_planets[0].id, 1, 10])
    # moves.append([p_mine[0].id, angle(p_mine[0], p_neutral[0]), 10])

    return moves