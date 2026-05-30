"""
Orbit Wars - ROI Expansion Agent

A simple, submission-safe heuristic agent:
1. Expand into high-production low-garrison targets first.
2. Avoid direct shots crossing the central sun.
3. Keep a reserve on every owned planet.
4. Do not over-commit multiple sources to the same target unless needed.

Observation format used by Orbit Wars:
planet = [id, owner, x, y, radius, ships, production]
fleet  = [id, owner, x, y, angle, from_planet_id, ships]
move   = [from_planet_id, angle_radians, num_ships]
"""

import math

PID = 0
OWNER = 1
X = 2
Y = 3
RADIUS = 4
SHIPS = 5
PROD = 6

CENTER_X = 50.0
CENTER_Y = 50.0
SUN_RADIUS = 10.0


def distance(a, b):
    return math.hypot(a[X] - b[X], a[Y] - b[Y])


def angle_to(a, b):
    return math.atan2(b[Y] - a[Y], b[X] - a[X])


def segment_distance_to_point(ax, ay, bx, by, px, py):
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    denom = abx * abx + aby * aby
    t = (apx * abx + apy * aby) / denom
    t = max(0.0, min(1.0, t))
    cx = ax + t * abx
    cy = ay + t * aby
    return math.hypot(px - cx, py - cy)


def crosses_sun(source, target):
    # Add a small margin because fleets spawn near planet edge and collision uses continuous segments.
    d = segment_distance_to_point(source[X], source[Y], target[X], target[Y], CENTER_X, CENTER_Y)
    return d <= SUN_RADIUS + 1.0


def reserve_for(planet):
    # Keep enough local defense to avoid emptying productive planets.
    return max(8, int(planet[PROD] * 5 + planet[RADIUS] * 2))


def ships_needed_for(target):
    # Enemy planets get extra safety because production may happen before arrival.
    margin = 2 if target[OWNER] == -1 else max(5, int(target[PROD] * 4))
    return int(target[SHIPS] + 1 + margin)


def target_score(source, target):
    d = distance(source, target)
    need = ships_needed_for(target)
    value = target[PROD] * 25.0 + target[RADIUS] * 2.0
    ownership_bonus = 8.0 if target[OWNER] == -1 else 0.0
    return (value + ownership_bonus) / (need + d * 0.35)


def best_target(source, targets, committed):
    candidates = [
        t for t in targets
        if committed.get(t[PID], 0) < ships_needed_for(t)
        and not crosses_sun(source, t)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda t: target_score(source, t))


def agent(obs):
    player = obs["player"]
    planets = obs["planets"]

    my_planets = [p for p in planets if p[OWNER] == player]
    targets = [p for p in planets if p[OWNER] != player]

    moves = []
    committed = {}

    for source in sorted(my_planets, key=lambda p: p[SHIPS], reverse=True):
        available = int(source[SHIPS] - reserve_for(source))
        if available <= 0:
            continue

        target = best_target(source, targets, committed)
        if target is None:
            continue

        need = ships_needed_for(target) - committed.get(target[PID], 0)
        send = min(available, need)

        if send > 0:
            moves.append([int(source[PID]), angle_to(source, target), int(send)])
            committed[target[PID]] = committed.get(target[PID], 0) + send

    return moves
