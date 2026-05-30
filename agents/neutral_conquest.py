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

from collections import namedtuple

PredictedPlanet = namedtuple("PredictedPlanet", ["x", "y"])

CENTER_X = 50.0
CENTER_Y = 50.0
ROTATION_RADIUS_LIMIT = 50.0
MAX_SPEED = 6.0


def fleet_speed(ships):
    if ships <= 1:
        return 1.0

    return 1.0 + (MAX_SPEED - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5


def is_orbiting(planet):
    dx = planet.x - CENTER_X
    dy = planet.y - CENTER_Y
    orbital_radius = math.sqrt(dx * dx + dy * dy)

    return orbital_radius + planet.radius < ROTATION_RADIUS_LIMIT


def rotate_point_around_center(x, y, radians):
    dx = x - CENTER_X
    dy = y - CENTER_Y

    cos_a = math.cos(radians)
    sin_a = math.sin(radians)

    new_x = CENTER_X + dx * cos_a - dy * sin_a
    new_y = CENTER_Y + dx * sin_a + dy * cos_a

    return new_x, new_y


def predict_planet_position(planet, turns_ahead, angular_velocity):
    if not is_orbiting(planet):
        return PredictedPlanet(planet.x, planet.y)

    future_angle = angular_velocity * turns_ahead
    future_x, future_y = rotate_point_around_center(
        planet.x,
        planet.y,
        future_angle
    )

    return PredictedPlanet(future_x, future_y)


def predicted_angle(source, target, send_ships, angular_velocity):
    speed = fleet_speed(send_ships)

    current_distance = dist(source, target)
    turns_to_arrive = current_distance / speed

    predicted_target = predict_planet_position(
        target,
        turns_to_arrive,
        angular_velocity
    )

    predicted_distance = math.dist(
        (source.x, source.y),
        (predicted_target.x, predicted_target.y)
    )

    turns_to_arrive = predicted_distance / speed

    predicted_target = predict_planet_position(
        target,
        turns_to_arrive,
        angular_velocity
    )

    return angle(source, predicted_target)

def angle(start_planet, target_planet):
    dx = target_planet.x - start_planet.x
    dy = target_planet.y - start_planet.y

    angle = math.atan2(dy, dx)

    return angle

def score_attack(source, target, distance, send_ships):
    score = 0

    if target.owner == -1:
        score += 40
    else:
        score += 15

    score += target.production * 25

    score += target.radius * 10

    score -= target.ships * 2

    score -= distance * 2

    score -= send_ships * 0.3

    return score


def dist(start_planet, target_planet):
    return math.dist((start_planet.x, start_planet.y), (target_planet.x, target_planet.y))

def score_neutral(start_planet, target_planet):
    distance = dist(start_planet, target_planet)
    needed_ships = target_planet.ships + 1

    return (
        target_planet.production * 3
        + target_planet.radius * 5
        - needed_ships * 5
        - distance * 0.7
    )


def agent(obs):
    moves = []

    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    angular_velocity = obs.get("angular_velocity", 0) if isinstance(obs, dict) else obs.angular_velocity

    planets = [Planet(*p) for p in raw_planets]

    p_neutral = [p for p in planets if p.owner == -1]
    p_enemy = [p for p in planets if p.owner != player and p.owner != -1]
    p_mine = [p for p in planets if p.owner == player]

    if len(p_mine) == 0:
        return []

    targets = p_neutral if len(p_neutral) > 0 else p_enemy

    best_move = None
    best_score = float("-inf")

    for target in targets:
        for source in p_mine:
            distance = dist(source, target)
            needed_ships = target.ships + 1

            if source.ships <= needed_ships + 5:
                continue

            send_ships = min(
                int(source.ships * 0.6),
                needed_ships + 15
            )

            if send_ships <= needed_ships:
                continue

            score = score_attack(source, target, distance, send_ships)

            if score > best_score:
                best_score = score

                aim_angle = predicted_angle(
                    source,
                    target,
                    send_ships,
                    angular_velocity
                )

                best_move = [source.id, aim_angle, send_ships]

    if best_move is not None:
        moves.append(best_move)

    return moves