import math
from collections import namedtuple
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

CENTER_X = 50.0
CENTER_Y = 50.0
SUN_RADIUS = 10.0
BOARD_SIZE = 100.0
ROTATION_RADIUS_LIMIT = 50.0
MAX_SPEED = 6.0

Point = namedtuple("Point", ["x", "y"])
Route = namedtuple("Route", ["valid", "angle", "turns", "reason"])
Proposal = namedtuple("Proposal", ["kind", "source", "target", "ships", "score", "route"])


def clamp(value, low, high):
    return max(low, min(high, value))


def distance(a, b):
    return math.dist((a.x, a.y), (b.x, b.y))


def angle_between(start, target):
    return math.atan2(target.y - start.y, target.x - start.x)


def fleet_speed(ships):
    ships = max(1, min(int(ships), 1000))

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
        return Point(planet.x, planet.y)

    future_angle = angular_velocity * turns_ahead
    future_x, future_y = rotate_point_around_center(planet.x, planet.y, future_angle)

    return Point(future_x, future_y)


def segment_distance_to_point(ax, ay, bx, by, px, py):
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay

    ab_len_sq = abx * abx + aby * aby

    if ab_len_sq == 0:
        return math.dist((ax, ay), (px, py))

    t = (apx * abx + apy * aby) / ab_len_sq
    t = clamp(t, 0.0, 1.0)

    closest_x = ax + abx * t
    closest_y = ay + aby * t

    return math.dist((closest_x, closest_y), (px, py))


def segment_hits_circle(ax, ay, bx, by, cx, cy, radius):
    return segment_distance_to_point(ax, ay, bx, by, cx, cy) <= radius


def segment_hits_sun(ax, ay, bx, by):
    return segment_hits_circle(ax, ay, bx, by, CENTER_X, CENTER_Y, SUN_RADIUS)


def is_out_of_bounds(x, y):
    return x < 0 or x > BOARD_SIZE or y < 0 or y > BOARD_SIZE


def first_guess_angle(source, target, ships, angular_velocity):
    speed = fleet_speed(ships)

    current_distance = distance(source, target)
    turns = current_distance / speed

    predicted_target = predict_planet_position(target, turns, angular_velocity)

    predicted_distance = math.dist(
        (source.x, source.y),
        (predicted_target.x, predicted_target.y)
    )

    turns = predicted_distance / speed

    predicted_target = predict_planet_position(target, turns, angular_velocity)

    return angle_between(source, predicted_target)


def simulate_route(source, target, ships, shot_angle, planets, angular_velocity, max_turns):
    speed = fleet_speed(ships)

    cos_a = math.cos(shot_angle)
    sin_a = math.sin(shot_angle)

    x = source.x + cos_a * (source.radius + 0.3)
    y = source.y + sin_a * (source.radius + 0.3)

    for turn in range(1, max_turns + 1):
        old_x = x
        old_y = y

        x += cos_a * speed
        y += sin_a * speed

        if is_out_of_bounds(x, y):
            return Route(False, shot_angle, turn, "out_of_bounds")

        if segment_hits_sun(old_x, old_y, x, y):
            return Route(False, shot_angle, turn, "sun_collision")

        for planet in planets:
            if planet.id == source.id and turn <= 2:
                continue

            future_pos = predict_planet_position(planet, turn, angular_velocity)

            hit = segment_hits_circle(
                old_x,
                old_y,
                x,
                y,
                future_pos.x,
                future_pos.y,
                planet.radius
            )

            if not hit:
                continue

            if planet.id == target.id:
                return Route(True, shot_angle, turn, "hit_target")

            return Route(False, shot_angle, turn, "hit_wrong_planet")

    return Route(False, shot_angle, max_turns, "no_hit")


def logistics_find_route(source, target, ships, planets, angular_velocity):
    speed = fleet_speed(ships)
    base_distance = distance(source, target)

    max_turns = int(base_distance / speed + 35)
    max_turns = clamp(max_turns, 25, 75)

    base_angle = first_guess_angle(source, target, ships, angular_velocity)

    route = simulate_route(
        source,
        target,
        ships,
        base_angle,
        planets,
        angular_velocity,
        max_turns
    )

    if route.valid:
        return route

    candidates = []

    for i in range(1, 16):
        delta = i * 0.025
        candidates.append(base_angle + delta)
        candidates.append(base_angle - delta)

    best_route = None

    for shot_angle in candidates:
        route = simulate_route(
            source,
            target,
            ships,
            shot_angle,
            planets,
            angular_velocity,
            max_turns
        )

        if not route.valid:
            continue

        if best_route is None or route.turns < best_route.turns:
            best_route = route

    return best_route


def incoming_enemy_pressure(my_planet, enemy_fleets):
    pressure = 0

    for fleet in enemy_fleets:
        fleet_pos = Point(fleet.x, fleet.y)
        dist_to_planet = distance(fleet_pos, my_planet)

        if dist_to_planet < 30:
            pressure += fleet.ships

    return pressure


def defense_report(my_planets, enemy_fleets, planets, angular_velocity):
    proposals = []

    if not enemy_fleets:
        return proposals

    threatened = []

    for target in my_planets:
        pressure = incoming_enemy_pressure(target, enemy_fleets)

        if pressure <= 0:
            continue

        required_defense = pressure - target.ships + 5

        if required_defense <= 0:
            continue

        threatened.append((target, pressure, required_defense))

    threatened = sorted(threatened, key=lambda x: x[1], reverse=True)[:2]
    sources = sorted(my_planets, key=lambda p: p.ships, reverse=True)[:3]

    for target, pressure, required_defense in threatened:
        for source in sources:
            if source.id == target.id:
                continue

            if source.ships <= required_defense + 8:
                continue

            send_ships = min(
                int(source.ships * 0.45),
                int(required_defense + 10)
            )

            if send_ships < 5:
                continue

            route = logistics_find_route(
                source,
                target,
                send_ships,
                planets,
                angular_velocity
            )

            if route is None:
                continue

            score = 1000 + pressure * 4 - route.turns * 3

            proposals.append(
                Proposal(
                    kind="defense",
                    source=source,
                    target=target,
                    ships=send_ships,
                    score=score,
                    route=route
                )
            )

    proposals.sort(key=lambda p: p.score, reverse=True)
    return proposals


def attack_score(source, target, route, send_ships):
    score = 0

    if target.owner == -1:
        score += 80
    else:
        score += 35

    score += target.production * 45
    score += target.radius * 12
    score -= target.ships * 2.5
    score -= route.turns * 4
    score -= send_ships * 0.35
    score += source.ships * 0.15

    return score


def tactic_report(my_planets, neutral_planets, enemy_planets, planets, angular_velocity):
    proposals = []

    if neutral_planets:
        targets = neutral_planets
    else:
        targets = enemy_planets

    targets = sorted(
        targets,
        key=lambda p: p.production * 50 + p.radius * 10 - p.ships * 2,
        reverse=True
    )[:6]

    sources = sorted(
        my_planets,
        key=lambda p: p.ships,
        reverse=True
    )[:4]

    for target in targets:
        for source in sources:
            needed_ships = target.ships + 1

            if target.owner != -1:
                needed_ships += 5

            if source.ships <= needed_ships + 8:
                continue

            send_ships = min(
                int(source.ships * 0.6),
                int(needed_ships + 18)
            )

            if send_ships <= needed_ships:
                continue

            route = logistics_find_route(
                source,
                target,
                send_ships,
                planets,
                angular_velocity
            )

            if route is None:
                continue

            score = attack_score(source, target, route, send_ships)

            proposals.append(
                Proposal(
                    kind="attack",
                    source=source,
                    target=target,
                    ships=send_ships,
                    score=score,
                    route=route
                )
            )

    proposals.sort(key=lambda p: p.score, reverse=True)
    return proposals


def nearest_enemy_distance(planet, enemy_planets):
    if not enemy_planets:
        return 999

    return min(distance(planet, enemy) for enemy in enemy_planets)


def supply_report(my_planets, enemy_planets, planets, angular_velocity):
    proposals = []

    if len(my_planets) < 2:
        return proposals

    front_planets = []
    back_planets = []

    for planet in my_planets:
        d_enemy = nearest_enemy_distance(planet, enemy_planets)

        if d_enemy < 35:
            front_planets.append(planet)
        else:
            back_planets.append(planet)

    if not front_planets:
        return proposals

    if not back_planets:
        back_planets = sorted(my_planets, key=lambda p: p.ships, reverse=True)

    front_planets = sorted(front_planets, key=lambda p: p.ships)[:2]
    back_planets = sorted(back_planets, key=lambda p: p.ships, reverse=True)[:2]

    for source in back_planets:
        if source.ships < 35:
            continue

        for target in front_planets:
            if source.id == target.id:
                continue

            if target.ships > source.ships:
                continue

            send_ships = int(source.ships * 0.3)

            if send_ships < 10:
                continue

            route = logistics_find_route(
                source,
                target,
                send_ships,
                planets,
                angular_velocity
            )

            if route is None:
                continue

            enemy_distance = nearest_enemy_distance(target, enemy_planets)

            score = 200
            score += source.ships * 0.4
            score -= target.ships * 0.6
            score -= enemy_distance * 1.2
            score -= route.turns * 2

            proposals.append(
                Proposal(
                    kind="supply",
                    source=source,
                    target=target,
                    ships=send_ships,
                    score=score,
                    route=route
                )
            )

    proposals.sort(key=lambda p: p.score, reverse=True)
    return proposals


def proposal_to_move(proposal):
    return [
        proposal.source.id,
        proposal.route.angle,
        int(proposal.ships)
    ]


def general_decision(defense_proposals, attack_proposals, supply_proposals):
    moves = []
    used_sources = set()

    for proposal in defense_proposals:
        if proposal.source.id in used_sources:
            continue

        moves.append(proposal_to_move(proposal))
        used_sources.add(proposal.source.id)

        if len(moves) >= 1:
            return moves

    for proposal in attack_proposals:
        if proposal.source.id in used_sources:
            continue

        if proposal.score < 20:
            continue

        moves.append(proposal_to_move(proposal))
        used_sources.add(proposal.source.id)

        if len(moves) >= 2:
            return moves

    if len(moves) == 0:
        for proposal in supply_proposals:
            if proposal.source.id in used_sources:
                continue

            if proposal.score < 30:
                continue

            moves.append(proposal_to_move(proposal))
            used_sources.add(proposal.source.id)

            if len(moves) >= 1:
                return moves

    return moves


def agent(obs):
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    raw_fleets = obs.get("fleets", []) if isinstance(obs, dict) else obs.fleets
    angular_velocity = obs.get("angular_velocity", 0) if isinstance(obs, dict) else obs.angular_velocity

    planets = [Planet(*p) for p in raw_planets]
    fleets = [Fleet(*f) for f in raw_fleets]

    my_planets = [p for p in planets if p.owner == player]
    neutral_planets = [p for p in planets if p.owner == -1]
    enemy_planets = [p for p in planets if p.owner != player and p.owner != -1]
    enemy_fleets = [f for f in fleets if f.owner != player]

    if len(my_planets) == 0:
        return []

    defense_proposals = defense_report(
        my_planets,
        enemy_fleets,
        planets,
        angular_velocity
    )

    attack_proposals = tactic_report(
        my_planets,
        neutral_planets,
        enemy_planets,
        planets,
        angular_velocity
    )

    supply_proposals = supply_report(
        my_planets,
        enemy_planets,
        planets,
        angular_velocity
    )

    return general_decision(
        defense_proposals,
        attack_proposals,
        supply_proposals
    )