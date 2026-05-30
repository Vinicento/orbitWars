import math

PID = 0
OWNER = 1
X = 2
Y = 3
SHIPS = 5


def agent(obs):
    player = obs["player"]
    planets = obs["planets"]
    mine = [p for p in planets if p[OWNER] == player]
    targets = [p for p in planets if p[OWNER] != player]
    moves = []
    if not targets:
        return moves

    for source in mine:
        target = min(targets, key=lambda t: math.hypot(source[X] - t[X], source[Y] - t[Y]))
        ships = int(target[SHIPS] + 1)
        if source[SHIPS] >= ships:
            angle = math.atan2(target[Y] - source[Y], target[X] - source[X])
            moves.append([int(source[PID]), angle, ships])

    return moves
