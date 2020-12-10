import math


def rad(angle: float) -> float:
    return angle / 360 * 2 * math.pi


def distance(p1: tuple, p2: tuple) -> float:
    deltaX = p1[0] - p2[0]
    deltaY = p1[1] - p2[1]
    return math.sqrt(deltaX * deltaX + deltaY * deltaY)
