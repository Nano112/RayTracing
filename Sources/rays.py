import numpy as np


class Ray:
    def __init__(self, origin: np.array, direction: np.array):
        self.origin = origin
        self.direction = direction


class Sphere:
    def __init__(self, origin: np.array, radius: int, albedo: np.array, is_mirror: bool = False):
        self.origin = origin
        self.radius = radius
        self.albedo = albedo
        self.is_mirror = is_mirror
