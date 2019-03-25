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

class Light:
    def __init__(self, position: np.array =np.array([0,0,0]), intensity: int = 0):
        self.position = position
        self.intensity = intensity


class Scene:
    def __init__(self, objects: list = [], light:Light = Light()):
        self.light = light
        self.objects = objects

