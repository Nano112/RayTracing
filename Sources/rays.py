import numpy as np


class Ray:
    def __init__(self, origin: np.array, direction: np.array):
        self.origin = origin
        self.direction = direction


class Material:
    def __init__(self, albedo:np.array = np.array([0, 0, 0]), is_mirror: bool = False, is_transparent: bool = False, refraction_index: np.float64 = 1.0):
        self.albedo = albedo
        self.is_mirror = is_mirror
        self.is_transparent = is_transparent
        self.refraction_index = refraction_index


class Sphere:
    def __init__(self, origin: np.array, radius: int, material: Material = Material()):
        self.origin = origin
        self.radius = radius
        self.material = material


class Light:
    def __init__(self, position: np.array = np.array([0, 0, 0]), intensity: int = 0):
        self.position = position
        self.intensity = intensity


class Scene:
    def __init__(self, objects: list = [], light:Light = Light()):
        self.light = light
        self.objects = objects

