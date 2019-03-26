import math
import PIL
from timeit import default_timer as timer
from numba import jit
import numpy as np
import scipy.misc as smp
from Sources.rays import *


def norm2(a: np.array):
    return np.sum(a * a)


def normalize(a: np.array):
    return a / math.sqrt(norm2(a))


def intersect_ray_sphere(r: Ray, s: Sphere):
    a: int = 1
    b: int = 2 * np.dot(r.direction, r.origin - s.origin)

    c: int = norm2(r.origin - s.origin) - (s.radius * s.radius)
    delta = (b * b) - (4 * a * c)
    if delta < 0:
        return None, None

    t1 = (-b - math.sqrt(delta)) / (2 * a)
    t2 = (-b + math.sqrt(delta)) / (2 * a)

    if t2 < 0:
        return None, None

    if t1 > 0:
        t = t1
    else:
        t = t2

    position = r.origin + t * r.direction
    normal = normalize(position - s.origin)
    return Ray(position, normal), t


def intersect_dans_scene(s, r: Ray):
    min_t = 1E99
    closest_ray = None
    cso = None

    for scene_object in s.objects:
        new_ray, t = intersect_ray_sphere(r, scene_object)
        if new_ray is not None and t < min_t:
            min_t = t
            closest_ray = new_ray
            cso = scene_object
    return closest_ray, cso, min_t


def sqr(a):
    return a * a


def get_luminosity(scene, original_ray, nb_iterations: int = 0):
    if nb_iterations >= 10:
        return 0, 0, 0

    ray, closest_scene_object, _ = intersect_dans_scene(scene, original_ray)
    if ray is not None:

        if closest_scene_object.material.is_mirror:
            mirror_direction = original_ray.direction - 2 * np.dot(ray.direction, original_ray.direction) * ray.direction
            mirror_ray = Ray(ray.origin + 0.001 * ray.direction, mirror_direction)
            return get_luminosity(scene, mirror_ray, nb_iterations + 1)

        elif closest_scene_object.material.is_transparent:
            n1 = 1
            n2 = closest_scene_object.material.refraction_index
            transparent_normal = ray.direction
            if np.dot(original_ray.direction, ray.direction) > 0:
                n1, n2 = n2, n1
                transparent_normal = -ray.direction
            radical = 1 - sqr(n1 / n2) * (1 - sqr(np.dot(transparent_normal, original_ray.direction)))
            if radical > 0:
                refracted_direction = (n1 / n2) * (original_ray.direction - np.dot(original_ray.direction,  transparent_normal) * transparent_normal) - transparent_normal * math.sqrt(radical)
                refracted_ray = Ray(ray.origin - 0.01 * transparent_normal, refracted_direction)
                return get_luminosity(scene, refracted_ray, nb_iterations + 1)
        else:
            ray_light = Ray(ray.origin + 0.001 * ray.direction, normalize(scene.light.position - ray.origin))
            closest_ray_light, closest_scene_object_to_ray_light, t_light = intersect_dans_scene(scene, ray_light)
            if False and closest_scene_object_to_ray_light is not None and t_light * t_light < norm2(
                    scene.light.position - ray.origin):
                intensity = [0, 0, 0]
            else:
                intensity = closest_scene_object.material.albedo * scene.light.intensity * \
                            np.dot(normalize(scene.light.position - ray.origin), ray.direction) \
                            / math.sqrt(norm2(scene.light.position - ray.origin))

            r = min(255, max(0, np.power(intensity[0], 1 / 2.2)))
            g = min(255, max(0, np.power(intensity[1], 1 / 2.2)))
            b = min(255, max(0, np.power(intensity[2], 1 / 2.2)))
            return r, g, b
    return None


def render_scene(scene: Scene, width: int = 200, height: int = 200, fov: int = 60):
    fov = fov * math.pi / 180
    image = np.zeros((height, width, 3), dtype=np.uint8)
    start = timer()
    for x in range(0, width):
        print(x)
        for y in range(0, height):
            direction = normalize(np.array([x - width / 2, y - height / 2, -width / (2 * math.tan(fov / 2))]))
            original_ray = Ray(np.array([0, 0, 0]), direction)
            luminosity = get_luminosity(scene, original_ray)
            if luminosity is not None:
                image[height - y - 1][x] = luminosity
    duration = timer() - start
    print(duration)
    return image


def create_scene():
    plane_size = 40000
    scene = Scene()

    white = Material(albedo=np.array([1, 1, 1]))
    red = Material(albedo=np.array([1, 0, 0]))
    green = Material(albedo=np.array([0, 1, 0]))
    blue = Material(albedo=np.array([0, 0, 1]))
    purple = Material(albedo=np.array([1, 0, 1]))
    yellow = Material(albedo=np.array([1, 1, 0]))
    cyan = Material(albedo=np.array([0, 1, 1]))
    orange = Material(albedo=np.array([1, 0.5, 0]))
    pink = Material(albedo=np.array([237 / 255, 64 / 255, 170 / 255]))

    mirror = Material(is_mirror=True)

    transparent = Material(is_transparent=True, refraction_index=1.3)

    scene.light.intensity = 1000000000
    scene.light.position = np.array([0, 100, -200])
    scene.objects.append(Sphere(np.array([0, -plane_size - 100, 0]), plane_size, mirror))  # Floor
    scene.objects.append(Sphere(np.array([plane_size + 400, 0, 0]), plane_size, mirror))  # Right Wall
    scene.objects.append(Sphere(np.array([0, plane_size + 800, 0]), plane_size, mirror))  # Roof
    scene.objects.append(Sphere(np.array([-plane_size - 400, 0, 0]), plane_size, mirror))  # Left Wall
    scene.objects.append(Sphere(np.array([0, 0, plane_size + 400]), plane_size, mirror))  # Back Wall
    scene.objects.append(Sphere(np.array([0, 0, -plane_size - 400]), plane_size, mirror))  # Front Wall

    scene.objects.append(Sphere(np.array([0, -15, -200]), 80, red))
    scene.objects.append(Sphere(np.array([0, 0, -100]), 20, red))
    scene.objects.append(Sphere(np.array([-30, -15, -300]), 10, red))
    scene.objects.append(Sphere(np.array([40, 15, -80]), 10, mirror))
    scene.objects.append(Sphere(np.array([-30, 15, -250]), 5, green))
    scene.objects.append(Sphere(np.array([-30, 0, -60]), 5, red))
    scene.objects.append(Sphere(np.array([-30, 30, -80]), 5, blue))
    scene.objects.append(Sphere(np.array([0, -20, -100]), 5, orange))
    return scene


def main():
    scene = create_scene()
    image = render_scene(scene, width=400, height=400, fov=110)
    img = PIL.Image.fromarray(image)
    img.save('Test.bmp')
    img.show()


main()
