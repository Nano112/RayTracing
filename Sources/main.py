import math
import PIL
import numpy as np
import scipy.misc as smp
from Sources.rays import *


def norm2(a: np.array):
    return np.sum(a * a)


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


def normalize(a: np.array):

    return a / math.sqrt(norm2(a))


def get_luminosity(scene, original_ray, nb_bounces:int = 0):
    if nb_bounces >= 10:
        return 0, 0, 0

    ray, closest_scene_object, _ = intersect_dans_scene(scene, original_ray)
    if ray is not None:

        if closest_scene_object.is_mirror:
            mirror_ray = Ray(ray.origin + 0.01 * ray.direction, ray.direction)
            return get_luminosity(scene, mirror_ray, nb_bounces+1)

        elif closest_scene_object.is_transparent:
            transparent_ray = ray.direction - 2 * np.dot(ray.direction )
            return get_luminosity(scene, Ray(ray.origin + 0.01 * ray.direction, ray.direction), nb_bounces+1)

        else:
            ray_light = Ray(ray.origin + 0.01 * ray.direction, normalize(scene.light.position - ray.origin))
            closest_ray_light, closest_scene_object_to_ray_light, t_light = intersect_dans_scene(scene, ray_light)
            if  closest_scene_object_to_ray_light is not None and t_light * t_light < norm2(scene.light.position - ray.origin):
                intensity = [0, 0, 0]
            else:
                intensity = closest_scene_object.albedo * scene.light.intensity * \
                            np.dot(normalize(scene.light.position - ray.origin), ray.direction) \
                            / math.sqrt(norm2(scene.light.position - ray.origin))

            r = min(255, max(0, np.power(intensity[0], 1 / 2.2)))
            g = min(255, max(0, np.power(intensity[1], 1 / 2.2)))
            b = min(255, max(0, np.power(intensity[2], 1 / 2.2)))
            return r, g, b
    return None


def render_scene(scene: Scene, width: int= 200, height: int = 200, fov: int = 60):
    fov = fov * math.pi / 180
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for x in range(0, width):
        for y in range(0, height):
            if (x*height+y) % 1000 == 0:
                print(repr(100*(x*height+y)/(width*height))+'%')
            direction = normalize(np.array([x - width / 2, y - height / 2, -width / (2 * math.tan(fov / 2))]))
            original_ray = Ray(np.array([0, 0, 0]), direction)
            luminosity = get_luminosity(scene, original_ray)
            if luminosity is not None:
                image[height-y-1][x] = luminosity
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

    mirror = Material(is_mirror=True)

    scene.light.intensity = 10000000
    scene.light.position = np.array([0, 100, -80])
    scene.objects.append(Sphere(np.array([0, -plane_size - 100, 0]), plane_size, purple))  # Floor
    scene.objects.append(Sphere(np.array([plane_size + 400, 0, 0]), plane_size, yellow))  # Right Wall
    scene.objects.append(Sphere(np.array([0, plane_size + 800, 0]), plane_size, cyan))  # Roof
    scene.objects.append(Sphere(np.array([-plane_size - 400, 0, 0]), plane_size, green))  # Left Wall
    scene.objects.append(Sphere(np.array([0, 0, plane_size + 400]), plane_size, white))  # Back Wall
    scene.objects.append(Sphere(np.array([0, 0, -plane_size - 400]), plane_size, white))  # Front Wall
    scene.objects.append(Sphere(np.array([0, -30, -200]), 80,  mirror))
    scene.objects.append(Sphere(np.array([0, 0, -100]), 20, mirror))
    scene.objects.append(Sphere(np.array([-20, -15, -110]), 10,  mirror))
    scene.objects.append(Sphere(np.array([40, 15, -80]), 10,  mirror))
    scene.objects.append(Sphere(np.array([-30, 0, -60]), 5, red))
    scene.objects.append(Sphere(np.array([-30, 30, -80]), 5, blue))
    scene.objects.append(Sphere(np.array([0, -20, -100]), 5, orange))
    return scene

def main():
    scene = create_scene()
    image = render_scene(scene, width=100, height=100, fov=110)
    img = PIL.Image.fromarray(image)
    img.save('Test.bmp')
    img.show()

main()
