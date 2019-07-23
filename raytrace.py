import argparse

from abc import ABC, abstractmethod
from enum import Enum
from typing import Sequence, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

EPSILON = 1e-10

class Material(Enum):
    MIRROR = 1
    LIGHT = 2
    DIFFUSE = 3

class Ray(object):
    def __init__(self, normal: np.array, start: np.array):
        assert normal.shape == (3,)
        assert start.shape == (3,)
        self.normal = normal
        self.start = start
    
    def __call__(self, distance: float) -> np.array:
        assert distance > 0
        return self.start + distance * self.normal

class Reflection(object):
    def __init__(self, distance: float, ray: Optional[Ray], color: np.array, material: Material):
        assert color.shape == (3,)
        assert distance > 0
        self.distance = distance
        self.ray = ray
        self.color = color
        self.material = material

class Thing(ABC):
    @abstractmethod
    def __init__(self, color: np.array, material: Material):
        assert color.shape == (3,)
        self.color = color
        self.material = material

    @abstractmethod
    def surface_normal(self, point: np.array) -> np.ndarray:
        pass

    @abstractmethod
    def intersect(self, ray: Ray) -> Tuple[float, Optional[np.array]]:
        pass

    def __call__(
            self,
            ray: Ray
    ) -> Reflection:
        distance, intersection = self.intersect(ray)
        if distance == np.inf:
            return Reflection(distance, None, self.color, self.material)

        surface_normal = self.surface_normal(intersection)

        if self.material == Material.MIRROR:
            normal = ray.normal - 2 * surface_normal.dot(ray.normal) * surface_normal
        elif self.material == Material.DIFFUSE:
            normal = normalize(np.random.normal(size=3))
            if np.sign(normal.dot(surface_normal)) == np.sign(ray.normal.dot(surface_normal)):
                normal = -normal
        else:
            normal = np.array([0, 0, 0])

        color_scale = 1
        if self.material == Material.LIGHT:
            color_scale = abs(ray.normal.dot(surface_normal))

        return Reflection(
                distance,
                Ray(normal, intersection),
                self.color * color_scale,
                self.material)
    
class Plane(Thing):
    def __init__(self, normal: np.array, start: np.array, color: np.array, material: Material):
        super().__init__(color, material)
        assert normal.shape == (3,)
        assert start.shape == (3,)
        self.normal = normal
        self.start = start

    def surface_normal(self, point: np.array) -> np.array:
        return self.normal

    def intersect(self, ray: Ray) -> Tuple[float, Optional[np.array]]:
        denominator = self.normal.dot(ray.normal)
        if denominator == 0:
            return np.inf, None

        distance = (self.start - ray.start).dot(self.normal) / denominator
        if distance <= EPSILON:
            return np.inf, None
            
        return distance, ray(distance)


class Sphere(Thing):
    def __init__(self, center: np.array, radius: float, color: np.array, material: Material):
        super().__init__(color, material)

        assert center.shape == (3,)
        self.center = center
        self.radius = radius

    def surface_normal(self, point: np.array) -> np.array:
        return (point - self.center) / self.radius

    def intersect(self, ray: Ray) -> Tuple[float, Optional[np.array]]:
        diff = self.center - ray.start
        under_rot = (diff.dot(ray.normal) ** 2
                     + self.radius ** 2
                     - (diff).dot(diff))
        if under_rot < 0:
            return np.inf, None
        rot = np.sqrt(under_rot)
        c = diff.dot(ray.normal)
        candidates = [c + rot, c - rot, np.inf]
        distance = min(c for c in candidates if c > EPSILON)

        if distance == np.inf:
            return np.inf, None
        return distance, ray(distance)
        

class World(object):
    def __init__(self, things: Sequence[Thing], max_bounces: int = 3):
        self.things = things
        self.max_bounces = max_bounces
        
    def __call__(self, ray: Ray):
        color = np.ones(3)
        for bounce in range(self.max_bounces):
            reflection = min((thing(ray) for thing in self.things), key = lambda r: r.distance)
            color *= reflection.color
            if (color == np.zeros(3)).all() or reflection.ray is None:
                break
            elif reflection.material == Material.LIGHT:
                return color
            ray = reflection.ray
        return np.zeros(3)


class Camera(object):
    def __init__(self,
                 resolution: Tuple[int, int],
                 left_edge: np.array,
                 position: np.array,
                 normal: np.array):
        assert normal.shape == (3,)
        assert left_edge.shape == (3,)
        assert position.shape == (3,)
        self.resolution = resolution
        self.left_edge = left_edge
        self.position = position
        self.normal = normal
        
        self.top_edge = np.cross(normal, left_edge) * resolution[1]/resolution[0]

    def __call__(self, world: World) -> np.array:
        image = np.zeros([*self.resolution, 3])
        for i in range(self.resolution[0]):
            for j in range(self.resolution[1]):
                normal = normalize(
                    self.normal
                    + (i - self.resolution[0] / 2) * self.left_edge / (self.resolution[0] / 2)
                    + (j - self.resolution[1] / 2) * self.top_edge / (self.resolution[1] / 2)
                )
                ray = Ray(normal, self.position)
                image[i, j, ...] = world(ray)
        return image


def normalize(a: np.array) -> np.array:
   return a / np.sqrt(a.dot(a))


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', default=100, type=int)
    parser.add_argument('--num_passes', default=10, type=int)
    parser.add_argument('--num_bounces', default=3, type=int)
    return parser.parse_args()

def main():
    args = _parse_args()
    world = World([
        Plane(np.array([0, 1, 0]), np.array([0, 2, 0]),
            np.array([1.0, 0.2, 0.1]), Material.DIFFUSE),
        Plane(np.array([1, 0, 0]), np.array([1, 0, 0]),
            np.array([1.0, 1.0, 1.0]), Material.LIGHT),
        Plane(np.array([1, 0, 0]), np.array([-1, 0, 0]),
            np.array([0.3, 1, 0.1]), Material.DIFFUSE),
        Plane(np.array([0, 0, 1]), np.array([0, 0, 1]),
            np.array([0.1, 0.4, 0.9]), Material.DIFFUSE),
        Plane(np.array([0, 0, -1]), np.array([0, 0, -1]),
            np.array([0.9, 0.9, 0.1]), Material.DIFFUSE),
        Sphere(np.array([-0.5, 1.5, 0]),  0.5,
            np.array([0.7, 0.7, 0.7]), Material.MIRROR),
        ], max_bounces=args.num_bounces)

    camera = Camera(
        (args.image_size, args.image_size),
        np.array([-1, 0, 0]),
        np.array([0, 0, 0]),
        np.array([0, 1, 0])
    )
    
    image = np.zeros([*camera.resolution, 3])
    for i in tqdm(range(args.num_passes)):
        image += camera(world)
    image /= np.max(image)

    plt.imshow(image)
    plt.show()

if __name__ == "__main__":
    main()
