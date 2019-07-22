import numpy as np
from abc import ABC, abstractmethod
from typing import Sequence, Tuple
from enum import Enum

import matplotlib.pyplot as plt

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
    def __init__(self, distance: float, ray: Ray, color: np.array, material: Material):
        assert color.shape == (3,)
        assert distance > 0
        self.distance = distance
        self.ray = ray
        self.color = color
        self.material = material

class Thing(ABC):
    @abstractmethod
    def __call__(self, ray: Ray):
        pass
    
class Plane(Thing):
    def __init__(self, normal: np.array, start: np.array, color: np.array, material: Material):
        assert normal.shape == (3,)
        assert start.shape == (3,)
        assert color.shape == (3,)
        self.normal = normal
        self.start = start
        self.color = color 
        self.material = material

    def __call__(self, ray: Ray) -> Reflection:
        denominator = self.normal.dot(ray.normal)
        if denominator == 0:
            return Reflection(np.inf, None, np.zeros(3), self.material)

        distance = (self.start - ray.start).dot(self.normal) / denominator
        if distance <= 0:
            return Reflection(np.inf, None, np.zeros(3), self.material)
            
        point = ray(distance)
        if self.material == Material.MIRROR:
            normal = ray.normal - 2 * self.normal.dot(ray.normal) * self.normal
        elif self.material == Material.DIFFUSE:
            normal = normalize(np.random.normal(size=3))
            if np.sign(normal.dot(self.normal)) == np.sign(ray.normal.dot(self.normal)):
                normal = normal - 2 * self.normal.dot(normal) * self.normal
        else:
            normal = self.normal
        return Reflection(distance,
                          Ray(normal, point),
                          self.color,
                          self.material)


class World(object):
    def __init__(self, things: Sequence[Thing]):
        self.things = things
        
    def __call__(self, ray: Ray, max_bounce: int = 3):
        color = np.ones(3)
        for r in range(max_bounce): 
            reflection = min((thing(ray) for thing in self.things), key = lambda r: r.distance)
            color *= reflection.color
            if (color == np.zeros(3)).all():
                break
            if reflection.material == Material.LIGHT:
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


def main():
    world = World([
        Plane(np.array([0, 1, 0]), np.array([0, 2, 0]), np.array([1.0, 1.0, 1.0]), Material.LIGHT),
        Plane(np.array([1, 0, 0]), np.array([1, 0, 0]), np.array([1.0, 0.2, 0.1]), Material.DIFFUSE),
        Plane(np.array([1, 0, 0]), np.array([-1, 0, 0]), np.array([0.3, 1, 0.1]), Material.DIFFUSE),
        Plane(np.array([0, 0, 1]), np.array([0, 0, 1]), np.array([0.1, 0.4, 0.9]), Material.DIFFUSE),
        Plane(np.array([0, 0, 1]), np.array([0, 0, -1]), np.array([0.5, 0.5, 0.5]), Material.MIRROR),
        ])

    camera = Camera(
        (100, 100),
        np.array([-1, 0, 0]),
        np.array([0, 0, 0]),
        np.array([0, 1, 0])
    )
    
    num_snaps = 100
    image = sum(camera(world) for i in range(num_snaps))/num_snaps

    plt.imshow(image)
    plt.show()

if __name__ == "__main__":
    main()