import numpy as np
from collections import deque
import random
import time


# ADJACENCY DEFINITIONS
def direct_adjacency():
    adjacency = (-1, 0, 0), \
                (0, -1, 0), \
                (0, 0, -1), \
                (+1, 0, 0), \
                (0, +1, 0), \
                (0, 0, +1)

    return adjacency


def diagonal_adjacency():
    adjacency = (-1, -1, +1), \
                (0, -1, +1), \
                (+1, -1, +1), \
                (-1, -1, 0), \
                (0, -1, 0), \
                (+1, -1, 0), \
                (-1, -1, -1), \
                (0, -1, -1), \
                (+1, -1, -1), \
                (-1, 0, +1), \
                (0, 0, +1), \
                (+1, 0, +1), \
                (-1, 0, 0), \
                (+1, 0, 0), \
                (-1, 0, -1), \
                (0, 0, -1), \
                (+1, 0, -1), \
                (-1, +1, +1), \
                (0, +1, +1), \
                (+1, +1, +1), \
                (-1, +1, 0), \
                (0, +1, 0), \
                (+1, +1, 0), \
                (-1, +1, -1), \
                (0, +1, -1), \
                (+1, +1, -1)

    return adjacency


def get_adjacency_for_selection(selection):
    adjacency = {0: direct_adjacency,
                 1: diagonal_adjacency
                 }
    return adjacency[selection]()


# CLASS
class SegmentationCoordinate:
    x: int
    y: int
    z: int

    def __init__(self, x=None, y=None, z=None):
        self.x, self.y, self.z = x, y, z

    def get_coordinates(self):
        return self.x, self.y, self.z

    def print(self):
        print("X: ", self.x, ", Y: ", self.y, ", Z: ", self.z, sep="")


class SegmentationObject:
    segmentation_object = []

    def __init__(self):
        self.segmentation_object = []

    # def __contains__(self, coordinate):
    #     for node in self.segmentation_object:
    #         if coordinate.get_coordinates() == node.get_coordinates():
    #             return True
    #
    #     return False

    def add(self, x, y, z):
        self.segmentation_object.append(SegmentationCoordinate(x, y, z))

    def is_empty(self):
        return self.segmentation_object == []

    def find(self, coordinate):
        for node in self.segmentation_object:
            if coordinate.get_coordinates() == node.get_coordinates():
                return node

        return None

    def size(self):
        return len(self.segmentation_object)

    def print(self):
        for node in self.segmentation_object:
            node.print()


class SegmentationMatrix:
    size_x: int
    size_y: int
    size_z: int
    mode: int
    total_of_pixels: int

    def __init__(self):
        self.size_x = 3
        self.size_y = 3
        self.size_z = 2
        self.total_of_pixels = -1
        self.input_matrix = np.zeros((self.size_z, self.size_y, self.size_x), dtype=bool)
        self.segmentation_objects = []

    def copy_matrix_from_numpy_array(self, numpy_array):
        if numpy_array.ndim == 3:
            self.size_x = np.shape(numpy_array)[1]
            self.size_y = np.shape(numpy_array)[2]
            self.size_z = np.shape(numpy_array)[0]
            self.input_matrix = numpy_array
        else:
            print("ERROR: Incompatible Matrix")

    def create_new_matrix(self, x, y, z):
        self.size_x = x
        self.size_y = y
        self.size_z = z
        self.input_matrix = np.zeros((z, y, x), dtype=bool)

    def generate_random_segmentation(self):
        for i in range(0, self.size_z):
            for j in range(0, self.size_y):
                for k in range(0, self.size_x):
                    self.input_matrix[i][j][k] = True if random.randint(0, 3) == 1 else 0

    def print_input_matrix(self):
        print("Segmentation Size is", self.size_x, "x", self.size_y, "with", self.size_z, "image(s) and 1 mode.")
        for i in range(0, self.size_z):
            for j in range(0, self.size_y):
                for k in range(0, self.size_x):
                    if self.input_matrix[i][j][k]:
                        print("1", end='')
                    else:
                        print(".", end='')

                    print(" ", end='')

                print()
            print()

    def get_all_input_coordinates(self):
        all_coordinates = deque()

        for i in range(0, self.size_z):
            for j in range(0, self.size_y):
                for k in range(0, self.size_x):
                    if self.input_matrix[i][j][k] > 0:
                        all_coordinates.append(SegmentationCoordinate(k, j, i))

        self.total_of_pixels = len(all_coordinates)
        return all_coordinates

    def print_size_input_objects(self):
        print("Number of 1s:", self.total_of_pixels)

    def create_lookup_coordinates_according_to_adjacency(self, node, adjacency):
        neighbors = []
        x, y, z = node.get_coordinates()

        for neighbor in adjacency:
            lookup_x = x + neighbor[0]
            lookup_y = y + neighbor[1]
            lookup_z = z + neighbor[2]
            if 0 <= lookup_x < self.size_x and 0 <= lookup_y < self.size_y and 0 <= lookup_z < self.size_z:
                if self.input_matrix[lookup_z][lookup_y][lookup_x]:
                    neighbors.append(SegmentationCoordinate(lookup_x, lookup_y, lookup_z))
                    print("New match")

        return neighbors

    def find_proximity(self, adjacency):
        # seg_tmp = self.input_matrix # TODO: Check if a local copy is faster than self.input_matrix

        all_coordinates = self.get_all_input_coordinates()
        all_mri_objects = []
        mri_object = []

        print("Number of 1s:", self.total_of_pixels)

        while len(all_coordinates) > 0:
            node = all_coordinates.popleft()
            print("New object", node.get_coordinates())
            mri_object.append(node)
            lookup_coordinates = self.create_lookup_coordinates_according_to_adjacency(node, adjacency)
            for neighbor in lookup_coordinates:
                if self.input_matrix[neighbor.z][neighbor.y][neighbor.x]: # Change this line to use a find function as opposed to a
                    print(len(all_mri_objects) + 1, ":", neighbor.get_coordinates(), "is a match for", node.get_coordinates(), "with", self.input_matrix[neighbor.z][neighbor.y][neighbor.x])
                    # all_coordinates.remove(neighbor) # TODO: Create a find function for all_coordinates deque
                    mri_object.append(SegmentationCoordinate(neighbor.x, neighbor.y, neighbor.z))


                # if all_coordinates.__contains__(neighbor):
                #     all_coordinates.remove(neighbor)
                #     mri_object.append(neighbor)

            all_mri_objects.append(mri_object)

    def find_independent_objects_from_adjacency(self):
        self.find_proximity(get_adjacency_for_selection(1))


# MAIN
start_time = time.time()

seg = SegmentationMatrix()
# seg.copy_matrix_from_numpy_array(segmentation_matrix)

seg.create_new_matrix(128, 128, 64)
# seg.create_new_matrix(16, 8, 3)
seg.generate_random_segmentation()
# seg.print_input_matrix()

seg.find_independent_objects_from_adjacency()

# TIMER
print("--- %s seconds ---" % (time.time() - start_time))
