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

    def is_equal(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

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
                        all_coordinates.append([k, j, i])

        self.total_of_pixels = len(all_coordinates)
        return all_coordinates

    def get_first_element_in_lookup_matrix(self, z):
        for i in range(z, self.size_z):
            for j in range(0, self.size_y):
                for k in range(0, self.size_x):
                    if self.input_matrix[i][j][k]:
                        return False, [k, j, i]

        return True, []

    def create_lookup_coordinates_according_to_adjacency(self, adjacency, node):
        neighbors = []
        self.input_matrix[node[2]][node[1]][node[0]] = False

        for neighbor in adjacency:
            lookup_x = node[0] + neighbor[0]
            lookup_y = node[1] + neighbor[1]
            lookup_z = node[2] + neighbor[2]
            if 0 <= lookup_x < self.size_x and 0 <= lookup_y < self.size_y and 0 <= lookup_z < self.size_z:
                if self.input_matrix[lookup_z][lookup_y][lookup_x]:
                    neighbors.append([lookup_x, lookup_y, lookup_z])
                    self.input_matrix[lookup_z][lookup_y][lookup_x] = False

        return neighbors

    def find_proximity(self, adjacency):
        is_lookup_matrix_empty, node = self.get_first_element_in_lookup_matrix(0)

        while not is_lookup_matrix_empty:
            segmentation_object = SegmentationObject()
            segmentation_object.add(node[0], node[1], node[2])

            index_coordinates = self.create_lookup_coordinates_according_to_adjacency(adjacency, node)
            lookup_coordinates = index_coordinates.copy()

            while len(index_coordinates) > 0:
                index = index_coordinates[0]
                tmp = self.create_lookup_coordinates_according_to_adjacency(adjacency, index)
                lookup_coordinates.extend(tmp)
                index_coordinates.extend(tmp)
                index_coordinates.remove(index)

            for neighbor in lookup_coordinates:
                segmentation_object.add(neighbor[0], neighbor[1], neighbor[2])

            self.segmentation_objects.append(segmentation_object)
            is_lookup_matrix_empty, node = self.get_first_element_in_lookup_matrix(node[2])

    def find_independent_objects_from_adjacency(self, mode):
        self.find_proximity(get_adjacency_for_selection(mode))

    def print_independent_objects(self):
        print("Displaying independent objects.")
        display_matrix = np.zeros((self.size_z, self.size_y, self.size_x), dtype=int)
        index = 1
        for segmentation_object in self.segmentation_objects:
            for coordinate in segmentation_object.segmentation_object:
                x, y, z = coordinate.get_coordinates()
                display_matrix[z][y][x] = index
            index += 1

        for i in range(0, self.size_z):
            for j in range(0, self.size_y):
                for k in range(0, self.size_x):
                    if display_matrix[i][j][k] != 0:
                        print(display_matrix[i][j][k], end='')
                    else:
                        print(".", end='')

                    print(" ", end='')
                print()
            print()


# MAIN
start_time = time.time()

seg = SegmentationMatrix()
# seg.copy_matrix_from_numpy_array(segmentation_matrix)

# seg.create_new_matrix(16, 8, 3)
seg.create_new_matrix(128, 128, 10)
# seg.create_new_matrix(256, 256, 30)

seg.generate_random_segmentation()
# seg.print_input_matrix()

seg.find_independent_objects_from_adjacency(1)

# seg.print_independent_objects()
print(len(seg.segmentation_objects), "independent object(s).")

# TIMER
print("--- %s seconds ---" % (time.time() - start_time))
