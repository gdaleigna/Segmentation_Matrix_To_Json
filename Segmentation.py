import numpy as np
from collections import deque
import random
import time

start_time = time.time()


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
        self.x = x
        self.y = y
        self.z = z

    def get_coordinates(self):
        return self.x, self.y, self.z

    def print(self):
        print("X: ", self.x, ", Y: ", self.y, ", Z: ", self.z, sep="")


class SegmentationObject:
    segmentation_object = []

    def __init__(self):
        self.segmentation_object = []

    def __contains__(self, coordinate):
        for node in self.segmentation_object:
            if coordinate.get_coordinates() == node.get_coordinates():
                return True

        return False

    def add(self, x, y, z):
        self.segmentation_object.append(SegmentationCoordinate(x, y, z))

    def is_empty(self):
        return self.segmentation_object == []

    def find_node(self, coordinate):
        for node in self.segmentation_object:
            if coordinate.get_coordinates() == node.get_coordinates():
                return node

        return None

    def size(self):
        return len(self.segmentation_object)


class SegmentationMatrix:
    size_x: int
    size_y: int
    size_z: int
    mode: int
    numberOfObjects: int

    def __init__(self):
        self.size_x = 3
        self.size_y = 3
        self.size_z = 2
        self.numberOfObjects = -1
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
        all_coordinates = SegmentationObject()

        for i in range(0, self.size_z):
            for j in range(0, self.size_y):
                for k in range(0, self.size_x):
                    if self.input_matrix[i][j][k] > 0:
                        all_coordinates.add(k, j, i)

        self.numberOfObjects = all_coordinates.size()
        return all_coordinates

    def print_size_input_objects(self):
        print("Number of 1s:", self.numberOfObjects)

    def print_all_coordinates(self):
        all_coordinates = self.get_all_input_coordinates()

        for i in all_coordinates:
            i.print()

    def is_input_matrix_empty(self, seg_tmp):
        for i in range(0, self.size_z):
            for j in range(0, self.size_y):
                for k in range(0, self.size_x):
                    if self.input_matrix[i][j][k][0] > 0:
                        return False

        return True

    def create_coordinate_for_lookup(self, coordinate, offset):
        node2 = SegmentationCoordinate(0, 0, 0)
        return node2

    def find_proximity(self, adjacency):
        seg_tmp = self.input_matrix
        all_mri_objects = []
        mri_object = []
        all_coordinates = self.get_all_input_coordinates()

        while not all_coordinates.is_empty():
            node = all_coordinates.segmentation_object[0]
            node2 = self.create_coordinate_for_lookup(node, 1)
            all_coordinates

    def find_independent_objects_from_adjacency(self):
        self.find_proximity(get_adjacency_for_selection(1))


# MAIN
seg = SegmentationMatrix()
# seg.copy_matrix_from_numpy_array(segmentation_matrix)

seg.create_new_matrix(12, 8, 3)
# seg.create_new_matrix(32, 16, 5)
seg.generate_random_segmentation()
seg.print_input_matrix()

seg.print_all_coordinates()
seg.find_independent_objects_from_adjacency()

# TIMER
print("--- %s seconds ---" % (time.time() - start_time))
