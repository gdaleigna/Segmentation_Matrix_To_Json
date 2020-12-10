import numpy as np
import json
from datetime import datetime

MIN_SIZE = 25


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

    def add(self, x, y, z):
        self.segmentation_object.append(SegmentationCoordinate(x, y, z))

    def size(self):
        return len(self.segmentation_object)

    def print(self):
        for node in self.segmentation_object:
            node.print()

    def print_json_list(self):
        nodes = {}
        for node in self.segmentation_object:
            x, y, z = node.get_coordinates()
            if not nodes.__contains__(z):
                nodes[z] = {}
            if not nodes[z].__contains__(y):
                nodes[z][y] = []
            nodes[z][y].append(x)

        lookup_tree = {}
        for section in sorted(nodes.keys()):
            for line in sorted(nodes[section].keys()):
                if not lookup_tree.__contains__(section):
                    lookup_tree[section] = {}
                lookup_tree[section][line] = sorted(nodes[section][line]).__str__()[1:-1]

        return lookup_tree


class SegmentationMatrix:
    file_name: str
    size_x: int
    size_y: int
    size_z: int
    mode: int

    def __init__(self, array, file_name, mode_selection, lookup_value):
        if array.ndim == 4:
            size_x = np.shape(array)[2]
            size_y = np.shape(array)[1]
            size_z = np.shape(array)[0]

            self.file_name = file_name
            self.size_x = size_x
            self.size_y = size_y
            self.size_z = size_z
            self.mode = np.shape(array)[3]
            self.input_matrix = np.zeros((size_z, size_y, size_x), dtype=bool)
            self.segmentation_objects = []

            for i in range(0, size_z):
                for j in range(0, size_y):
                    for k in range(0, size_x):
                        self.input_matrix[i][j][k] = True if array[i][j][k][mode_selection] == lookup_value \
                            else False

        else:
            print("ERROR: Incompatible Matrix")

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

    # get_first_element_in_lookup_matrix() acts as a isEmpty() which also returns the first coordination if found
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

    def write_to_json(self):
        file_name = "Segmentation_" + datetime.now().strftime("%Y-%m-%d at %H.%M.%S") + ".json"

        json_data = {
            "file_name": "Brats18_2013_2_1_flair.nii",
            "version": 0,
            "size_x": self.size_x,
            "size_y": self.size_y,
            "size_z": self.size_z,
            "data": []
        }

        name_index = 0
        for segmentation_object in self.segmentation_objects:
            if segmentation_object.size() >= MIN_SIZE:  # Filters out objects that are smaller than MIN_SIZE pixels
                name_index += 1
                json_data['data'].append({
                    "name": "Segmentation Object " + str(name_index),
                    "active": True,
                    "coordinates": segmentation_object.print_json_list(),
                    "color": None,
                    "hidden": False,
                    "size": segmentation_object.size()
                })

        json_object = json.dumps(json_data, indent=4)
        with open(file_name, "w") as outfile:
            outfile.write(json_object)

        print("JSON successfully saved to " + file_name)
