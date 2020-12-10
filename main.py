import numpy as np
import random
import time
from segmentation import *


# MAIN
start_time = time.time()

x = 256
y = 256
z = 16
modes = 2
range_factor = 4

numpy_array = np.zeros((z, y, x, modes))

for i in range(0, z):
    for j in range(0, y):
        for k in range(0, x):
            numpy_array[i][j][k] = True if random.randint(0, range_factor) == 1 else 0

seg = SegmentationMatrix(numpy_array, "Brats18_2013_2_1_flair.nii", 0, 1)

# seg.create_new_matrix(128, 128, 10)
# seg.generate_random_segmentation(3)
# seg.print_input_matrix()

seg.find_independent_objects_from_adjacency(1)
print(len(seg.segmentation_objects), "independent object(s).")

# seg.print_independent_objects()
seg.write_to_json()

# TIMER
print("--- %s seconds ---" % (time.time() - start_time))