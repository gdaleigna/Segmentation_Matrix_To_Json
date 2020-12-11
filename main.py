import random
from segmentation import *

# MAIN
# CREATING RANDOM SEGMENTATION ARRAY
x = 512
y = 512
z = 10
modes = 2
range_factor = 2
numpy_array = np.zeros((z, y, x, modes))
for i in range(0, z):
    for j in range(0, y):
        for k in range(0, x):
            numpy_array[i][j][k] = True if random.randint(0, range_factor) == 1 else 0

# START OF SEGMENTATION CONVERSION
seg = SegmentationMatrix(numpy_array, "Brats18_2013_2_1_flair.nii", 0, 1, 25)
seg.find_independent_objects_from_adjacency(1)
seg.write_to_json("")
# END OF SEGMENTATION CONVERSION
