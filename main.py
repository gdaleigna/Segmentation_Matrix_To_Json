from segmentation import *

# MAIN
# CREATING RANDOM SEGMENTATION ARRAY (NOT NEEDED)
import random
x = 256
y = 256
z = 10
modes = 2
range_factor = 5
numpy_array = np.zeros((z, y, x, modes))
for i in range(0, z):
    for j in range(0, y):
        for k in range(0, x):
            numpy_array[i][j][k] = True if random.randint(0, range_factor) == 1 else 0

# START OF SEGMENTATION CONVERSION
# seg = SegmentationMatrix(numpy_array, "Brats18_2013_2_1_flair.nii", 0, 1, 10)
# seg.find_independent_objects_from_adjacency(1)
# seg.write_json_to_directory("/Users/greg/Downloads")
# END OF SEGMENTATION CONVERSION


read_matrix = read_segmentation_json("/Users/greg/Downloads/Brats18_2013_2_1_flair_seg_2020-12-14_at_21.22.37.json")
