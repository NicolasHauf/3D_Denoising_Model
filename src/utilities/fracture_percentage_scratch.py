import numpy as np
from tqdm import tqdm
import os
import skimage.io as io
import collections
import matplotlib.pyplot as plt


models = os.listdir('/scratch/users/haufns/denoising_model/test_data/frag_results_3d/')
models.sort()

rfs = ['/scratch/users/haufns/denoising_model/test_data/frag_results_3d/' + model for model in models]


for i in range(len(rfs)):
    rf = rfs[i]

    if not os.path.isdir(rf):
        continue
    
    slice_list = os.listdir(rf)
    
    nz_percentage = 0
    nz_bin_percentage = 0
    for slice_name in tqdm(slice_list):
        slice_image = io.imread(os.path.join(rf, slice_name))
        bin_image = np.round(slice_image / 255)
        nz = np.count_nonzero(slice_image)
        nz_bin = np.count_nonzero(bin_image)

        for i in range(len(slice_image.shape)):
            nz /= slice_image.shape[i]
            nz_bin /= slice_image.shape[i]

        nz_percentage += nz
        nz_bin_percentage += nz_bin
    
    print(rf)
    print('Percentage of non-zero pixels (estimated porosity):', nz_percentage / len(slice_list))
    print('Percentage of non-zero pixels binarized (estimated porosity):', nz_bin_percentage / len(slice_list), '\n')
    
