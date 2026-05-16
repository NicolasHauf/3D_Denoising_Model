import numpy as np
from tqdm import tqdm
import os
import skimage.io as io
import collections
import matplotlib.pyplot as plt


unet_result_folder = os.path.join('..', 'results', 'U_net', 'seg_merged')
unet_result_unmerged_folder = os.path.join('..', 'results', 'U_net', 'frag_results')
scratch_denoising = os.path.join('/scratch', 'users', 'haufns', 'best_denoising_result')
scratch_super_resolution = os.path.join('/scratch', 'users', 'haufns', 'best_super_resolution_result')

unet_3d_reconstructed = os.path.join('..', 'results', 'U_net', 'seg_merged_3d')
LF_result_folder = os.path.join('..', 'results', 'Local_threshold')
CV_result_folder = os.path.join('..', 'results', 'CV', 'merged_cv_v2')
unet_3d_result_folder = os.path.join('..', 'results', 'U_net', 'frag_results_3d')

result_folders = [scratch_denoising, scratch_super_resolution, CV_result_folder, unet_result_folder, LF_result_folder]

for rf in result_folders:
    slice_list = os.listdir(rf)

    nz_percentage = 0
    for slice_name in tqdm(slice_list):
        slice_image = io.imread(os.path.join(rf, slice_name))

        nz = np.count_nonzero(slice_image)

        for i in range(len(slice_image.shape)):
            nz /= slice_image.shape[i]
        nz_percentage += nz
        
    print(rf)
    print('Percentage of non-zero pixels (estimated porosity):', nz_percentage / len(slice_list), '\n')
