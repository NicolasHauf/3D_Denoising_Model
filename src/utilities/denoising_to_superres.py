import numpy as np
import os
import skimage.io as io
from scipy import ndimage

denoising_result_folder = "/scratch/users/haufns/denoising_model/test_data/frag_results_3d/"

super_resolution_input_folder = "/scratch/users/haufns/super_resolution_model/dataset/"

file_name_list = os.listdir(denoising_result_folder)


def create_block(input_folder, file_name, size=32, lower_coordinates=[0, 0, 0], output_directory_super_res='dataset', train=True):

    if not os.path.exists(os.path.join(output_directory_super_res, 'train', 'HR')):
        os.makedirs(os.path.join(output_directory_super_res, 'train', 'HR'))
    if not os.path.exists(os.path.join(output_directory_super_res, 'train', 'LR')):
        os.makedirs(os.path.join(output_directory_super_res, 'train', 'LR'))
    if not os.path.exists(os.path.join(output_directory_super_res, 'test', 'HR')):
        os.makedirs(os.path.join(output_directory_super_res, 'test', 'HR'))
    if not os.path.exists(os.path.join(output_directory_super_res, 'test', 'LR')):
        os.makedirs(os.path.join(output_directory_super_res, 'test', 'LR'))

    #output_directory_file = os.path.join(output_directory, f'image_x{lower_coordinates[0]:03d}_y{lower_coordinates[1]:03d}_z{lower_coordinates[2]:03d}.tiff')
    if train:
        super_res_directory_file_gt = os.path.join(output_directory_super_res, 'train', 'HR', f'gt_{file_name}')
        super_res_directory_file_input = os.path.join(output_directory_super_res, 'train', 'LR', f'input_{file_name}')
    else:
        super_res_directory_file_gt = os.path.join(output_directory_super_res, 'test', 'HR', f'gt_{file_name}')
        super_res_directory_file_input = os.path.join(output_directory_super_res, 'test', 'LR', f'input_{file_name}')
       

    block = np.zeros([size, size, size])
    block_lr = np.zeros([32, 32, 32])
    if not os.path.exists(input_folder + file_name):
        print("block out of bounds!")
        return None
    f = io.imread(input_folder + file_name)
    print(np.mean(f))
    for z in range(len(f)):

        block[z, :, :] = f[z]
        if z % 2 == 0:
            block_lr[int(z / 2), :, :] = ndimage.zoom(f[z], 0.5)
   
    if np.mean(block) > 3: # removes low contrast blocks, because it is better to train the superresolution model without them (low resolution blocks are mostly around the cylinder where there is no material)
        io.imsave(super_res_directory_file_gt, block)
        io.imsave(super_res_directory_file_input, block_lr)
        print(super_res_directory_file_gt, super_res_directory_file_input)

    return block



for file_index in range(len(file_name_list)):
    create_block(denoising_result_folder, file_name_list[file_index], size=64, lower_coordinates=[0, 0, 0], output_directory_super_res=super_resolution_input_folder, train = file_index < 0.8 * len(file_name_list))
