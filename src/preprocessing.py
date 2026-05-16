# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 16:25:03 2020
Expanded on 2025
Cleaned up 2026

@author: Nicolas Hauf,Dongwon Lee.
@nicolashauf@gmail.com
@dongwon.lee@mechbau.uni-stuttgart.de
@ pre-processing before training
"""

from img_seg_3d import *

# data locations
tpath = os.path.join(os.getcwd(), '..')
subpath = 'sample_data'
data_folder = 'original_images'
seg_data_folder = os.path.join('CV', 'merged_cv_v2')

save_subpath = 'results'
save_trainingdata_folder = os.path.join('U_net', 'training_data', 'org_frag_data_3d')
save_seg_trainingdata_folder = os.path.join('U_net', 'training_data', 'seg_frag_data_3d')


block_size = 64

org_flist,org_fpath = readingfiles(os.path.join(tpath,subpath),data_folder)

org_savepath = os.path.join(tpath,subpath,save_trainingdata_folder)

seg_flist,seg_fpath = readingfiles(os.path.join(tpath, save_subpath),seg_data_folder)

seg_savepath = os.path.join(tpath,subpath,save_seg_trainingdata_folder)

# make directories
if not os.path.exists(org_savepath):
    os.makedirs(org_savepath)

if not os.path.exists(seg_savepath):
    os.makedirs(seg_savepath)

#%% get the averaged minimum of sample data
avg_min_stack =0
avg_max_stack =0
number_stacks = len(org_flist)

for i in range(number_stacks):
    first_image = io.imread(os.path.join(org_fpath,org_flist[i]))
    original_size = first_image.shape
    g_min = np.min(first_image) 
    g_max = np.max(first_image)
    avg_min_stack += g_min / number_stacks
    avg_max_stack += g_max / number_stacks

g_avg_min = int(avg_min_stack)
g_avg_max = int(avg_max_stack)

# org data

overlap = 1

n_files = len(org_flist)
# the last iteration only works with fewer slices and is resized in the create_3d_blocks function

for i in range(0, n_files, block_size - overlap):
    rdata = []
    for j in range(0, original_size[0], block_size - overlap):
        rdata = [io.imread(os.path.join(org_fpath, org_flist[k]))[j: min(j + block_size, original_size[0])] for k in range(i, min(i + block_size, n_files))]
        rdata = np.array(rdata, dtype=float)
        
        np.clip(rdata, g_avg_min, g_avg_max, out=rdata)
        rdata -= g_avg_min
        rdata = ((255./(g_avg_max-g_avg_min)) * rdata).astype(np.uint8)
        
        f_name = org_flist[i][:-4] + "_{:04d}".format(j) + org_flist[i][-4:]
        create_3d_blocks_v1(rdata, block_size, filename=f_name, spath=org_savepath, overlap=overlap, datatype='uint8')


    
# seg data

n_files = len(seg_flist)
for i in range(0, n_files, block_size - overlap):
    rdata = []
    for j in range(0, original_size[0], block_size - overlap):
        rdata = [io.imread(os.path.join(seg_fpath, seg_flist[k]))[j: min(j + block_size, original_size[0])] for k in range(i, min(i + block_size, n_files))]
        rdata = np.array(rdata, dtype=float)

        f_name = seg_flist[i][:-4] + "_{:04d}".format(j) + seg_flist[i][-4:]
        create_3d_blocks_v1(rdata, block_size, filename=f_name, spath=seg_savepath, overlap=overlap, datatype='uint8')

