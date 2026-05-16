# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 16:25:03 2020
Expanded on 2025
Cleaned up 2026


@author: Nicolas Hauf, Dongwon Lee.
@nicolashauf@gmail.com
@dongwon.lee@mechbau.uni-stuttgart.de
@ Functions for segmentations
"""

import os

import skimage.filters as filt
import skimage.io as io
import scipy.ndimage.morphology as smorph
import skimage.morphology as morph
import numpy as np

# for testing
import skimage.exposure as exposure
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.color import label2rgb
from skimage.util import view_as_blocks
import tifffile as tif

from tqdm import tqdm


def create_3d_blocks_v1(data, block_size, filename, spath, overlap=1, datatype='uint8'):
    padding_needed = [0, 0]
    if data.shape[0] != block_size:
        padding_needed[0] = block_size - data.shape[0]
    if data.shape[1] != block_size:
        padding_needed[1] = block_size - data.shape[1]
    
    data = np.resize(data, (data.shape[0] + padding_needed[0], data.shape[1] + padding_needed[1], data.shape[2]))
    for l in range(0, data.shape[2], block_size - overlap):
        block = data[:, :, l: min(l + block_size, data.shape[2])]

        if block.shape[2] != block_size:
            block = np.resize(block, [block_size, block_size, block_size])

        
        fname = filename[:-4] + "_{:04d}".format(l) + '.tif'
        assert block.shape == (block_size, block_size, block_size)
        save3dImages(block, spath, fname, datatype=datatype)


def create_3d_blocks_v2(data, block_size, filename, spath, datatype='uint8'):
    padding_needed = padding([data.shape[1], data.shape[2]], block_size)
    data = np.resize(data, (block_size, data.shape[1] + padding_needed[0], data.shape[2] + padding_needed[1]))

    data = view_as_blocks(data, (block_size, block_size, block_size))
    shape = data.shape


    data = data.reshape([shape[0] * shape[1] * shape[2], shape[3], shape[4], shape[5]])

    filecounter = 0
    for cropped_block in data:
        fname = filename[:-4] + '_cropped' + "_{:04d}".format(filecounter) + '.tif'
        save3dImages(cropped_block, spath, fname, datatype=datatype)
        filecounter +=1
    return data


def padding(original_shape, block_size):
    return (block_size - (original_shape[0] % block_size), block_size - (original_shape[1] % block_size))


def merge_3d_blocks(data, original_shape, filename, spath, datatype='uint8'):
    block_size = data.shape[1]
    padding_needed = padding(original_shape, block_size)
    data = np.reshape(data, [1, 184, 184, 16, 16, 16])
    data = np.reshape(data, [block_size, original_shape[0] + padding_needed[0], original_shape[1] + padding_needed[1]])
    data = np.resize(data, (block_size,  original_shape[0],  original_shape[1]))

    return data



def readingfiles(tpath,folder):
    rpath = os.path.join(tpath,folder)
    flist = os.listdir(rpath)
    flist.sort()
    return flist,rpath


def create_ringmask(rdata):
    thres = filt.threshold_otsu(rdata)
    ringmask =  smorph.binary_fill_holes(rdata>thres)
    return ringmask


# sato segmentation
def seq_sato(rdata,d_range = range(1,20,1)):
    wsrdata = filt.sato(rdata,d_range,black_ridges = True)
    
    #local threshold
    loc_thres = filt.threshold_local(wsrdata,51,offset = -(1e-3))
    bimg = wsrdata>loc_thres
    
    # erosion
    img_erosion = morph.erosion(bimg,morph.diamond(1))
    img_spots = morph.remove_small_objects(bimg,30,connectivity =1)
    
    rmask = create_ringmask(rdata)
    fimg = img_spots * rmask
    return fimg

def seq_sato_unfiltered(rdata,d_range = range(1,20,1)):
    wsrdata = filt.sato(rdata,d_range,black_ridges = True)
    #local threshold
    
    loc_thres = filt.threshold_local(wsrdata,51,offset = -0.001)
    
    bimg = wsrdata > loc_thres
    #erosion
    img_erosion = morph.binary_erosion(bimg,morph.diamond(1))
    img_spots = morph.remove_small_objects(img_erosion,50,connectivity = 1)
    
    rmask = create_ringmask(rdata)
    fimg = img_spots * rmask
    return fimg

# adaptive threshold segmentation
def seq_adthres(rdata,kernel = 101,offset = 0,remove_object_size = 1e+3):
    loc_thres = filt.threshold_local(rdata,kernel, offset =offset)
    bimg = rdata < loc_thres
    
    #bimg = morph.erosion(bimg,morph.diamond(1)) # added
    reimg = morph.remove_small_objects(bimg,remove_object_size)
    
    return reimg
    

def comparison(ref,img):
    #plt.close('all')
    rtemp = exposure.equalize_hist(ref)
    labels = label(img)
    limg = label2rgb(labels,
                     image = rtemp,
                     bg_label = 0,
                     image_alpha = 1,
                     alpha = 0.2
                     )
    fig = plt.imshow(limg)    
    return fig
    
def image_slicer(rdata,img_size = 512):
    rx,ry = rdata.shape
    for iTemp in range(0,rx,img_size):
        for jTemp in range(0,ry,img_size):
            rtemp  = rdata[iTemp:iTemp + img_size,jTemp:jTemp+img_size]
            yield rtemp
            
def overlap_image_slicer2(rdata,img_size =400,target_size = 256):
    rx,ry = rdata.shape
    zero_temp = np.zeros([img_size,img_size])
    scut = int((img_size - target_size)/2/2)
    
    x_counter = 0
    y_counter = 0
#    for iTemp in range(0,rx,img_size):
#       for jTemp in range(0,ry,img_size):
    for iTemp in range(0,rx, img_size - scut):
        for jTemp in range(0,ry,img_size-scut):
            zero_temp = np.zeros([img_size,img_size])
            #crdata= rdata[iTemp:img_size-scut*x_counter,jTemp:img_size-scut*y_counter]
            crdata = rdata[iTemp:(iTemp + img_size), jTemp:(img_size+jTemp)];
            
            ss = crdata.shape
            sx = ss[0]
            sy = ss[1]
            
            zero_temp[0:sx,0:sy] = crdata
            y_counter += 1

            yield zero_temp    
        x_counter +=1    
        
            
    
def overlap_cropped_image_saver2(rdata,img_size,
                        filename,spath,datatype='uint8'):
    generator_fun = overlap_image_slicer2(rdata,img_size,target_size = 256)
    filecounter = 0
    for cropped_img in zip(generator_fun):
        fstr = ""
        ftemp = filename.split('.')[0:-1]
        fftemp = fstr.join(ftemp)
        fname = fftemp + '_cropped' + "_{:04d}".format((filecounter)) + '.tif'
        #savename = os.path.join(sfolder,"uint8_{:04d}".format(counter) + ".tif")
        
        csavepath = os.path.join(spath)
        if datatype == 'uint8_norm':
            # with normalization
            saveImages(changer16to8(cropped_img[0]),
                       csavepath,fname,
                       datatype = 'uint8')
        elif datatype == 'histeq':
            saveImages(cropped_img[0],csavepath,fname,
                       datatype = 'uint8')
        else:
            saveImages(cropped_img[0],csavepath,fname,
                   datatype = datatype)
        filecounter +=1;
        
    
    
            
def overlap_image_slicer(rdata,img_size = 300,target_size = 256):
    rx,ry = rdata.shape
    zero_temp = np.zeros([rx+(img_size - target_size),ry + (img_size - target_size)])
    sp_x = int((img_size - target_size)/2);
    sp_y = int((img_size - target_size)/2);
    zero_temp[sp_x:(rx+sp_x),sp_y:(ry+sp_y)] = rdata
    zx,zy = zero_temp.shape
    x_counter = 0;
    
    for iTemp in range(0,zx+img_size,img_size):
        csxp = iTemp - (sp_x*x_counter)
        cexp = iTemp + img_size - (sp_x*x_counter)
        x_counter = x_counter + 1;
        y_counter = 0;
        for jTemp in range(0,zy+img_size,img_size):
            csyp = jTemp - (sp_y*y_counter)             
            ceyp = jTemp + img_size - (sp_y*y_counter)
            rtemp  = zero_temp[csxp:cexp,csyp:ceyp]
            y_counter = y_counter + 1;     
            yield rtemp

def overlap_cropped_image_saver(rdata,img_size,
                        filename,spath,datatype='uint8'):
    generator_fun = overlap_image_slicer(rdata,img_size,target_size = 256)
    filecounter = 0
    for cropped_img in zip(generator_fun):
        #print(np.count_nonzero(cropped_img), np.array(cropped_img).shape, np.array(cropped_img).shape[1] ** 2)
        fstr = ""
        ftemp = filename.split('.')[0:-1]
        fftemp = fstr.join(ftemp)
        fname = fftemp + '_cropped' + "_{:04d}".format((filecounter)) + '.tif'
        #savename = os.path.join(sfolder,"uint8_{:04d}".format(counter) + ".tif")
        
        csavepath = os.path.join(spath)
        if datatype == 'uint8_norm':
            # with normalization
            saveImages(changer16to8(cropped_img[0]),
                       csavepath,fname,
                       datatype = 'uint8')
        elif datatype == 'histeq':
            saveImages(cropped_img[0],csavepath,fname,
                       datatype = 'uint8')
        else:
            saveImages(cropped_img[0],csavepath,fname,
                   datatype = datatype)
        filecounter +=1;
    #print(filecounter, img_size, cropped_img[0].shape, rdata.shape)
    
def cropped_image_saver(rdata,img_size,
                        filename,spath,datatype='uint8'):
    generator_fun = image_slicer(rdata,img_size)
    filecounter = 0
    for cropped_img in zip(generator_fun):
        fstr = ""
        ftemp = filename.split('.')[0:-1]
        fftemp = fstr.join(ftemp)
        fname = fftemp + '_cropped' + "_{:04d}".format((filecounter)) + '.tif'
        #savename = os.path.join(sfolder,"uint8_{:04d}".format(counter) + ".tif")
        
        csavepath = os.path.join(spath)
        if datatype == 'uint8_norm':
            # with normalization
            saveImages(changer16to8(cropped_img[0]),
                       csavepath,fname,
                       datatype = 'uint8')
        elif datatype == 'histeq':
            saveImages(cropped_img[0],csavepath,fname,
                       datatype = 'uint8')
        else:
            saveImages(cropped_img[0],csavepath,fname,
                   datatype = datatype)
        filecounter +=1
        
def merge_image(tsize,
                f_counter,img_size,mpath,mflist):
    zeroTemp = np.zeros(tsize)
    for iTemp in range(0,tsize[0],img_size):
        for jTemp in range(0,tsize[1],img_size):
            mdata =io.imread(os.path.join(mpath,mflist[f_counter]))
            # if the size of image exceeds boundary of template
            
            if (jTemp + img_size) > tsize[1]:
                mdata = mdata[:,0:(tsize[1]-jTemp)]
            if (iTemp + img_size) > tsize[0]:
                mdata= mdata[0:(tsize[0] - iTemp),:]
                
            zeroTemp[iTemp:iTemp+img_size,jTemp:jTemp+img_size] = mdata
            f_counter+=1
            
    return zeroTemp, f_counter
            

def overlap_merge_image(tsize,fsize,
                f_counter,org_size,img_size,mpath,mflist):
    zeroTemp = np.zeros(tsize)
    scut =int((org_size-img_size)/2/2) # cut the half of margin (cut it from each side)
    pimg_size = img_size + scut*2
    for iTemp in range(0,tsize[0],pimg_size):
        for jTemp in range(0,tsize[1],pimg_size):
            mdata = io.imread(os.path.join(mpath,mflist[f_counter]))
            # if the size of image exceeds boundary of template
            
            mdata= mdata[(scut-1):-1-scut,(scut-1):-1-scut]
            
            
            if (jTemp + pimg_size) > tsize[1]:
                mdata = mdata[:,0:(tsize[1]-jTemp)]
            if (iTemp + pimg_size) > tsize[0]:
                mdata= mdata[0:(tsize[0] - iTemp),:]
            
            zeroTemp[iTemp:iTemp+pimg_size,jTemp:jTemp+pimg_size] = mdata
            
            f_counter+=1
            
    return zeroTemp,f_counter    

#for random forest basically same with overlap_merge_image
def overlap_merge_image_rf(tsize,fsize,
                f_counter,org_size,img_size,mpath,mflist):
    zeroTemp = np.zeros(tsize)
    scut =int((org_size-img_size)/2/2) # cut the half of margin (cut it from each side)
    pimg_size = img_size + scut*2
    for iTemp in range(0,tsize[0],pimg_size):
        for jTemp in range(0,tsize[1],pimg_size):
            mdata =io.imread(os.path.join(mpath,mflist[f_counter]))
            # if the size of image exceeds boundary of template
            mdata = mdata[:,:,1]
            
            mdata= mdata[(scut-1):-1-scut,(scut-1):-1-scut]
            
            
            if (jTemp + pimg_size) > tsize[1]:
                mdata = mdata[:,0:(tsize[1]-jTemp)]
            if (iTemp + pimg_size) > tsize[0]:
                mdata= mdata[0:(tsize[0] - iTemp),:]
            
            zeroTemp[iTemp:iTemp+pimg_size,jTemp:jTemp+pimg_size] = mdata
            
            f_counter+=1
            
    return zeroTemp,f_counter    

def overlap_merge_image_weka2(tsize,fsize,
                f_counter,org_size,img_size,mpath,mflist):
    zeroTemp = np.zeros(tsize)
    shiftTemp = np.zeros(tsize)
    scut =int((org_size-img_size)/2/2) # cut the half of margin (cut it from each side)
    pimg_size = img_size + scut*2
    
    #fsize = number of tiles x,y
    sx = 0;
    
    scut = int(scut/2)
    
    for iTemp in range(0,fsize[0],1):
        sy = 0;
        for jTemp in range(0,fsize[1],1):
            mdata = io.imread(os.path.join(mpath,mflist[f_counter]))
            tile = mdata
            
            
            if iTemp == 0 and jTemp ==0:
                mdata = mdata[0:-scut,0:-scut]
                #breakpoint()                
            if iTemp ==0 and jTemp > 0:
                mdata = mdata[0:-scut,scut:-scut]
                #breakpoint()
                
            if iTemp >0 and jTemp ==0:
                mdata = mdata[scut:-scut,0:-scut]
                
            if iTemp >0 and jTemp >0:
                mdata = mdata[scut:-scut,scut:-scut]   
            
            
            #breakpoint();
            ss = mdata.shape
            img_x = ss[0]
            img_y = ss[1]
            
            
            
            if sy + img_y > tsize[1]:
                mdata=mdata[:,0:tsize[1]-sy]
                
                #breakpoint()
            if sx + img_x > tsize[0]:
                # contain till 242
                mdata= mdata[0:tsize[0]-sx,:]
                
                
                
            
            
            zeroTemp[sx:sx+img_x , sy:sy+img_y] = mdata    
            
            
            f_counter+=1
            
            sy = sy+img_y
            
        
        sx = sx+img_x
    
    return zeroTemp,f_counter        


def overlap_merge_image_weka(tsize,fsize,
                f_counter,org_size,img_size,mpath,mflist):
    zeroTemp = np.zeros(tsize)
    shiftTemp = np.zeros(tsize)
    scut =int((org_size-img_size)/2/2) # cut the half of margin (cut it from each side)
    pimg_size = img_size + scut*2
    
    for iTemp in range(0,tsize[0],pimg_size):
        for jTemp in range(0,tsize[1],pimg_size):
            mdata =io.imread(os.path.join(mpath,mflist[f_counter]))
            
            
            mdata= mdata[(scut):-1-scut+1,(scut):-1-scut+1]    
            
            
            if (jTemp + pimg_size) > tsize[1]:
                mdata = mdata[:,0:(tsize[1]-jTemp)]
            if (iTemp + pimg_size) > tsize[0]:
                mdata= mdata[0:(tsize[0] - iTemp),:]
            
 
            zeroTemp[iTemp:iTemp+pimg_size,jTemp:jTemp+pimg_size] = mdata
            
            f_counter+=1
            
    
    shiftTemp[0:-1-scut,0:-1-scut] = zeroTemp[scut:-1,scut:-1];        
    return shiftTemp,f_counter        
    
# save the image into unsigned int 8 format
def saveImages(rdata,rpath,fname,datatype = 'uint8'):
    if not os.path.exists(rpath):
        os.mkdir(rpath)
    io.imsave(os.path.join(rpath,fname),
                  rdata.astype(datatype),
                  check_contrast = False)    


def save3dImages(rdata, rpath, fname, datatype='uint8'):
    if not os.path.exists(rpath):
        os.makedirs(rpath)
    tif.imsave(os.path.join(rpath,fname),
                  rdata.astype(datatype),
                  bigtiff=True)    

                  

def normalize_data(rdata,r_format = 16):
    rdata.astype('double')
    enumerator = rdata - np.min(rdata)
    denominator = np.max(rdata) - np.min(rdata)
    norm_img = (enumerator / denominator ) *( pow(2,r_format) -1)
    return norm_img

# save the binary files into unsigned int 8 format 
# 1 -> 255
def saveCheckImages(rdata,rpath,fname):
    if not os.path.exists(rpath):
        os.mkdir(rpath)
    #norm_data = normalize_data(rdata,r_format = 8)
    norm_data = rdata * 255
    io.imsave(os.path.join(rpath,fname),
              norm_data.astype('uint8'),
              check_contrast = False)

def changer16to8(data):
    dataf = data.astype(float)
    dmin = np.min(data)
    dmax = np.max(data)
    np.clip(dataf,dmin,dmax,out = dataf)
    dataf -= dmin
    datab =  ((255./(dmax-dmin)) * dataf).astype(np.uint8)
    return datab


def preprocess_local_threshold(rdata,mean_val):
    mask_mask = create_ringmask(rdata)
    masked_img = rdata*mask_mask
    
    out_img = np.invert(mask_mask)* mean_val
    outimg  = masked_img + out_img
    
    return outimg
    

#rangenbit_changer(train_path,'cropped_test')
def rangenbit_changer(path,folder):
    sfolder = os.path.join(path,folder + '_uint8')
    if not os.path.exists(sfolder):
        os.mkdir(sfolder)
    
    super_path = os.path.join(path,folder)
    flist = os.listdir(super_path)
    #flist.sort(key = lambda x: int(os.path.splitext(x)[0]))
    counter = 0
    
    for iTemp in tqdm(range(len(flist))):
        data = io.imread(os.path.join(super_path,flist[iTemp]))
        dataf = data.astype(float)
        dmin = np.min(data)
        dmax = np.max(data)
        np.clip(dataf,dmin,dmax,out=dataf)
        dataf -= dmin
        datab = ((255./(dmax-dmin)) * dataf).astype(np.uint8)
        #savename = os.path.join(sfolder,flist[iTemp])
        savename = os.path.join(sfolder,"uint8_{:04d}".format(counter) + ".tif")
        counter += 1
        io.imsave(savename,datab)    
    
    
    