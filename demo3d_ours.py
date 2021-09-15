import GeodisTK
import time
import psutil
import numpy as np
import SimpleITK as sitk 
import matplotlib.pyplot as plt
from PIL import Image
import os

def _itk_read_image_from_file(image_path):
    return sitk.ReadImage(image_path, sitk.sitkFloat32)

def _itk_read_array_from_file(image_path):
    return sitk.GetArrayFromImage(_itk_read_image_from_file(image_path))

def _itk_write_array_to_file(image_array, ref_image, output_path):
    itk_image = sitk.GetImageFromArray(image_array)
    itk_image.SetSpacing(ref_image.GetSpacing())
    itk_image.SetOrigin(ref_image.GetOrigin())
    itk_image.SetDirection(ref_image.GetDirection())
    sitk.WriteImage(itk_image, output_path, True)

def geodesic_distance_3d(I, S, spacing, lamb, iter):
    '''
    Get 3D geodesic disntance by raser scanning.
    I: input image array, can have multiple channels, with shape [D, H, W] or [D, H, W, C]
       Type should be np.float32.
    S: binary image where non-zero pixels are used as seeds, with shape [D, H, W]
       Type should be np.uint8.
    spacing: a tuple of float numbers for pixel spacing along D, H and W dimensions respectively.
    lamb: weighting betwween 0.0 and 1.0
          if lamb==0.0, return spatial euclidean distance without considering gradient
          if lamb==1.0, the distance is based on gradient only without using spatial distance
    iter: number of iteration for raster scanning.
    '''
    return GeodisTK.geodesic3d_raster_scan(I, S, spacing, lamb, iter)

def demo_geodesic_distance3d():
    input_name = "data/img3d.nii.gz"
    img = sitk.ReadImage(input_name)
    I   = sitk.GetArrayFromImage(img)
    spacing_raw = img.GetSpacing()
    spacing = [spacing_raw[2], spacing_raw[1],spacing_raw[0]]
    I = np.asarray(I, np.float32)
    I = I[18:38, 63:183, 93:233 ]
    S = np.zeros_like(I, np.uint8)
    S[10][60][70] = 1
    # t0 = time.time()
    # D1 = GeodisTK.geodesic3d_fast_marching(I,S, spacing)
    t1 = time.time()
    D2 = geodesic_distance_3d(I,S, spacing, 1.0, 4)
    # dt1 = t1 - t0
    dt2 = time.time() - t1
    D3 = geodesic_distance_3d(I,S, spacing, 0.0, 4)
    # print("runtime(s) fast marching {0:}".format(dt1))
    print("runtime(s) raster scan   {0:}".format(dt2))  

    # import pdb; pdb.set_trace()
    print("inspecting")

def demo_geodesic_distance3d_ours():
    """inspect our training samples"""

    imgpath = "/data/tianmu/data/dynamic_segmentation/brats2015/train/all/sample_54686_image_norm_crop_resize.mha"
    labelpath = "/data/tianmu/data/dynamic_segmentation/brats2015/train/all/sample_54686_label_binary_crop_resize.mha"
    savepath = "/data/tianmu/data/dynamic_segmentation/brats2015/train/all_geodesic"

    img = _itk_read_image_from_file(imgpath)
    label = _itk_read_image_from_file(labelpath)

    spacing_raw = img.GetSpacing()
    spacing = [spacing_raw[2], spacing_raw[1],spacing_raw[0]]
    
    imgdata = _itk_read_array_from_file(imgpath)
    labeldata = _itk_read_array_from_file(labelpath)
    
    seedsdata = np.zeros(labeldata.shape)

    fg_indices = np.where(labeldata == 1)
    bg_indices = np.where(labeldata == 0)
    num_clicks_fg = 10
    num_clicks_bg = 10
    fg_selected = np.random.choice(fg_indices[0].shape[0], num_clicks_fg, replace=False).tolist()
    bg_selected = np.random.choice(bg_indices[0].shape[0], num_clicks_bg, replace=False).tolist()

    for fg in fg_selected:
        seedsdata[fg_indices[0][fg], fg_indices[1][fg], fg_indices[2][fg]] = 1
    
    for bg in bg_selected:
        seedsdata[bg_indices[0][bg], bg_indices[1][bg], bg_indices[2][bg]] = 0
    seedsdata = seedsdata.astype(np.uint8)

    t1 = time.time()
    D_spatial = geodesic_distance_3d(imgdata, seedsdata, spacing, 0.0, 4)
    t2 = time.time()
    dt2 = t2 - t1
    print(f"D_spatial takes {dt2} secs")
    D_gradient = geodesic_distance_3d(imgdata, seedsdata, spacing, 1.0, 4)
    t3 = time.time()
    dt3 = t3 - t2
    print(f"D_gradient takes {dt3} secs")
    D_spatial_gradient = geodesic_distance_3d(imgdata, seedsdata, spacing, 0.5, 4)
    t4 = time.time()
    dt4 = t4 - t3
    print(f"D_spatial_gradient takes {dt4} secs")

    _itk_write_array_to_file(imgdata, img, os.path.join(savepath, "sample_54686_image_original.mha"))
    _itk_write_array_to_file(labeldata, img, os.path.join(savepath, "sample_54686_label_original.mha"))
    _itk_write_array_to_file(seedsdata, img, os.path.join(savepath, "sample_54686_seed.mha"))
    _itk_write_array_to_file(D_spatial, img, os.path.join(savepath, "sample_54686_geod_spatial.mha"))
    _itk_write_array_to_file(D_gradient, img, os.path.join(savepath, "sample_54686_geod_gradient.mha"))
    _itk_write_array_to_file(D_spatial_gradient, img, os.path.join(savepath, "sample_54686_geod_spatial_gradient.mha"))

if __name__ == '__main__':
    demo_geodesic_distance3d_ours()
