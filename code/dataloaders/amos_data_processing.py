import glob
import os

import h5py
import numpy as np
import SimpleITK as sitk
import ipdb
slice_num = 0
mask_path = sorted(glob.glob("/data_hdd/users/pengzelin/SSL4MIS/data/Dataset702_AbdomenMR/imagesTs/*.nii.gz"))
#ipdb.set_trace()
for case in mask_path:
    img_itk = sitk.ReadImage(case)
    origin = img_itk.GetOrigin()
    spacing = img_itk.GetSpacing()
    direction = img_itk.GetDirection()
    image = sitk.GetArrayFromImage(img_itk)
    msk_path = case.replace("imagesTs", "labelsTs").replace("_0000.nii.gz", ".nii.gz")
    #ipdb.set_trace()
    if os.path.exists(msk_path):
        #print(msk_path)
        msk_itk = sitk.ReadImage(msk_path)
        mask = sitk.GetArrayFromImage(msk_itk)
        image = (image - image.min()) / (image.max() - image.min())
        print(image.shape)
        #ipdb.set_trace()
        image = image.astype(np.float32)
        item = case.split("/")[-1].split(".")[0]
        if image.shape != mask.shape:
            print("Error")
        print(item)

        #for slice_ind in range(image.shape[0]):
        ##write to h5 file

        # f = h5py.File(
        #     '/data_hdd/users/pengzelin/SSL4MIS/data/AbdomenMR/test/{}.h5'.format(item), 'w')
        # f.create_dataset(
        #     'image', data=image, compression="gzip")
        # f.create_dataset('label', data=mask, compression="gzip")
        # f.close()
        #slice_num += 1

        ##write to h5 file for each slice file
        
        for slice_ind in range(image.shape[0]):
            f = h5py.File(
                '/data_hdd/users/pengzelin/SSL4MIS/data/AbdomenMR/test/slice/{}_slice_{}.h5'.format(item, slice_ind), 'w')
            f.create_dataset(
                'image', data=image[slice_ind], compression="gzip")
            f.create_dataset('label', data=mask[slice_ind], compression="gzip")
            f.close()
            slice_num += 1
print("Converted all AMOS volumes to 2D slices")
print("Total {} slices".format(slice_num))
