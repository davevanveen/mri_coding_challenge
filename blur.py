import os
import numpy as np
from scipy.ndimage import gaussian_filter 
import utils_io as utils

DEFAULT_SIG = 5

def blurring3d(arr_orig, sigma=DEFAULT_SIG):
    '''
    Applies 2D Guassian blurring filter to each slice 
    
    INPUTS: arr_orig:	3D array to slice and then filter by slice
    		sigma:		standard deviation for Gaussian kernel
    OUTPUT: 3D array, blurred version of arr_3d  
    '''

    arr_blur = np.zeros_like((arr_orig)) # initialize blurred array

    for i in range(len(arr_orig)):
    	img = arr_orig[i]
    	arr_blur[i] = gaussian_filter(img, sigma) # blur each slice
    return arr_blur


if __name__ == '__main__':
	# load original array
	cwd = os.path.dirname(os.path.abspath('blur.py'))

	num = 1
	path_in = '/data/input_dicom/P%d_dcm/' % num
	arr = utils.load_orig_scan(cwd + path_in) # 3D array of scan data

	# blur array, save as dicom
	path_out = 'data/blurred_dicom/P%d_dcm_blur/' % num
	blurred = blurring3d(arr)
	utils.write_vol_to_dicom(blurred, path_out)
