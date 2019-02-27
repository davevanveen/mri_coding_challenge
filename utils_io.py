'''Utilities module with functions for data I/O, conversion, etc.'''

import numpy as np
import pydicom
import os
import errno
import h5py
from pydicom.uid import ImplicitVRLittleEndian
from pydicom.dataset import Dataset, FileDataset

DIM = 512 # dimension (height, width) of each dicom slice. TODO: soft code this


def load_orig_scan(path):
    '''
    Loads original scans obtained from http://old.mridata.org/fullysampled/knees
    
    INPUT:  Path containing dicom file(s), where each file is one scan slice
    OUTPUT: A single 3D numpy array containing all scan slices  
    '''
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    vol = np.stack([s.pixel_array for s in slices])
    return np.array(vol)


def get_arr_info_rescale(arr):
    '''Return minimum value and dynamic range of input array'''
    min_val = arr.min()
    max_val = arr.max()
    dr = max_val - min_val
    return min_val, dr


def convert_to_h5_format(vol):
    '''Convert array into .h5 format, i.e. dtype=float32, range=[0,1]'''
    dtype = np.float32
    vol = np.array(vol, dtype=dtype)
    
    min_val, dr = get_arr_info_rescale(vol)
    vol = (1/dr)*vol - (min_val/dr)
    return vol


def convert_to_dcm_format(vol, temp): # vol to be converted, orig format values
    '''
    Convert array vol into the format of array orig
    INPUTS  vol:     Array in .h5 format, i.e. dtype=float32, range=[0,1]
            temp:    Array in template .dcm format, i.e. dtype=int16, range=unspecified
    OUTPUT  Appropriately scaled array 
     '''
    dtype = temp.dtype
    max_val = temp.max()
    vol = vol*max_val
    vol = vol.astype(dtype)
    return vol


def write_slice_to_dicom(file_path, arr):
    '''
    Write 2D numpy arr to an individual dicom file
    Note: Minimal dicom meta-data included. This should be improved in future work.
    '''
    file_meta = Dataset()
    file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = '1.1'
    file_meta.MediaStorageSOPInstanceUID = '1.2'
    file_meta.ImplementationClassUID = '1.3'
    
    ds = FileDataset(file_path, {}, file_meta=file_meta, preamble=b'\x00'*128)
    ds.BitsAllocated = 16
    ds.Rows = DIM
    ds.Columns = DIM
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PixelData = arr.tostring()
    ds.save_as(file_path, write_like_original=True)    
    return


def write_vol_to_dicom(vol, path):
    '''Write 3D numpy array to individual dicom files in given path'''
    for i in range(len(vol)):
        img = vol[i]
        idx = str(i).zfill(3)
        fn_out = 'sec_%s.dcm' % idx
        if not os.path.exists(os.path.dirname(path)):
        	create_path(path)
        write_slice_to_dicom(path + fn_out, img)
    return


def load_processed_scan(path):
    '''
    Loads processed scans obtained from user pre-processing.
    
    INPUT:  Path containing dicom file(s), where each file is one scan slice
    OUTPUT: A single 3D numpy array containing all scan slices

    Note:   When converting 3D arrays to dicom, I didn't know how to best encode
            Slice Location. Since processed scans were already stored in sorted
            order, below I use the .dcm filename strings to sort each slice into
            its corresponding index of the 3D array.
            Thus this function is different from load_orig_scan(), which loads
            dicom files according to Slice Location.
            I settled for this hacky technique due to time contraint, but it 
            should be improved in future work.   
    '''
    file_list = os.listdir(path)
    vol = np.zeros((len(file_list),DIM,DIM), dtype='int16')

    for i in range(len(file_list)):
        idx = int(file_list[i].split('.dcm')[0].split('sec_')[1]) # extract slice index
        dcm_slice = pydicom.read_file(path + file_list[i])
        np_slice = np.array(dcm_slice.pixel_array)
        vol[idx] = np_slice
    return vol


def write_h5(file_path, data):
    '''Write array data to .h5 file in a specified path'''
    print(file_path)
    if not os.path.exists(os.path.dirname(file_path)):
    	create_path(file_path)
    hf = h5py.File(file_path, 'w')
    hf.create_dataset('scan', data=data)
    hf.close()
    return


def read_h5(file_path):
    '''Read specified .h5 file and return data in numpy array'''
    hf = h5py.File(file_path, 'r')
    h5_vol = np.array(hf.get('scan')) #convert to numpy array
    hf.close()
    return h5_vol


def get_middle_slice(arr):
    '''Return middle 2D slice of 3D array'''
    num_slices = len(arr)
    idx_mid = int(num_slices / 2 if (num_slices % 2 == 0) else (num_slices + 1) / 2)
    return arr[idx_mid]


def create_path(path):
	try:
		os.makedirs(os.path.dirname(path))
	except OSError as exc: # guard against race condition
		if exc.errno != errno.EEXIST:
			raise
