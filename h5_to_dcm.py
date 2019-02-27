'''
Converts hdf5 file to dicom file(s). File paths specified via command line.

Input hdf5 (.h5 file) format: 	dtype=float32, range=[0,1]
Output dcm format: 				dtype=int16, range=unspecified
Data from the .h5 file is converted to dcm format according to a template. 
'''

import os
import parser_io as parser
import h5py
import utils_io as utils

# define paths
cwd = os.path.dirname(os.path.abspath('h5_to_dcm.py'))
args = parser.parse_args()
path_in = '/' + args.i 
path_h5 = '' + args.h
path_out = args.o 

# write new dicom files
h5_vol = utils.read_h5(path_h5) # 3D array of scan data 
temp_vol = utils.load_orig_scan(cwd + path_in) # template array
dcm_vol = utils.convert_to_dcm_format(h5_vol, temp_vol) # format according to template
utils.write_vol_to_dicom(dcm_vol, path_out)