'''
Converts dicom file(s) to hdf5 file. File paths specified via command line.

Input hdf5 (.h5 file) format: 	dtype=float32, range=[0,1]
Output dcm format: 				dtype=int16, range=unspecified
'''

import os
import parser_io as parser
import h5py
import utils_io as utils

# define paths
cwd = os.path.dirname(os.path.abspath('dcm_to_h5.py'))
args = parser.parse_args()
path_in = '/' + args.i # path to input dicom file(s) 
path_h5 = args.h # path to output hdf5 file
file_path = path_h5 #+ '.h5'

# load data, convert, write to hdf5
vol = utils.load_orig_scan(cwd + path_in) # 3D array of scan data
vol = utils.convert_to_h5_format(vol)
utils.write_h5(file_path, vol)