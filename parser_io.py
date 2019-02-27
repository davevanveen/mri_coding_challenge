'''Parser for handiling data I/O from command line'''

import argparse
import json

def parse_args(config_file="configs_io.json"):

	# default file paths if not entered by user
	default_configs = json.load(open(config_file))
	i = default_configs["input_dicom_path"]
	h = default_configs["hdf5_path"]
	o = default_configs["output_dicom_path"]

	parser = argparse.ArgumentParser()
	
	parser.add_argument('--i', type = str, default = i, \
		help = '--input-dicom, i.e. path to input DICOM directory.')
	parser.add_argument('--h', type = str, default = h, \
		help = 'if converting from dicom: --output-hdf5, i.e. path to output hdf5 file. \
				if converting to dicom: --input-hdf5, i.e. path to input hdf5 file.')
	parser.add_argument('--o', type = str, default = o, \
		help = '--output-dicom, i.e. path to output DICOM directory.')

	args = parser.parse_args()

	# TODO: Add error handling if user enters path that does not exist

	return args