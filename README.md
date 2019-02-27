# MRI + Deep Learning Coding Challenge

This repository was created for a 72-hour coding challenge in which the goal was to improve MRI reconstruction with deep learning techniques. More details on this challenge and my submitted write-up can be found [here](https://github.com/davevanveen/mri_coding_challenge/blob/master/writeup.pdf).


Below is an example slice of 3D volume from an MRI reconstruction. At left is the original image (left), the blurred image (middle), and the model result (right). As discussed in the write-up, there are multiple methods which I suspect would deliver improved model performance.
<img src="https://github.com/davevanveen/mri_coding_challenge/blob/master/plots/orig_v_blur_v_result.png" width="800">


### Repository Overview

In this table, task refers to each of three parts of the challenge: (I) DICOM I/O (II) Simulated fast acquisitions (III) Deep learning super resolution model

File | Task | Description
--- | --- | ---
requirements.txt | N/A | System packages. Run `pip install -r  requirements.txt`
dcm_to_h5.py | I | Conversion of dicom files from directory to a single hdf5 file
h5_to_dcm.py | I | Conversion of single hdf5 file to individual dicom files
utils_io.py | I, II | Functions for performing data I/O
parser_io.py | I | Parser for command line input of data I/O tasks
configs_io.json | I | Default configurations for parser_io
blur.py | II | Gaussian blurring filter to each slice of 3D scan
model.py | III | Definition of neural network using PyTorch
main.py | III | Main script for training and testing model
utils_model.py | III | Functions and class definitions for model
parser_model.py | III | Parser for command line input of model training parameters
configs_model.json | III | Default configurations for parser_model
prototype.ipynb | II, III | Code for plotting figure output

Note: To run this code, one must also download MRI dicom files in a directory called `data/`. The data I used for my coding challenge can be found at <http://old.mridata.org/fullysampled/knees>


### Example commands
* File I/O
`python dcm_to_h5 --i path_to_input_dicom_directory --h path_to_output_hdf5_file`
* Model training
`python main.py *args`.
For a list of command line arguments, please see `parser_model.py`



