# MRI + Deep Learning Coding Challenge

This repository was created for a 72-hour coding challenge in which the goal was to improve MRI reconstruction with deep learning techniques. More details on this challenge and my submitted write-up can be found at XXXXX.


Here is one example result, a slice of 3D volume from an MRI reconstruction. At left is the original image (left), the blurred image (middle), and the model result (right). As discussed in the write-up, there are multiple methods which I suspect would deliver improved model performance; however, I did not implement them within the time constraint.
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


### Repository Overview

In this table, task refers to each of three parts of the challenge: (I) DICOM I/O (II) Simulated fast acquisitions (III) Deep learning super resolution model

File | Task | Description
--- | --- | ---
requirements.txt | N/A | System packages. Run `pip install -r  requirements.txt`
dcm_to_h5.py | I | Conversion of dicom files from directory to a single hdf5 file

Note: To run this code, one must also download MRI dicom files in a directory called `data/`. The data I used for my coding challenge can be found at <http://old.mridata.org/fullysampled/knees>

Example commands
* File I/O
`python dcm_to_h5 --i path_to_input_dicom_directory --h path_to_output_hdf5_file`
* Model training
`python main.py` *args
For a list of command line arguments, please see `parser_model.py`

