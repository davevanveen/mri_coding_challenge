import os
from os import listdir
from os.path import join
import PIL
from PIL import Image
import numpy as np
import torch
import torchvision
from imgaug import augmenters as iaa
import imgaug as ia 
from torchvision.transforms import Compose, ToTensor, Resize, Grayscale
import pydicom


cwd = os.path.dirname(os.path.abspath('utils_model.py'))
DIM = 512 # dimension (height, width) of each dicom slice. TODO: soft code this


def input_transform(upscale_factor=2, im_dim=DIM):
    '''Define transform for input data, i.e. blurred'''
    return Compose([
            BlurTransform(),
            Resize(im_dim // upscale_factor),
            ToTensor(),
            ])


def target_transform():
    '''Define transform for target data'''
    return Compose([
            ToTensor(),
            ])


def load_img(file_path):
    '''Loads dicom file and returns 2D PIL Image
    Note: 	Must convert dtype and range to maintain contrast'''
    dcm_slice = pydicom.read_file(file_path)
    np_slice = np.array(dcm_slice.pixel_array)
    temp = np_slice.astype('float16') / np_slice.max()
    im_slice = Image.fromarray(np.uint32(temp*255), mode='I')
    return im_slice


'''TODO: Combine the below three functions into one which takes argument path

   Note: Neither training nor testing directories are included in submisson, thus
         user must hard code his or her own directories to use these functions.
         Please see plot.ipynb for a demonstration of inference on trained model.'''
def get_training_set(upscale_factor):
	train_dir = cwd + '/data/train/'
	return DatasetFromFolder(train_dir,
                             input_transform = input_transform(upscale_factor),
                             target_transform = target_transform())

def get_test_set(upscale_factor):
	test_dir = cwd + '/data/test/'
	return DatasetFromFolder(test_dir,
                             input_transform = input_transform(upscale_factor),
                             target_transform = target_transform())
def get_inference_slice(upscale_factor):
    '''Load single .dcm for inference by trained net'''
    inf_dir = cwd + '/data/output_dicom/test_read/'
    return DatasetFromFolder(inf_dir,
                             input_transform = input_transform(upscale_factor),
                             target_transform = target_transform())


def is_dicom_file(filename):
    return any(filename.endswith(extension) for extension in ['.dcm'])


class DatasetFromFolder(torch.utils.data.Dataset):
    def __init__(self, img_dir, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.filenames = [join(img_dir, x) for x in listdir(img_dir) if is_dicom_file(x)]
        self.input_transform = input_transform
        self.target_transform = target_transform
        return

    def __getitem__(self, index):
        img_in = load_img(self.filenames[index])
        target = img_in.copy()
        if self.input_transform:
            img_in = self.input_transform(img_in)
        if self.target_transform:
            target = self.target_transform(target)

        return img_in, target

    def __len__(self):
        return len(self.filenames)


class BlurTransform:
    '''
    Custom transform because torchvision does not support blurring
    Used imgaug library: https://github.com/aleju/imgaug
    Note:   For Task III, I did not use the output from Task II. If
            I had used that output, I would have needed to match up 
            corresponding input (blurred) slices and target (original)
            slices (separate files) in PyTorch. How to execute this 
            was not immediately obvious to me; as such, I built 
            BlurTransform() in a PyTorch transform function.
    '''
    def __init__(self):
        self.aug = iaa.Sequential([
                iaa.GaussianBlur(sigma=5)
        ])

    def __call__(self, img):
        img = np.array(img)
        return Image.fromarray(self.aug.augment_image(img), mode='I')

def set_dtype(CUDA):
    if CUDA: # if cuda is available
        return torch.cuda.FloatTensor
    else:
        return torch.FloatTensor