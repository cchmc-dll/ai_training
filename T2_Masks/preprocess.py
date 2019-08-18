"""
Tools for converting, normalizing, and fixing the T2_masks data.
"""


import glob
import os
import errno
import warnings
import shutil
import SimpleITK as sitk
import numpy as np
from nipype.interfaces.ants import N4BiasFieldCorrection

from T2_masks.train import config
from T2_masks.resample_isotropically import resample_sitk_image
from dltk.io.augmentation import *
from dltk.io.preprocessing import *
from matplotlib import pyplot as plt
import pandas as pd


def append_basename(in_file, append):
    dirname, basename = os.path.split(in_file)ZAS
    base, ext = basename.split(".", 1)
    return os.path.join(dirname, base + append + "." + ext)


def get_background_mask(in_folder, out_file, truth_name="Label"):
    """
    This function computes a common background mask for all of the data in a subject folder.
    :param in_folder: a subject folder from the T2_masks dataset.
    :param out_file: an image containing a mask that is 1 where the image data for that subject contains the background.
    :param truth_name: how the truth file is labeled int he subject folder
    :return: the path to the out_file
    """
    background_image = None
    for name in config["all_modalities"] + [truth_name]:
        image = sitk.ReadImage(get_image(in_folder, name))
        if background_image:
            if name == truth_name and not (image.GetOrigin() == background_image.GetOrigin()):
                image.SetOrigin(background_image.GetOrigin())
            background_image = sitk.And(image == 0, background_image)
        else:
            background_image = image == 0
    sitk.WriteImage(background_image, out_file)
    return os.path.abspath(out_file)


def convert_image_format(in_file, out_file):
    sitk.WriteImage(sitk.ReadImage(in_file), out_file)
    return out_file


def window_intensities(in_file, out_file, min_percent=1, max_percent=99):
    image = sitk.ReadImage(in_file)
    image_data = sitk.GetArrayFromImage(image)
    out_image = sitk.IntensityWindowing(image, np.percentile(image_data, min_percent), np.percentile(image_data,
                                                                                                     max_percent))
    sitk.WriteImage(out_image, out_file)
    return os.path.abspath(out_file)

def resize_image(in_file,out_file):
    """ Resizes image into Reference image size
    :param in_file: input file path
    :param out_file: output file path
    :return: file path to the bias corrected image
    """
    print(in_file)
    print(out_file)
    sitk_t1 = sitk.ReadImage(in_file)
    t1 = sitk.GetArrayFromImage(sitk_t1)
    # Resizing image to [128, 256, 256] required padding
    t1_padded = resize_image_with_crop_or_pad(t1, [88, 256, 256], mode='symmetric')
    # Visualise using matplotlib.
    f, axarr = plt.subplots(1, 2, figsize=(15,5));
    f.suptitle('Automatically crop or pad a volume image')
    axarr[0].imshow(np.squeeze(t1[t1.shape[0]//2, :, :]), cmap='gray');
    axarr[0].axis('off')
    axarr[0].set_title('Original image {}'.format(t1.shape))
    axarr[1].imshow(np.squeeze(t1_padded[t1_padded.shape[0]//2, :, :]), cmap='gray');
    axarr[1].axis('off')
    axarr[1].set_title('Padded to {}'.format(t1_padded.shape))
    f.subplots_adjust(wspace=0.05, hspace=0, top=0.8)
    plt.show();
    sitk.WriteImage(t1_padded, out_file)
    return os.path.abspath(out_file)

def resample_type4(in_file, out_file):
    """Apply a Gaussian filter before downsampling.
    By the Nyquist-Shannon theorem we can only reliably reproduce
    information up to the Nyquist frequency. As this frequency
    is scaled by the same factor as the downsampling, we cannot have
    frequencies above alpha*Nyquist frequence in the image.
    We use a Gaussian filter as a low-pass filter.
    As the Gaussian is the same in the frequency domain,
    we can reduce the amount of information in a 85um grid,
    to one of 200um.
    Approximately, sigma = 85/200 *1/(2*sqrt(pi)), where the normalization
    is from the choice of Gaussian. Hence sigma ~ 0.11, or ~1.412 image units.
    """
    sigma = 0.2
    image =  sitk.ReadImage(in_file)
    # 0,1,2 <-> (x,y,z)
    image = sitk.RecursiveGaussian(image, sigma=sigma*0.2, direction=0)
    image = sitk.RecursiveGaussian(image, sigma=sigma*0.2, direction=1)

    #image = sitk.IntensityWindowing(image,
    #                                lower_bound, upper_bound, 0, 255)
    #image = sitk.Cast(image, sitk.sitkUInt8)

    resampled_image = resample_sitk_image(
            image, spacing=(0.2, 0.2, 1),
            interpolator='linear', fill_value=0)
    t1 = sitk.GetArrayFromImage(resampled_image)
    #print('t1 shape' + t1.shape)
      # Visualise using matplotlib.
    f, axarr = plt.subplots(1, 1, figsize=(15,5));
    f.suptitle('Raw image')
    axarr.imshow(np.squeeze(t1[t1.shape[0]//2, :, :]), cmap='gray');
    axarr.axis('off')
    axarr.set_title('Original image {}'.format(t1.shape))
    f.subplots_adjust(wspace=0.05, hspace=0, top=0.8)
    plt.show()
    sitk.WriteImage(resampled_image, out_file)




def correct_bias(in_file, out_file):
    """
    Corrects the bias using ANTs N4BiasFieldCorrection. If this fails, will then attempt to correct bias using SimpleITK
    :param in_file: input file path
    :param out_file: output file path
    :return: file path to the bias corrected image
    """
    print(in_file)
    print(out_file)
    correct = N4BiasFieldCorrection()
    #correct.inputs.input_image = in_file
    #correct.inputs.output_image = out_file
    #try:
    #    done = correct.run()
    #    print(done)
    #    return done.outputs.output_image
    #except IOError as ioex:
        #print('errno:', ioex.errno)
        #print('err message:', os.strerror(ioex.errno))

        #warnings.warn(RuntimeWarning("ANTs N4BIasFieldCorrection could not be found."
        #                             "Will try using SimpleITK for bias field correction"
        #                             " which will take much longer. To fix this problem, add N4BiasFieldCorrection"
        #                             " to your PATH system variable. (example: EXPORT ${PATH}:/path/to/ants/bin)"))
    raw_image = sitk.ReadImage(in_file)
    print("Pixel Type    {}".format(raw_image.GetPixelID()))
    print("Size          {}".format(raw_image.GetSize()))
    print("Origin        {}".format(raw_image.GetOrigin()))
    print("Spacing       {}".format(raw_image.GetSpacing()))
    print("Direction     {}".format(raw_image.GetDirection()))
    output_image = sitk.N4BiasFieldCorrection(raw_image)
    sitk.WriteImage(output_image, out_file)
    return os.path.abspath(out_file)
    
    #raw_image = sitk.ReadImage(in_file)
    #t1 = sitk.GetArrayFromImage(raw_image)
    ##sitk.Show(raw_image,'Raw Image')
    # # Visualise using matplotlib.
    #f, axarr = plt.subplots(1, 1, figsize=(15,5));
    #f.suptitle('Raw image')
    #axarr.imshow(np.squeeze(t1[t1.shape[0]//2, :, :]), cmap='gray');
    #axarr.axis('off')
    #axarr.set_title('Original image {}'.format(t1.shape))
    #f.subplots_adjust(wspace=0.05, hspace=0, top=0.8)
    #plt.show()

   

def rescale(in_file, out_file, minimum=0, maximum=20000):
    image = sitk.ReadImage(in_file)
    sitk.WriteImage(sitk.RescaleIntensity(image, minimum, maximum), out_file)
    return os.path.abspath(out_file)


def get_image(subject_folder, name):
    file_card = os.path.join(subject_folder, "*" + name + ".nii")
    try:
        return glob.glob(file_card)[0]
    except IndexError:
        raise RuntimeError("Could not find file matching {}".format(file_card))


def background_to_zero(in_file, background_file, out_file):
    sitk.WriteImage(sitk.Mask(sitk.ReadImage(in_file), sitk.ReadImage(background_file, sitk.sitkUInt8) == 0),
                    out_file)
    return out_file


def check_origin(in_file, in_file2):
    image = sitk.ReadImage(in_file)
    image2 = sitk.ReadImage(in_file2)
    if not image.GetOrigin() == image2.GetOrigin():
        image.SetOrigin(image2.GetOrigin())
        sitk.WriteImage(image, in_file)


def normalize_image(in_file, out_file, bias_correction=True):
    if bias_correction:
        correct_bias(in_file, out_file)
    else:
        #shutil.copy(in_file, out_file)
        image = sitk.ReadImage(in_file)
        sitk.WriteImage(image, out_file)

    return out_file


def convert_T2_masks_folder(in_folder, out_folder, truth_name="Label",
                         no_bias_correction_modalities=None):
    for name in config["all_modalities"]:
        image_file = get_image(in_folder, name)
        out_file = os.path.abspath(os.path.join(out_folder, name + ".nii.gz"))
        perform_bias_correction = no_bias_correction_modalities and name not in no_bias_correction_modalities
       # normalize_image(image_file, out_file, bias_correction=perform_bias_correction)
        resample_type4(image_file,out_file)
    # copy the truth file
    try:
        truth_file = get_image(in_folder, truth_name)
    except RuntimeError:
        truth_file = get_image(in_folder, truth_name.split("_")[0])
    out_file = os.path.abspath(os.path.join(out_folder, "truth.nii.gz"))
    shutil.copy(truth_file, out_file)
    check_origin(out_file, get_image(in_folder, config["all_modalities"][0]))


def convert_T2_masks_data(T2_masks_folder, out_folder, overwrite=False, no_bias_correction_modalities=("Flair",)):
    """
    Preprocesses the T2_masks data and writes it to a given output folder. Assumes the original folder structure.
    :param T2_masks_folder: folder containing the original T2_masks data
    :param out_folder: output folder to which the preprocessed data will be written
    :param overwrite: set to True in order to redo all the preprocessing
    :param no_bias_correction_modalities: performing bias correction could reduce the signal of certain modalities. If
    concerned about a reduction in signal for a specific modality, specify by including the given modality in a list
    or tuple.
    :return:
    """
    print(T2_masks_folder)
    print(out_folder)
    for subject_folder in glob.glob(os.path.join(T2_masks_folder, "*")):
        if os.path.isdir(subject_folder):
            subject = os.path.basename(subject_folder)
            new_subject_folder = os.path.join(out_folder,
                                              subject)
            if not os.path.exists(new_subject_folder) or overwrite:
                if not os.path.exists(new_subject_folder):
                    print('Creating folder here:',new_subject_folder,'\n')
                    os.makedirs(new_subject_folder)
                convert_T2_masks_folder(subject_folder, new_subject_folder,
                                     no_bias_correction_modalities=no_bias_correction_modalities)
