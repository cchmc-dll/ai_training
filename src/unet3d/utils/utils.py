import pickle
import os
import collections

import nibabel as nib
import numpy as np
from nilearn.image import reorder_img, new_img_like

from .nilearn_custom_utils.nilearn_utils import crop_img_to
from .sitk_utils import resample_to_spacing, calculate_origin_offset
from random import shuffle
from sklearn.model_selection import train_test_split

def pickle_dump(item, out_file):
    with open(out_file, "wb") as opened_file:
        pickle.dump(item, opened_file)


def pickle_load(in_file):
    with open(in_file, "rb") as opened_file:
        return pickle.load(opened_file)


def get_affine(in_file):
    return read_image(in_file).affine


def read_image_files(image_files, image_shape=None, crop=None, label_indices=None):
    """
    
    :param image_files: 
    :param image_shape: 
    :param crop: 
    :param use_nearest_for_last_file: If True, will use nearest neighbor interpolation for the last file. This is used
    because the last file may be the labels file. Using linear interpolation here would mess up the labels.
    :return: 
    """
    if label_indices is None:
        label_indices = []
    elif not isinstance(label_indices, collections.Iterable) or isinstance(label_indices, str):
        label_indices = [label_indices]
    image_list = list()
    for index, image_file in enumerate(image_files):
        if (label_indices is None and (index + 1) == len(image_files)) \
                or (label_indices is not None and index in label_indices):
            interpolation = "nearest"
        else:
            interpolation = "linear"
        image_list.append(read_image(image_file, image_shape=image_shape, crop=crop, interpolation=interpolation))

    return image_list


def read_image(in_file, image_shape=None, interpolation='linear', crop=None):
    print("Reading: {0}".format(in_file))
    image = nib.load(os.path.abspath(in_file))
    image = fix_shape(image)
    if crop:
        image = crop_img_to(image, crop, copy=True)
    if image_shape:
        return resize(image, new_shape=image_shape, interpolation=interpolation)
    else:
        return image


def fix_shape(image):
    if image.shape[-1] == 1:
        return image.__class__(dataobj=np.squeeze(image.get_data()), affine=image.affine)
    return image


def resize(image, new_shape, interpolation="linear"):
    image = reorder_img(image, resample=interpolation)
    zoom_level = np.divide(new_shape, image.shape)
    new_spacing = np.divide(image.header.get_zooms(), zoom_level)
    new_data = resample_to_spacing(image.get_data(), image.header.get_zooms(), new_spacing,
                                   interpolation=interpolation)
    new_affine = np.copy(image.affine)
    np.fill_diagonal(new_affine, new_spacing.tolist() + [1])
    new_affine[:3, 3] += calculate_origin_offset(new_spacing, image.header.get_zooms())
    return new_img_like(image, new_data, affine=new_affine)


def create_validation_split(problem_type,data, training_file, validation_file, train_split=0.8,testing_file=None, valid_test_split=0,overwrite=0):
    """
    Splits the data into the training and validation indices list.
    :param data_file: pytables hdf5 data file
    :param training_file:
    :param validation_file:
    :param data_split:
    :param overwrite:
    :return:
    """
    if overwrite or not os.path.exists(training_file):
        print("Creating validation split...")
        nb_samples = data.shape[0]
        print('Total Samples : ', nb_samples)
        sample_list = list(range(nb_samples))
        if problem_type is 'Classification':
            truth = data.read()
            classes = np.unique(truth).tolist()
            truth = truth.tolist()
            for i in classes:
                print("Number of samples for class ", i, " is : ", truth.count(i) ,'\n')
        
            x_train,x_valid,y_train,y_valid = train_test_split(sample_list,truth,stratify=truth,test_size=1-data_split)

            if valid_test_split > 0:
                x_valid,x_test,y_valid,y_test = train_test_split(x_valid,y_valid,stratify=y_valid,test_size=1-valid_test_split)
                pickle_dump(x_test,testing_file)
                print('Test Data Split:')
                for i in classes:
                    print("Number of samples for class ", i, " is : ", y_test.count(i) ,'\n')
        
            print('Train Data Split:')
            for i in classes:
                print("Number of samples for class ", i, " is : ", y_train.count(i) ,'\n')

            print('Valid Data Split:')
            for i in classes:
                print("Number of samples for class ", i, " is : ", y_valid.count(i) ,'\n')    

            pickle_dump(x_train, training_file)
            pickle_dump(x_valid, validation_file)
            return x_train, x_valid
        else:
            training_list, validation_list = split_list(sample_list, split=data_split)
            if valid_test_split > 0:
                validation_list,test_list = split_list(validation_list,split=valid_test_split)
                pickle_dump(test_list,testing_file)
            pickle_dump(training_list, training_file)
            pickle_dump(validation_list, validation_file)
            return training_list, validation_list
    else:
        print("Loading previous validation split...")
        return pickle_load(training_file), pickle_load(validation_file)



def split_list(input_list, split=0.8, shuffle_list=True):
    if shuffle_list:
        shuffle(input_list)
    n_training = int(len(input_list) * split)
    training = input_list[:n_training]
    testing = input_list[n_training:]
    return training, testing
