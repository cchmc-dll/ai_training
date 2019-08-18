import os
import glob
import cmd
import sys
from argparse import ArgumentParser
from .pyimagesearch.io.nifti_loader import nifti_loader
from .pyimagesearch.io.TIF_loader import TIF_loader, MiddleTIFLoader
import numpy as np
import pandas as pd
import tables
import pprint
from .unet3d.normalize import normalize_data_storage,normalize_clinical_storage,normalize_data_storage_2D
from .unet3d.generator import get_validation_split

config = dict()

def parse_command_line_arguments():
    print('argv type: ', type(sys.argv))
    print('argv: ',sys.argv)
    parser = ArgumentParser(fromfile_prefix_chars='@')

    req_group = parser.add_argument_group(title='Required flags')
    req_group.add_argument('--training_model_name', required=True, help='Filename of trained model to be saved')
    req_group.add_argument('--data_file', required=True, help='Source of images to train with')
    req_group.add_argument('--training_split', required=True, help='.pkl file with the training data split')
    req_group.add_argument('--validation_split', required=True, help='.pkl file with the validation data split')
    req_group.add_argument('--data_split', required=True, type=float, default=0.8)
    req_group.add_argument('--image_masks', required=True, help='Comma separated list of mask names, ex: Muscle,Bone,Liver')
    req_group.add_argument('--problem_type', required=True, help='Segmentation, Classification, or Regression, default=Segmentation')
    req_group.add_argument('-o,', '--output_dir', required=True, help='Path to directory where output files will be saved')

    parser.add_argument('--CPU', default=4, type=int, help='Number of CPU cores to use, default=4')
    parser.add_argument('--patch_shape', default=None)
    parser.add_argument('--skip_blank', action='store_true')
    parser.add_argument('--input_type', default='Image')
    parser.add_argument('--image_shape', default=(256, 256))
    parser.add_argument('--overwrite', default=1, type=int, help='0=false, 1=true')

    parser.add_argument('--labels', default='1', help='Comma separated list of the label numbers on the input image')
    parser.add_argument('--all_modalities', default='CT', help='Comma separated list of desired image modalities')
    parser.add_argument('--training_modalities', help='Comma separated list of desired image modalities for training only')

    return parser.parse_args()

def build_config_dict(config):
    config["labels"] = tuple(config['labels'].split(','))  # the label numbers on the input image
    config["n_labels"] = len(config["labels"])

    config['all_modalities'] = config['all_modalities'].split(',')

    try:
        config["training_modalities"] = config['training_modalities'].split(',')  # change this if you want to only use some of the modalities
    except AttributeError:
        config["training_modalities"] = config['all_modalities']

    # calculated values from cmdline_args
    config["n_channels"] = len(config["training_modalities"])
    config["input_shape"] = tuple([config["n_channels"]] + list(config["image_shape"]))
    config['image_masks'] = config['image_masks'].split(',')
    config['training_model'] = os.path.join(config['output_dir'], config['training_model_name'])

    return config



def get_image_loader(problem_type,input_images):
    if config["image_format"] is "TIF":
        if config['use_middle_image']:
            return MiddleTIFLoader(
                problem_type,
                input_images,
                config["input_shape"],
                config["image_modalities"],
                config["image_masks"],
                config['slice_number']
            )
        else:
            return TIF_loader(
                problem_type,
                input_images,
                config["input_shape"],
                config["image_modalities"],
                config["image_masks"],
                config['slice_number']
            )
    elif config["image_format"] is "NIFTI":
        return nifti_loader(
            problem_type,
            input_images,
            config["input_shape"],
            config["image_modalities"],
            config["image_masks"]
        )
    else:
        raise RuntimeError(f'Unsupported image format: {config["image_format"]}')


def main(*arg):
    if arg:
        sys.argv.append(arg[0])
    args = parse_command_line_arguments()
    config = build_config_dict(vars(args))
    pprint.pprint(config)
    run_preprocess(config)


def run_preprocess(overwrite=False):
    # Step 1: Check if Input Folders are defined
    try:
        input_type = config["input_type"]
    except:
        print("Error: Input type for preprocessing not defined | \t Set  config[\"input_type\"] to \"Image\", \"Clinical\" or \"Both\" \n")
    
    input_images = None
    input_clinical = None
    if (input_type == "Image" or input_type == "Both"):
        try:
            input_images = os.path.abspath(config["input_images"])
        except:
            print("Error: Input Image Folder for preprocessing not defined | \t Set config[\"input_images\"] \n")

    if (input_type == "Clinical" or input_type == "Both"):
        try:
            input_clinical = os.path.abspath(config["input_clinical"])
        except:
            print("Error: Input Clinical Folder with .csv for preprocessing not defined | \t Set config[\"input_clinical\"] \n")

   
    # Step 2: Check if the Output File is defined
    try:
        output_file = os.path.abspath(os.path.join("datasets", config["output_file"]))
    except:
        print("Error: Input type for preprocessing not defined | \t Set  config[\"input_type\"] to \"Image\",\"Clinical\" or \"Both\" \n")
       
    # Step 3: Check if Output file already exists, If it exists, require user permission to overwrite
    if 'overwrite' in config:
        overwrite = config["overwrite"]
    elif os.path.exists(output_file):
        overwrite = input("Output file exists, do you want to overwrite? (y/n) \n")
        overwrite = True if overwrite == 'y' else False   
    # Open the hdf5 file
    hdf5_file = tables.open_file(output_file, mode='w')
      
    # Step 4: Check problem specific parameters are defined
    problem_type = config['problem_type']
    if (input_type=="Both"):
        # Step 6: Load Imaging Data to hdf5 after checking if samples have both image and clinical data. If any 1 is missing, those samples are neglected.
        image_loader = get_image_loader(problem_type,input_images)
        subject_ids = image_loader.get_sample_ids()
        image_storage = None
        df_features =  pd.read_csv(os.path.join(input_clinical,'Features.csv'))
        df_features.set_index('Key',inplace=True)
        # If Both, select only samples that have both clinical and imaging data
        features = list(df_features)
        feature_array = []
        subject_ids_final = []
        for i,subject in enumerate(subject_ids):
            if subject in df_features.index:
                feature_array.append(df_features.loc[subject,features])
                subject_ids_final.append(subject)  
        image_loader.set_sample_ids(subject_ids_final)
        image_storage = image_loader.load_toHDF5(hdf5_file)

        # Load Clinical data to hdf5
        feature_array = np.asarray(feature_array)
        clinical_storage = hdf5_file.create_array(hdf5_file.root, 'cldata',  obj=feature_array)
        id_storage = hdf5_file.create_array(hdf5_file.root, 'subject_ids', obj=subject_ids_final)
        print("Input Data Preprocessed and Loaded to HDF5")

        # Step 7: Normalize Data Storage
        if config["normalize"]:
                normalize_data_storage(image_storage)
                normalize_clinical_storage(clinical_storage)
                print("Data in HDF5 File is normalized for training")     
         
    elif (input_type=="Image"):
        # Step 6: Load Imaging Data
        image_loader = get_image_loader(problem_type,input_images)
        image_storage = image_loader.load_toHDF5(hdf5_file=hdf5_file)
        subject_ids_final = image_loader.get_sample_ids()
        id_storage = hdf5_file.create_array(hdf5_file.root, 'subject_ids', obj=subject_ids_final)
        print("Input Data Preprocessed and Loaded to HDF5")

        # Step 7: Normalize Data Storage
        if config["normalize"]:
            if len(config["input_shape"]) > 2:
                normalize_data_storage(image_storage)
                print("Data in HDF5 File is normalized for training")    
            else:
                normalize_data_storage_2D(image_storage)


    # Step 6: Load Clinical data
    elif (input_type=="Clinical"):
        df_features =  pd.read_csv(os.path.join(input_clinical,'Features.csv'))
        df_features.set_index('Key',inplace=True)
        # If Both, select only samples that have both clinical and imaging data
        features = list(df_features)
        feature_array = []
        subject_ids = []
        
        subject_ids = df_features.index
        feature_array = df_features[features]
        
        df_truth = pd.read_csv(os.path.join(input_clinical,'Truth.csv'))
        df_truth.set_index('Key',inplace=True)
        truth = df_truth.loc[subject_ids,config["clinical_truthname"]]
        truth = truth.tolist()
        subject_ids = subject_ids.tolist()
        feature_array = np.array(feature_array)
        truth = np.asarray(truth)

        clinical_storage = hdf5_file.create_array(hdf5_file.root, 'cldata',  obj=feature_array)
        truth_storage = hdf5_file.create_array(hdf5_file.root, 'truth',obj=truth)
        id_storage = hdf5_file.create_array(hdf5_file.root, 'subject_ids', obj=subject_ids)
       
         # Step 7: Normalize Data Storage
        if config["normalize"]:
                normalize_clinical_storage(clinical_storage)
                print("Data in HDF5 File is normalized for training")
   
    hdf5_file.close()
    
   
if __name__ == "__main__":
    main()