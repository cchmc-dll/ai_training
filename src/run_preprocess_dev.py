from argparse import ArgumentParser
import os
import glob
import cmd
import pprint
import sys

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


def main(*arg):
    if arg:
        sys.argv.append(arg[0])
    args = parse_command_line_arguments()
    config = build_config_dict(vars(args))
    pprint.pprint(config)
    #run_preprocess(config)


## DEBUGGING: IF you want to overwrite options in your configuration loaded from config file.
config["input_type"] = "Image"
config["input_shape"] = (256,256)
config["input_images"] = "datasets/ImageDataCombined"
config["image_format"] = "TIF" # or "NIFTI"
config["slice_number"] = 0 # Use this if you have a stacked TIF and want only one slice for 2D problems.
                           # slice number goes from 0 to length of Stack
config['use_middle_image'] = True

config["output_file"] = "combined_aug3_205_fixed.h5"

config["overwrite"] = 1
config["problem_type"] = "Segmentation"
config["image_modalities"] = ["CT"]
config["image_masks"] = ["Muscle" ] #["Label"]   # For Image Masks, will serve as label for segmentation problems
config["n_channels"] = 0            # All image channels that will be used as input, image_mask can be input for classification problems and output for segmentation problems.

config["clinical_truthname"] =  None # For CSV File
config["normalize"] = True


if __name__ == "__main__":
    main()