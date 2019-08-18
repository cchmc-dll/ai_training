import os

from T2_masks.train_2 import config
from unet3d.prediction_isense import run_validation_cases #, run_validation_cases_zsplit

config["validation_file"] =  os.path.abspath("NewTest_ids.pkl")
config["data_file"] = os.path.abspath("NewTest_data_151pts.h5")
config["model_file"] = os.path.abspath("liver_segmentation_3DUnet_isense.h5")

config["image_shape"] = (256,256,40)  # Input image shape
config["patch_shape"] = (256,256,32) # switch to None to train on the whole image None #
config["validation_patch_overlap"] = (0,0,28)
config["z_axis_split"] = False# Use this if your image shape is different from the image shape used for training.  
                              # Use only if whole image training on reduced number of Z_slices was done.
                              # IF training was done using patches, provide patch shape and set this parameter to False


def main():
    prediction_dir = os.path.abspath("prediction_NewTest_3DUnet_isense_40slices")

    if not config["z_axis_split"]:
        run_validation_cases(validation_keys_file=config["validation_file"],
                             model_file=config["model_file"],
                             training_modalities=config["training_modalities"],
                             labels=config["labels"],
                             hdf5_file=config["data_file"],
                             output_label_map=True,
                             output_dir=prediction_dir,
                             overlap=config["validation_patch_overlap"],
                             threshold=0.50)

    else:
        run_validation_cases_zsplit(validation_keys_file=config["validation_file"],
                             model_file=config["model_file"],
                             training_modalities=config["training_modalities"],
                             labels=config["labels"],
                             hdf5_file=config["data_file"],
                             output_label_map=True,
                             output_dir=prediction_dir,
                             overlap=config["validation_patch_overlap"],
                             threshold=0.50)


if __name__ == "__main__":
    main()
