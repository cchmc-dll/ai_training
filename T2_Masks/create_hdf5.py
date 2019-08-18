import os
import glob

from T2_masks.train_2 import config, fetch_training_data_files
from unet3d.data import write_data_to_file, open_data_file

config["data_file"] = 'T2_25pts_resize.h5'
config["all_modalities"] = ["T2"]
config["training_modalities"] = config["all_modalities"]  # change this if you want to only use some of the modalities
config["nb_channels"] = len(config["training_modalities"])
def main(overwrite=True):
    # convert input images into an hdf5 file
    if overwrite or not os.path.exists(config["data_file"]):
        training_files, subject_ids = fetch_training_data_files(return_subject_ids=True)
        write_data_to_file(training_files, config["data_file"], image_shape=config["image_shape"],
                           subject_ids=subject_ids)


if __name__ == "__main__":
    main()
