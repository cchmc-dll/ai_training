
import os
import glob

from T2_masks.train_2 import config, fetch_training_data_files
from unet3d.data import write_data_to_file, open_data_file
from unet3d.generator import get_validation_split

config["training_file"] = os.path.abspath("T2_debug_Train_ids.pkl")
config["validation_file"] = os.path.abspath("T1_debug_Test_ids.pkl")
config["validation_split"] = 0.80
config["data_file"] = os.path.abspath("T2_25pts_resize.h5")
data_file_opened = open_data_file(config["data_file"])
def main(overwrite=True):
      training_list, validation_list = get_validation_split(data_file_opened,
                                                          data_split=config["validation_split"],
                                                          overwrite=overwrite,
                                                          training_file=config["training_file"] ,
                                                          validation_file=config["validation_file"])
      print('validation list is ', validation_list)
      print('Training list is ', training_list)


if __name__ == "__main__":
    main()
