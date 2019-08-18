import os

from T2_masks.train_2 import config
from unet3d.prediction import run_validation_cases

config["validation_file"] =  os.path.abspath("NewTest_ids.pkl")
config["data_file"] = os.path.abspath("NewTest_data_151pts.h5")
config["model_file"] = os.path.abspath("liver_segmentation_model_581_resize_1GPU.h5")
def main():
    prediction_dir = os.path.abspath("prediction_NewTest_581Model")
    run_validation_cases(validation_keys_file=config["validation_file"],
                         model_file=config["model_file"],
                         training_modalities=config["training_modalities"],
                         labels=config["labels"],
                         hdf5_file=config["data_file"],
                         output_label_map=True,
                         output_dir=prediction_dir)


if __name__ == "__main__":
    main()
