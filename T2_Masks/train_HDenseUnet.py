import os
import glob

from unet3d.data import write_data_to_file, open_data_file
from unet3d.generator import get_training_and_validation_generators
from unet3d.model.unet_multiGPU import unet_model_3d_multiGPU, get_multiGPUmodel
from unet3d.training import load_old_model, train_model
from keras.optimizers import SGD
from H_DenseUNet.hybridnet import dense_rnn_net
from H_DenseUNet.denseunet3d import denseunet_3d
from H_DenseUNet.loss import weighted_crossentropy
import argparse
from keras.utils import plot_model
import keras.backend as K
K.set_image_dim_ordering('tf')

config = dict()
config["pool_size"] = (2, 2, 2)  # pool size for the max pooling operations
config["image_shape"] = (256, 256, 40)  # This determines what shape the images will be cropped/resampled to.
config["patch_shape"] = (32, 32, 32)  # switch to None to train on the whole image None #
config["labels"] = (1,)  # the label numbers on the input image
config["n_labels"] = len(config["labels"])
config["n_base_filters"] = 16
config["all_modalities"] = ["T2"]
config["training_modalities"] = config["all_modalities"]  # change this if you want to only use some of the modalities
config["nb_channels"] = len(config["training_modalities"])
if "patch_shape" in config and config["patch_shape"] is not None:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["patch_shape"]))
else:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["image_shape"]))
config["truth_channel"] = config["nb_channels"]
config["deconvolution"] = True  # if False, will use upsampling instead of deconvolution

config["batch_size"] = 16
config["validation_batch_size"] = 16
config["n_epochs"] = 50  # cutoff the training after this many epochs
config["patience"] = 5  # learning rate will be reduced after this many epochs if the validation loss is not improving
config["early_stop"] = 15  # training will be stopped after this many epochs without the validation loss improving
config["initial_learning_rate"] = 0.0001
config["learning_rate_drop"] = 0.5  # factor by which the learning rate will be reduced
config["validation_split"] = 0.95  # portion of the data that will be used for training
config["flip"] = False  # augments the data by randomly flipping an axis during
config["permute"] = False  # data shape must be a cube. Augments the data by permuting in various directions
config["distort"] = None # switch to None if you want no distortion
config["augment"] = config["flip"] or config["distort"]
config["validation_patch_overlap"] = 0  # if > 0, during training, validation patches will be overlapping
config["training_patch_start_offset"] = (16, 16, 16)  # randomly offset the first patch index by up to this offset
config["skip_blank"] = True  # if True, then patches without any target will be skipped

config["data_file"] = os.path.abspath("T2_25pts_resize.h5")
config["model_file"] = os.path.abspath("liver_segmentation_model_HDenseUnet.h5")
config["training_file"] = os.path.abspath("training_resize_ids.pkl")
config["validation_file"] = os.path.abspath("validation_resize_ids.pkl")
config["overwrite"] = False  # If True, will previous files. If False, will use previously written files.

config["GPU"] = 1
config['model_summaryfile'] = config["model_file"].split('.')[0] + '_summary.txt'


def fetch_training_data_files(return_subject_ids=False):
    training_data_files = list()
    subject_ids = list()
    for subject_dir in glob.glob(os.path.join(os.path.dirname(__file__), "data", "preprocessed", "*")):
        subject_ids.append(os.path.basename(subject_dir))
        subject_files = list()
        for modality in config["training_modalities"] + ["Label"]:
            subject_files.append(os.path.join(subject_dir, modality + ".nii"))
        training_data_files.append(tuple(subject_files))
    if return_subject_ids:
        return training_data_files, subject_ids
    else:
        return training_data_files


def main(overwrite=False):
    # convert input images into an hdf5 file
    if overwrite or not os.path.exists(config["data_file"]):
        training_files, subject_ids = fetch_training_data_files(return_subject_ids=True)

        write_data_to_file(training_files, config["data_file"], image_shape=config["image_shape"],
                           subject_ids=subject_ids)

    data_file_opened = open_data_file(config["data_file"])

    if not overwrite and os.path.exists(config["model_file"]):
        base_model = load_old_model(config["model_file"])
        model = get_multiGPUmodel(base_model=base_model,n_labels=config["n_labels"],GPU=config["GPU"])
    else:
        # instantiate new model HYbrid Dense-Unet model from HDense project
        parser = argparse.ArgumentParser(description='Keras DenseUnet Training')
        parser.add_argument('-b', type=int, default= 1 )#config["batch_size"])
        parser.add_argument('-input_size', type=int, default= config["patch_shape"][0]) # 224 ) 
        parser.add_argument('-input_cols', type=int, default= config["patch_shape"][2]) #  8)
        args = parser.parse_args()
        #print(args.b)
        #model = dense_rnn_net(args)
        base_model = denseunet_3d(args)
        sgd = SGD(lr=1e-3, momentum=0.9, nesterov=True)
        model = base_model
        base_model.compile(optimizer=sgd, loss=[weighted_crossentropy])

    # get training and testing generators

     # Save Model
    plot_model(base_model,to_file="liver_segmentation_HDenseUnet.png",show_shapes=True)

    # Open the file
    with open(config['model_summaryfile'],'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        base_model.summary(line_length=150,print_fn=lambda x: fh.write(x + '\n'))

    train_generator, validation_generator, n_train_steps, n_validation_steps = get_training_and_validation_generators(
        data_file_opened,
        batch_size=config["batch_size"]*config["GPU"],
        data_split=config["validation_split"],
        overwrite=overwrite,
        validation_keys_file=config["validation_file"],
        training_keys_file=config["training_file"],
        n_labels=config["n_labels"],
        labels=config["labels"],
        patch_shape=config["patch_shape"],
        validation_batch_size=config["validation_batch_size"]*config["GPU"],
        validation_patch_overlap=config["validation_patch_overlap"],
        training_patch_start_offset=config["training_patch_start_offset"],
        permute=config["permute"],
        augment=config["augment"],
        skip_blank=config["skip_blank"],
        augment_flip=config["flip"],
        augment_distortion_factor=config["distort"])
      
    print('INFO: Training Details','\n Batch Size : ',config["batch_size"]*config["GPU"]
                                  ,'\n Epoch Size : ',config["n_epochs"])

    # run training
    train_model(model=model,
                model_file=config["model_file"],
                training_generator=train_generator,
                validation_generator=validation_generator,
                steps_per_epoch=n_train_steps,
                validation_steps=n_validation_steps,
                initial_learning_rate=config["initial_learning_rate"],
                learning_rate_drop=config["learning_rate_drop"],
                learning_rate_patience=config["patience"],
                early_stopping_patience=config["early_stop"],
                n_epochs=config["n_epochs"],base_model=base_model)
    data_file_opened.close()


if __name__ == "__main__":
    main(overwrite=config["overwrite"])
