import os
import glob
import cmd
import sys
from argparse import ArgumentParser
import pprint

from .pyimagesearch.nn.conv import *
import numpy as np
import pandas as pd
import tables
from random import shuffle
from .unet3d.normalize import normalize_data_storage,normalize_clinical_storage
from .unet3d.utils.utils import pickle_dump, pickle_load, create_validation_split
from sklearn import preprocessing
from sklearn.metrics import classification_report
import keras
from keras.utils import np_utils
from keras.utils import plot_model
from keras.optimizers import SGD,Adam
from .pyimagesearch.callbacks import TrainingMonitor
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
from keras.utils.training_utils import multi_gpu_model
from alt_model_checkpoint import AltModelCheckpoint
from imutils import paths
import matplotlib.pyplot as plt
import imutils
import cv2
import os
from .pyimagesearch.utils.generator_utils_Elan import *
from keras import Model
from .pyimagesearch.callbacks.datagenerator import *
from keras.layers import concatenate
from keras.layers.core import Dense
from sklearn.model_selection import train_test_split
from src.unet3d.metrics import weighted_dice_coefficient_loss

# Tensorboard specific imports
from time import time
from tensorflow.python.keras.callbacks import TensorBoard

config = dict()


def parse_command_line_arguments():
    print('argv type: ', type(sys.argv))
    print('argv: ', sys.argv)
    parser = ArgumentParser(fromfile_prefix_chars='@')
    req_group = parser.add_argument_group(title='Required flags')
    req_group.add_argument(
        '--problem_type',
        required=True,
        help='Segmentation, Classification, or Regression')
    req_group.add_argument(
        '--input_type',
        required=True,
        help='Image,Clinical,Both')
    req_group.add_argument(
        '--data_file',
        required=True,
        help='Name of the h5 data file')
    req_group.add_argument(
        '--batch_size',
        required=True,
        type=int,
        help='Batch size for training')
    req_group.add_argument(
        '--n_epochs',
        required=True,
        type=int,
        help='Maximum no. of epochs for training')
    req_group.add_argument(
        '--training_model',
        required=True,
        help='Name of the training model')

    parser.add_argument(
        '--data_dir',
        default='Data',
        help='Provide folder containing data files')
    parser.add_argument(
        '--training_split',
        default='training.pkl',
        help='Provide pickle file containing training indices')
    parser.add_argument(
        '--validation_split',
        default='validation.pkl',
        help='Provide pickle file containing validation indices')
    parser.add_argument(
        '--GPU',
        default=1,
        type=int,
        help='Number of GPUS for training, default=1')
    parser.add_argument(
        '--CPU',
        default=4,
        type=int,
        help='Number of CPU cores to use, default=4')
    parser.add_argument(
        '--n_classes',
        default=2,
        type=int,
        help='Enter the number of classes for classification problems')
    parser.add_argument(
        '--patch_shape',
        default=None,
        help='Enter patch shape for patch wise training, set to None for whole image training')
    parser.add_argument(
        '--skip_blank',
        default=1,
        type=int,
        help='Set to 1 if blank patches should be skipped, else set to 0')
    parser.add_argument(
        '--patch_overlap',
        default=0,
        type=int,
        help='Set to 0 for no patch overlap, else set to desired overlap size')
    parser.add_argument(
        '--validation_batch_size',
        default=None,
        help='If None, training batch size will be used, else provide validation batch size')
    parser.add_argument(
        '--flip',
        default=False,
        help='If False, no flipping will be used, if True, input images will be randomly flipped')
    parser.add_argument(
        '--permute',
        default=False,
        help='If False, no permutation will be used, if True, input images will be randomly permutted')
    parser.add_argument(
        '--distort',
        default=0,
        type=int,
        help='If 0, no distortion will be added')
    parser.add_argument(
        '--reduce',
        default=0,
        type=int,
        help='Number of slices in the z-axis to be truncated for 3D input images, truncation is applied symmetrically')
    parser.add_argument(
        '--CL_features',
        default=10,
        type=int,
        help='Number of non-imaging features to be used, default is 10')
    parser.add_argument('--lr',help='Initial Learning Rate, default is 1e-3')
    parser.add_argument('--metrics', help='Metric to monitor during training, default is val_acc')
    parser.add_argument('--loss',help='Loss function default is binary_crossentropy')
    parser.add_argument('--opt', default='sgd',help='Optimizer, default = sgd')
    parser.add_argument('--learning_rate_epochs',help='Epochs to drop lr')
    parser.add_argument('--image_shape', default='256,256')

    return parser.parse_args()


def build_config_dict(config):
   # config["labels"] = tuple(config['labels'].split(','))  # the label numbers on the input image
   # config["n_labels"] = len(config["labels"])


    # calculated values from cmdline_args
    #config["n_channels"] = len(config["training_modalities"])

    config["image_shape"] = map(int, (config['image_shape'].split(',')))
    config["image_shape"] = tuple(list(config["image_shape"]))
    
    # Save absolute path for input folders
    
    return config


def step_decay(epoch, initial_lrate, drop, epochs_drop):
    return initial_lrate * math.pow(drop, math.floor((1+epoch)/float(epochs_drop)))


def main(*arg):
    if arg:
        sys.argv.append(arg[0])
    args = parse_command_line_arguments()
    config = build_config_dict(vars(args))
    pprint.pprint(config)
    run_training(config)


def run_training(config):
# Step 1: Check if input type is defined
    try:
        input_type = config["input_type"]
    except:
        raise Exception("Error: Input type not defined | \t Set  config[\"input_type\"] to \"Image\", \"Clinical\" or \"Both\" \n")

# Step 2: Check if problem type is defined   
    try:
        problem_type = config["problem_type"]
    except:
        raise Exception("Error: Problem type not defined | \t Set  config[\"problem_type\"] to \"Classification\", \"Segmentation\" or \"Regression\" \n")
    
             
# Step 3: Check if the Data File is defined and open it
    try:
        data_file = tables.open_file(os.path.abspath(os.path.join(config["data_dir"],config["data_file"])),mode='r')
    except:
        raise Exception("Error: Could not open data file, check if config[\"data_file\"] is defined \n")

# Step 4:Check if datafile contains all the data arrays required for the problem type
# and load the pickle files containing training, validation split. If no pickle file is presnet, a 80/20 split of all the data in the datafile will be used for training and validation.
    training_file = os.path.abspath(os.path.join(config["data_dir"],config['training_split']))
    validation_file = os.path.abspath(os.path.join(config["data_dir"],config['validation_split']))
    if data_file.__contains__('/truth'): 
        if config["input_type"] is "Both" and data_file.__contains__('/cldata') and data_file.__contains__('/imdata'):
            training_list, validation_list =  create_validation_split(config["problem_type"], data_file.root.truth, training_file, validation_file,train_split=0.80,overwrite=0)
        elif config["input_type"] is "Image" and data_file.__contains__('/imdata'):
            training_list, validation_list =  create_validation_split(config["problem_type"], data_file.root.truth, training_file, validation_file,train_split=0.80,overwrite=0) 
        elif config["input_type"] is "Clinical" and data_file.__contains__('/cldata'):
            training_list, validation_list =  create_validation_split(config["problem_type"], data_file.root.truth, training_file, validation_file,train_split=0.80,overwrite=0) 
        else:
            print('Input Type: ', input_type)
            print('Clincial data: ', data_file.__contains__('/cldata'))
            print('Image data: ', data_file.__contains__('/imdata'))
            raise Exception("data file does not contain the input group required to train")
    else:
        print('Truth data: ', data_file.__contains__('/truth'))
        raise Exception("data file does not contain the truth group required to train")
    
    
# Step 5: Define Data Generators and Models for Specific Problem Types and Input Types:
    Ngpus = config['GPU']
    Ncpus = config['CPU']
    batch_size = config['batch_size']*Ngpus
    
    n_epochs = config['n_epochs']
    num_validation_steps = None
    num_training_steps = None
    model1 = None
    classWeight = None

    if problem_type is 'Classification':
        classes = np.unique(data_file.root.truth)
        print(classes)
        classes = [y.decode("utf-8") for y in classes]
            # Calculate class_weights for balanced training among classes
        Y = data_file.root.truth.read()
        Y =  np.asarray([y.decode("utf-8") for y in Y])
        #Convert to Binary categories
        le = preprocessing.LabelEncoder()
        Y = np_utils.to_categorical(le.fit_transform(Y), config['n_classes'])
        classTotals = Y.sum(axis=0)
        classWeight = classTotals.max() / classTotals
        print('classWeight: ', classWeight)

        if input_type is "Both":
            num_validation_patches,all_patches,validation_list_valid = get_number_of_patches(data_file, validation_list, patch_shape = config["patch_shape"],skip_blank=config["skip_blank"],patch_overlap=config["validation_patch_overlap"])
            num_training_patches,all_patches,training_list_valid =     get_number_of_patches(data_file, training_list, patch_shape = config["patch_shape"],skip_blank=config["skip_blank"],patch_overlap=config["validation_patch_overlap"])
            num_validation_steps = get_number_of_steps(num_validation_patches,config["validation_batch_size"])
            num_training_steps =  get_number_of_steps(num_training_patches, batch_size)
            training_generator = DataGenerator_3DCL_Classification(data_file, training_list_valid,
                                        batch_size=config['batch_size'],
                                        n_classes=config['n_classes'],
                                        classes = classes,
                                        augment=config['augment'],
                                        augment_flip=config['flip'],
                                        augment_distortion_factor=config['distort'],
                                        skip_blank=False,
                                        permute=config['permute'],reduce=config['reduce'])
            validation_generator = DataGenerator_3DCL_Classification(data_file, validation_list_valid,
                                        batch_size=config['validation_batch_size'],
                                        n_classes=config['n_classes'],
                                        classes = classes,
                                        augment=config['augment'],
                                        augment_flip=config['flip'],
                                        augment_distortion_factor=config['distort'],
                                        skip_blank=False,
                                        permute=config['permute'],reduce=config['reduce'])
        elif input_type is "Image":
            num_validation_patches,all_patches,validation_list_valid = get_number_of_patches(data_file, validation_list, patch_shape = config["patch_shape"],skip_blank=config["skip_blank"],patch_overlap=config["validation_patch_overlap"])
            num_training_patches,all_patches,training_list_valid =     get_number_of_patches(data_file, training_list, patch_shape = config["patch_shape"],skip_blank=config["skip_blank"],patch_overlap=config["validation_patch_overlap"])
            num_validation_steps = get_number_of_steps(num_validation_patches,config["validation_batch_size"])
            num_training_steps =  get_number_of_steps(num_training_patches, batch_size)
            
            training_generator = DataGenerator_3D_Classification(data_file, training_list_valid,
                                    batch_size=config['batch_size'],
                                        n_classes=config['n_classes'],
                                        classes = classes,
                                        augment=config['augment'],
                                        augment_flip=config['flip'],
                                        augment_distortion_factor=config['distort'],
                                        skip_blank=False,
                                        permute=config['permute'],reduce=config['reduce'])
            validation_generator = DataGenerator_3D_Classification(data_file, validation_list_valid,
                                        batch_size=config['validation_batch_size'],
                                        n_classes=config['n_classes'],
                                        classes = classes,
                                        augment=config['augment'],
                                        augment_flip=config['flip'],
                                        augment_distortion_factor=config['distort'],
                                        skip_blank=False,
                                        permute=config['permute'],reduce=config['reduce'])
        elif input_type is "Clinical":
            validation_list_valid =  validation_list
            num_validation_patches = len(validation_list)
            training_list_valid =  training_list
            num_training_patches = len(training_list_valid)
            num_validation_steps = get_number_of_steps(num_validation_patches,config["validation_batch_size"])
            num_training_steps =  get_number_of_steps(num_training_patches, batch_size)
                
            training_generator = DataGenerator_CL_Classification(data_file, training_list_valid,
                                        batch_size=config['batch_size'],
                                        n_classes=config['n_classes'],
                                        classes = classes)
            validation_generator = DataGenerator_CL_Classification(data_file, validation_list_valid,
                                        batch_size=config['validation_batch_size'],
                                        n_classes=config['n_classes'],
                                        classes = classes)
        
        if input_type is "Both":
            # create the MLP and CNN models
            mlp = MLP.build(dim=config['CL_features'],num_outputs=8,branch=True)
            cnn = Resnet3D.build_resnet_18(config['input_shape'],num_outputs=8,branch=True)
    
            # create the input to our final set of layers as the *output* of both
            # the MLP and CNN
            combinedInput = concatenate([mlp.output, cnn.output])
    
            # our final FC layer head will have two dense layers, the final one is the fused classification head
            x = Dense(8, activation="relu")(combinedInput)
            x = Dense(4, activation="relu")(x)
            x = Dense(2, activation="softmax")(x)
    
            # our final model will accept categorical/numerical data on the MLP
            # input and images on the CNN input, outputting a single value (the
            # predicted price of the house)
            model1 = Model(inputs=[mlp.input, cnn.input], outputs=x)
            plot_model(model1, to_file="Combined.png", show_shapes=True)
        elif input_type is "Image":
            # create the MLP and CNN models
            model1 = Resnet3D.build_resnet_18(config['input_shape'],num_outputs=2,reg_factor=1e-4,branch=False)
            plot_model(model1, to_file="Resnet_nolabel.png", show_shapes=True)
        elif input_type is "Clinical":
            # create the MLP and CNN models
            model1 = MLP.build(dim=config['CL_features'],num_outputs=2,branch=False)
            plot_model(model1, to_file="MLP.png", show_shapes=True)

    elif problem_type is 'Segmentation':
        if input_type is "Image":
            num_validation_patches,all_patches,validation_list_valid = get_number_of_patches(data_file, validation_list, patch_shape = config["patch_shape"],skip_blank=config["skip_blank"],patch_overlap=config["validation_patch_overlap"])
            num_training_patches,all_patches,training_list_valid =     get_number_of_patches(data_file, training_list, patch_shape = config["patch_shape"],skip_blank=config["skip_blank"],patch_overlap=config["validation_patch_overlap"])
            num_validation_steps = get_number_of_steps(num_validation_patches,config["validation_batch_size"])
            num_training_steps =  get_number_of_steps(num_training_patches, batch_size)
            
            training_generator = DataGenerator_3D_Segmentation(data_file, training_list_valid,
                                        batch_size=config['batch_size'],
                                        n_labels=config['n_labels'],
                                        labels = labels,
                                        augment=config['augment'],
                                        augment_flip=config['flip'],
                                        augment_distortion_factor=config['distort'],
                                        patch_shape = config['patch_shape'],
                                        patch_overlap = 0,
                                        patch_start_offset = 0,
                                        skip_blank=False,
                                        permute=config['permute'],reduce=config['reduce'])
            validation_generator = DataGenerator_3D_Segmentation(data_file, validation_list_valid,
                                        batch_size=config['validation_batch_size'],
                                        n_labels=config['n_labels'],
                                        labels = labels,
                                        augment=config['augment'],
                                        augment_flip=config['flip'],
                                        augment_distortion_factor=config['distort'],
                                        patch_shape = config['patch_shape'],
                                        patch_overlap = 0,
                                        patch_start_offset = 0,
                                        skip_blank=False,
                                        permute=config['permute'],reduce=config['reduce'])
            model1 = isensee2017_model.build()


# Step 6: Train model after compiling with problem specific parameters
    ## Make Model MultiGPU
    if Ngpus > 1:
        model = multi_gpu_model(model1, gpus=Ngpus)
       # model1.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])
    else:
        model = model1
    
    ## Tensorboard Paths for Monitoring
    figPath = os.path.sep.join([config["monitor"], "{}.png".format(os.getpid())])
    jsonPath = None
    tensorboard = TensorBoard(log_dir=config['monitor']+"\{}".format(time()))
    
    # OPTIMIZER
    if(config['opt'] == 'adam'):
        opt = Adam
    else:  
        opt = SGD(lr=1e-3, momentum=0.9) # Continuous Learning Rate Decay   

    # Loss Function
    if(config['loss'] == 'dsc'):
        loss_function = weighted_dice_coefficient_loss
    else:  
       loss_function = "binary_crossentropy"

    # Learning rate
    if config['lr']:
        learning_rate = config['lr']
    else:  
        learning_rate = 1e-3

    # Monitor Metrics
    if config['metrics']:
        metrics = config['lr']
    else:  
        metrics = "val_acc" #["accuracy"]    


    ## General Callbacks for all problems
    earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0005, patience=30, verbose=0, mode='auto')
    checkpoint = AltModelCheckpoint(config["training_model"]+'_model.h5', model1, monitor="val_acc",save_best_only=True, verbose=1)
    logger = CSVLogger(config["training_model"]+'_log.txt', append=True)
    if config["learning_rate_epochs"]:
        lr_scheduler = LearningRateScheduler(partial(step_decay, initial_lrate=learning_rate,
                                                        drop=0.5, epochs_drop=None))
    else:
        callbacks.append(ReduceLROnPlateau(factor=0.5, patience=30,verbose=1))
    
    callbacks = [lr_scheduler,tensorboard,checkpoint,earlystop]

    


    model.compile(optimizer=optimizer(lr=initial_learning_rate), loss=loss_function, metrics=metrics)
    ##
    # define the set of callbacks to be passed to the model during training
    #callbacks = [TrainingMonitor(figPath,jsonPath=jsonPath)]
    
    with open(config['training_model'] + '_summary.txt','w') as fh:
    # Pass the file handle in as a lambda function to make it callable
        model1.summary(line_length=150,print_fn=lambda x: fh.write(x + '\n'))

    # train the network
    print("[INFO] training network...")
    #aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,height_shift_range=0.1, shear_range=0.1, zoom_range=0.2,horizontal_flip=True, fill_mode="nearest")
    #H = model.fit(trainX, trainY, validation_data=(testX, testY), class_weight=classWeight, batch_size=Nbatches*Ngpus, epochs=Nepochs, callbacks=callbacks, verbose=1)
    H = model.fit_generator(generator=training_generator,
                        steps_per_epoch=num_training_steps,
                        epochs=n_epochs,
                        validation_data=validation_generator,
                        validation_steps=num_validation_steps,
                        callbacks=callbacks,
                        class_weight = classWeight,
                        use_multiprocessing=False, workers=Ncpus)
  
# Step 7: plot the training + testing loss and accuracy
    Fepochs = len(H.history['loss'])
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, Fepochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, Fepochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, Fepochs), H.history["acc"], label="acc")
    plt.plot(np.arange(0, Fepochs), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    figpath_final = config["input_type"]+'.png'
    plt.savefig(figpath_final)
    plt.show()
    hdf5_file.close()


if __name__ == "__main__":
    main()
