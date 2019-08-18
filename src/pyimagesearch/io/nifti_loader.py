# import the necessary packages
import os
import tables
import glob
from ...unet3d.normalize import normalize_data_storage, reslice_image_set
import numpy as np
from nilearn.image import new_img_like

class nifti_loader:
    def __init__(self,problem_type='Classification',input_images=None,input_shape=(128,128,32),image_modalities=['T2'],mask=None):
        self.input_images = input_images
        self.input_shape = input_shape
        self.problem_type = problem_type
        self.image_modalities = image_modalities
        self.mask = mask
        self.n_channels = len(self.image_modalities)+len(self.mask)
       
        if self.problem_type is 'Segmentation':
            training_data_files = list()
            subject_ids = list()
            for subject_dir in glob.glob(os.path.join(self.input_images, "*")):
                subject_ids.append(os.path.basename(subject_dir))
                subject_files = list()
                for modality in self.image_modalities + self.mask:
                    subject_files.append(os.path.join(subject_dir, modality + ".nii"))
                training_data_files.append(tuple(subject_files))
            self.data_files = training_data_files
            self.ids = subject_ids
            self.image_data_shape = tuple([0, len(self.image_modalities)] + list(self.input_shape))
            self.truth_data_shape = tuple([0, 1] + list(self.input_shape))
            
            
        elif problem_type is 'Classification':
            training_data_files = list()
            subject_ids = list()
            for classes in glob.glob(os.path.join(self.input_images, "*")):
                for subject_dir in glob.glob(os.path.join(classes, "*")):
                    subject_ids.append(os.path.basename(subject_dir))
                    subject_files = list()
                    for modality in self.image_modalities + self.mask:
                        subject_files.append(os.path.join(subject_dir, modality + ".nii"))
                    training_data_files.append(tuple(subject_files))
            self.data_files = training_data_files
            self.ids = subject_ids
            self.image_data_shape = tuple([0, len(self.image_modalities)+len(self.mask)] + list(self.input_shape))
            self.truth_data_shape = tuple([0,])

        #elif problem_type is 'Regression':
        #    training_data_files = list()
        #    subject_ids = list()
        #    for subject_dir in glob.glob(os.path.join(self.input_images, "*")):
        #        subject_ids.append(os.path.basename(subject_dir))
        #        subject_files = list()
        #        for modality in self.image_modalities:
        #            subject_files.append(os.path.join(subject_dir, modality + ".nii"))
        #        training_data_files.append(tuple(subject_files))
        #    self.data_files = training_data_files
        #    self.ids = subject_ids
    
    def get_sample_ids(self):
        return self.ids

    def set_sample_ids(self,new_ids):
        self.ids = new_ids

    def load_toHDF5(self,hdf5_file=None,verbose=-1):
		    # initialize the list of features and labels
            n_samples = len(self.ids)
            filters = tables.Filters(complevel=5, complib='blosc')
            image_storage = hdf5_file.create_earray(hdf5_file.root, 'imdata', tables.Float32Atom(), shape=self.image_data_shape, filters=filters, expectedrows=n_samples)
            affine_storage = hdf5_file.create_earray(hdf5_file.root, 'affine', tables.Float32Atom(), shape=(0, 4, 4), filters=filters, expectedrows=n_samples)
            
            if self.problem_type is "Classification":
                truth_storage =  hdf5_file.create_earray(hdf5_file.root, 'truth', tables.StringAtom(itemsize=15), shape=self.truth_data_shape, filters=filters, expectedrows=n_samples)
            elif self.problem_type is "Segmentation":
                truth_storage =  hdf5_file.create_earray(hdf5_file.root, 'truth', tables.UInt9Atom(), shape=self.truth_data_shape, filters=filters, expectedrows=n_samples)
           
            # loop over the input images
            for (i, imagePath) in enumerate(self.data_files):
			    # load the image and extract the class label assuming
			    # that our path has the following format:
			    # /path/to/dataset/{class}/{image}.jpg
                if self.problem_type is "Classification":
                    subject_name = imagePath[0].split(os.path.sep)[-2]
                    if subject_name in self.ids:
                        images = reslice_image_set(in_files=imagePath, image_shape=self.input_shape, label_indices=len(imagePath)-1, crop=True)
                        label = imagePath[0].split(os.path.sep)[-3]
                        subject_data = [image.get_data() for image in images]
                        affine = images[0].affine
                        image_storage.append(np.asarray(subject_data)[np.newaxis])
                        affine_storage.append(np.asarray(affine)[np.newaxis])
                        truth_storage.append(np.asarray(label)[np.newaxis])
               
                elif self.problem_type is "Segmentation":
                    images = reslice_image_set(in_files=imagePath, image_shape=self.input_shape, label_indices=len(imagePath)-1, crop=True)
                    subject_data = [image.get_data() for image in images]
                    affine = images[0].affine
                    image_storage.append(np.asarray(subject_data[:self.n_channels])[np.newaxis])
                    affine_storage.append(np.asarray(affine)[np.newaxis])
                    truth_storage.append(np.asarray(subject_data[self.n_channels],dtype=np.uint8)[np.newaxis][np.newaxis])
                
                    
                #elif self.problem_type is "Regression":
                #    image = cv2.imread(imagePath)
                    
			    # show an update every `verbose` images
                if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                    print("[INFO] processed {}/{}".format(i + 1,len(self.ids)))
            return(image_storage)  

    def load(self, verbose=-1):
		    # initialize the list of features and labels
            # imagePaths = self.data_files
            data = []
            labels = []
		    # loop over the input images
            for (i, imagePath) in enumerate(self.data_files):
			    # load the image and extract the class label assuming
			    # that our path has the following format:
			    # /path/to/dataset/{class}/{image}.jpg
                if self.problem_type is "Classification":
                    images = reslice_image_set(in_files=imagePath, image_shape=self.input_shape, label_indices=len(imagePath)-1, crop=True)
                    label = imagePath[0].split(os.path.sep)[-3] # treat our processed image as a "feature vector"
			                                                 # by updating the data list followed by the labels
                elif self.problem_type is "Segmentation":
                     images = reslice_image_set(in_files=imagePath, image_shape=self.input_shape, label_indices=len(imagePath)-1, crop=True)
                #elif self.problem_type is "Regression":
                #    image = cv2.imread(imagePath)
                data.append(images)
                labels.append(label)
			    # show an update every `verbose` images
                if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                    print("[INFO] processed {}/{}".format(i + 1,
					    len(self.data_files)))
		    # return a tuple of the data and labels
            return (self.ids,np.array(data), np.array(labels))

