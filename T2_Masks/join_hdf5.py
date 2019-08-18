import os
import glob

from T2_masks.train_2 import config, fetch_training_data_files
from unet3d.data import write_data_to_file, open_data_file
import tables as tb

# Join 2 hdf5 files
# input_file
config["data_file1"] = 'T2_data_200_resize.h5'
config["data_file2"] = 'Test_data_381pts_resize.h5'
# ouput_file
config["out_file"] = 'T2_581pts_resize.h5'

data_file1 = open_data_file(config["data_file1"]) 
data_file2 = open_data_file(config["data_file2"])
out_file  = tb.open_file(config["out_file"], 'w')

def main(overwrite=True):
    for node in data_file1:
        print(node)
    for node in data_file2:
        print(node)
    # Copy Subject_ids array first
    x = data_file1.root.subject_ids
    y = data_file2.root.subject_ids
    z = out_file.create_array('/', 'subject_ids', atom=x.atom, shape=(x.nrows + y.nrows,))
    z[:x.nrows] = x[:]
    z[x.nrows:] = y[:]
    print('After Copying Arrays in data_file 1')
    for node in out_file:
        print(node)
   # Copy E-arrays
    # File 1
    #x_data = data_file1.root.data_storage
    #x_truth = data_file1.root.truth_storage
    #x_affine = data_file1.root.affine_storage
    ## File 2
    #y_data = data_file2.root.data_storage
    #y_truth = data_file2.root.truth_storage
    #y_affine = data_file2.root.affine_storage
    ## Copy to source 1 to new file
    z_data = data_file1.copy_node('/',name='data',newparent=out_file.root,newname='data')
    z_truth = data_file1.copy_node('/',name='truth',newparent=out_file.root,newname='truth')
    z_affine = data_file1.copy_node('/',name='affine',newparent=out_file.root,newname='affine')
    print('After Copying Earrays in data_file 1')
    for node in out_file:
        print(node)
    # Append source 2 to the new file
    z_data.append(data_file2.root.data[:])
    z_truth.append(data_file2.root.truth[:])
    z_affine.append(data_file2.root.affine[:])
    print('After Copying Earrays in data_file 2')
    for node in out_file:
        print(node)
    print('HDF5 files merged')
    out_file.close()
    
    # convert input images into an hdf5 file
    #if overwrite or not os.path.exists(config["out_file"]):
    #    nchannel = len(data_file1[0])-1
    #    data_storage,truth_storage,affine_storage = create_data_file(out_file=config["out_file"],n_channels=nchannel,image_shape=



if __name__ == "__main__":
    main()
