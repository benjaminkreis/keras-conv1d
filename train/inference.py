import os
import sys
import h5py
import numpy as np
from numpy.random import seed
from optparse import OptionParser
from pandas import read_csv, DataFrame
from sklearn.preprocessing import minmax_scale
from keras.layers.convolutional import Conv1D
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
from callbacks import all_callbacks




#https://confluence.slac.stanford.edu/display/PSDM/How+to+access+HDF5+data+from+Python

def print_hdf5_file_structure(file_name) :
    """Prints the HDF5 file structure"""
    file = h5py.File(file_name, 'r') # open read-only
    item = file #["/Configure:0000/Run:0000"]
    print_hdf5_item_structure(item)
    file.close()

 
def print_hdf5_item_structure(g, offset='    ') :
    """Prints the input file/group/dataset (g) name and begin iterations on its content"""
    if   isinstance(g,h5py.File) :
        print g.file, '(File)', g.name
 
    elif isinstance(g,h5py.Dataset) :
        print '(Dataset)', g.name, '    len =', g.shape #, g.dtype
 
    elif isinstance(g,h5py.Group) :
        print '(Group)', g.name
 
    else :
        print 'WORNING: UNKNOWN ITEM IN HDF5 FILE', g.name
        sys.exit ( "EXECUTION IS TERMINATED" )
 
    if isinstance(g, h5py.File) or isinstance(g, h5py.Group) :
        for key,val in dict(g).iteritems() :
            subg = val
            print offset, key, #,"   ", subg.name #, val, subg.len(), type(subg),
            print_hdf5_item_structure(subg, offset + '    ')
 


#path = "/home/ec2-user/ben_higgstagging/train/train_deep_simple_all/"
path = "train_simple/"
h5File = h5py.File(path+'/KERAS_check_model_last_weights.h5')


# Print h5 contents
#for item in h5File.attrs.keys():
#    print(item + ":", h5File.attrs[item])


print_hdf5_file_structure(path+'/KERAS_check_model_last_weights.h5')
    
