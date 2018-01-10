import os
import sys
import h5py
import math
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
from keras_conv1d import get_data


#######################################
## Print a bias or weight array to C++
#######################################
def print_array_to_cpp(name, a, odir ):

    #put output in subdir for tarballing later
    f=open("{}/{}.h".format(odir,name),"w")

    #meta data
    f.write("//Numpy array shape {}\n".format(a.shape))
    f.write("//Min {}\n".format(np.min(a)))
    f.write("//Max {}\n".format(np.max(a)))
    f.write("\n")
    
    #c++ variable 
    if "w" in name: 
        f.write("weight_default_t {}".format(name))
    elif "b" in name: 
        f.write("bias_default_t {}".format(name))
    else:
        raise Exception('ERROR: Unkown weights type')


    for x in a.shape:
        f.write("[{}]".format(x))
    f.write(" = {")
    
    #fill c++ array.  
    #not including internal brackets for multidimensional case
    i=0;
    zero_ctr = 0;
    for x in np.nditer(a, order='C'):
        if x == 0: 
            zero_ctr += 1
        if i==0:
            f.write("{}".format(x))
        else:
            f.write(", {}".format(x))
        i=i+1
    f.write("};")
    f.close()

    return zero_ctr;



# The following two functions from
# https://confluence.slac.stanford.edu/display/PSDM/How+to+access+HDF5+data+from+Python

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
            

################
## Look at h5
################

#path = "/home/ec2-user/ben_higgstagging/train/train_deep_simple_all/"
path = "train_simple/"
h5File = h5py.File(path+'/KERAS_check_model_last_weights.h5')

# Print h5 contents
#for item in h5File.attrs.keys():
#    print(item + ":", h5File.attrs[item])

print_hdf5_file_structure(path+'/KERAS_check_model_last_weights.h5')
    

##################################
## Get weights, biases, and data
##################################

#Get kernels/filters/weights and biases
conv_k = h5File['/conv1d_1/conv1d_1/kernel:0'][()]
conv_b = h5File['/conv1d_1/conv1d_1/bias:0'][()]
dense_k = h5File['/dense_1/dense_1/kernel:0'][()]
dense_b = h5File['/dense_1/dense_1/bias:0'][()]

print_array_to_cpp("w1",conv_k,"weights")
print_array_to_cpp("b1",conv_b,"weights")

print_array_to_cpp("w2",dense_k,"weights")
print_array_to_cpp("b2",dense_b,"weights")

x_train, y_train = get_data()


##################
## Do inference 
##################

# This is probably written with more lines of code than necessary for future porting to HLS4ML


## Settings
padding = 'same' #same or valid
stride = 3
filter_width = 2
n_filters = 2
n_channels = 2


### Get a sample
x_sample = x_train[0,:,:]
y_sample = y_train[0,:]


print "x sample shape: ",x_sample.shape
print "y sample shape: ",y_sample.shape
print "x sample: ",x_sample
print "y sample: ",y_sample

print_array_to_cpp("w_x",x_sample,".")
print_array_to_cpp("w_y",y_sample,".")

### Padding
### Equations from tensorflow documentation

# Valid
in_width = x_sample.shape[0]
out_width  = int(math.ceil(float(in_width - filter_width + 1) / float(stride)))
pad_left = 0
pad_right = 0

# Same
#in_width = x_sample.shape[0]
#out_width  = int(math.ceil(float(in_width) / float(stride)))
#if (in_width % stride == 0):
#    pad_along_width = max(filter_width - stride, 0)
#else:
#    pad_along_width = max(filter_width - (in_width % stride), 0)
#pad_left = pad_along_width // 2
#pad_right = pad_along_width - pad_left
print "in_width: ",in_width
print "out_width: ",out_width
#print "pad_along_width: ",pad_along_width
print "pad_left: ",pad_left
print "pad_right: ", pad_right
x_sample = np.pad(x_sample, [(pad_left,pad_right),(0,0)], 'constant')

print "x sample shape: ",x_sample.shape
print "y sample shape: ",y_sample.shape
print "x sample: ",x_sample
print "y sample: ",y_sample
#for j in range(0,x_sample.shape[1]):
#    for i in range(0,x_sample.shape[0]):
#        print x_sample[i,j]
#    print "\n"
in_width = x_sample.shape[0]


##############
## Convolve 
##############

print "conv_k shape: ",conv_k.shape

conv_out = np.zeros((out_width,n_filters))

for i in range(0,out_width):
    for f in range(0,n_filters): 
        channel_sum = 0
        for c in range(0,n_channels):
            
            #Select data
            x_buffer = x_sample[:,c]
            x_buffer = x_buffer[i*stride:i*stride+filter_width]

            #Select filter
            my_filter = conv_k[:,c,f]
            
            if i==0 and f==0 and c==0:
                print "buffer shape: ",x_buffer.shape
                print "filter shape: ",my_filter.shape
                
            my_dot = np.dot(x_buffer,my_filter)
            channel_sum += my_dot
        conv_out[i,f] = channel_sum + conv_b[f]

print "conv_out: ",conv_out
print_array_to_cpp("w_conv_out",conv_out,".")

#for j in range(0,conv_out.shape[1]):
#    for i in range(0,conv_out.shape[0]):
#        print conv_out[i,j]
#    print "\n"

print "conv_out shape, pre-flatten: ",conv_out.shape
print "conv_out, pre-flatten: ",conv_out
conv_out = conv_out.flatten()
print "conv_out shape: ",conv_out.shape
print "conv_out: ",conv_out


#relu
conv_out = conv_out * (conv_out > 0)
print "conv_out, post relu: ",conv_out

##########
## Dense
##########
dnn_out = np.dot(conv_out, dense_k)+dense_b

print "dnn_out: ",dnn_out

#softmax
dnn_out = np.exp(dnn_out) / sum(np.exp(dnn_out))

print "Final output: ",dnn_out


