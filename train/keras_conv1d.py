import os
import sys
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

my_filters = 8
my_kernel_size = 3
my_strides = 1
my_padding = 'same'
n_channels = 9

def get_data():
    
    print "Preparing data..."
        
    # Get y data
    Y = read_csv("../data/uci_har_dataset/train/y_train.txt", header=None)
    # -- One hot encode
    y_train_preOHE = Y.values
    y_train_preOHE = y_train_preOHE-1 #just need to offset the integers by 1 to start at 0
    y_train = to_categorical(y_train_preOHE)

    # Get x data
    x_path = "../data/uci_har_dataset/train/inertial_signals/"
    channel_files = os.listdir(x_path)
    n_channels = len(channel_files)

    X = np.zeros((len(y_train), 16, n_channels)) #empty array to fill
    print X.shape
    
    i_channel=0
    for my_channel in channel_files:
        my_channel_data = read_csv(x_path+"/"+my_channel, header=None, delim_whitespace=True)
        print my_channel_data.as_matrix().shape
        X[:,:,i_channel]=my_channel_data.as_matrix()[:,::8]
        i_channel += 1
    x_train = X

    print "x shape: {}".format(x_train.shape)
    print "y shape: {}".format(y_train.shape)
    #print "n categories = ",y_train.shape[1]
    
    from sklearn.utils import shuffle

    from sklearn.preprocessing import StandardScaler
    #x_train = StandardScaler().fit_transform(x_train)
    x_train, y_train = shuffle(x_train, y_train, random_state=123421412)
    return (x_train, y_train)


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option('-o','--output'   ,action='store',type='string',dest='outputDir'   ,default='train_simple/', help='output directory')
    (options,args) = parser.parse_args()
    
    if os.path.isdir(options.outputDir):
        print "output dir exists -- continuing anyway"
        #raise Exception('output directory must not exists yet')
    else:
        os.mkdir(options.outputDir)


    ##############
    ## Get data
    #############
    x_train, y_train = get_data()

    
    ###################
    ## Set up network
    ###################

    seed(123421412)
    model = Sequential()
    

    model.add(Conv1D(my_filters, my_kernel_size, input_shape = x_train.shape[1:3], padding=my_padding, strides = my_strides, activation = 'relu', kernel_initializer='glorot_uniform'))
    model.add(Flatten())
    #model.add(Dense(32, activation = 'relu', kernel_initializer='lecun_uniform'))
    model.add(Dense(y_train.shape[1], activation = 'softmax', kernel_initializer='lecun_uniform'))

    print model.summary()
    
    outfile = open(options.outputDir + '/' + 'KERAS_model.json','wb')
    jsonString = model.to_json()
    import json
    with outfile:
        obj = json.loads(jsonString)
        json.dump(obj, outfile, sort_keys=True,indent=4, separators=(',', ': '))
        outfile.write('\n')


    ###########
    ## Train
    ###########

    startlearningrate=0.001
    adam = Adam(lr=startlearningrate)
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])

    callbacks=all_callbacks(stop_patience=1000,
                            lr_factor=0.5,
                            lr_patience=10,
                            lr_epsilon=0.000001,
                            lr_cooldown=2,
                            lr_minimum=0.0000001,
                            outputDir=options.outputDir)
    model.fit(x_train, y_train, batch_size = 128, epochs = 100, validation_split = 0.25, verbose = 0, callbacks = callbacks.callbacks)


    ###############
    ## Evaluate
    ###############

    # Just using training data for now

    print "Metrics: ", model.metrics_names
    print "Evaluate: ", model.evaluate(x_train, y_train, batch_size=128)

    # Predictions
    print(y_train[0:100:10])
    print(model.predict(x_train[0:100:10]))
    print "\n"
    

    print(model.predict(x_train[0:1]))
