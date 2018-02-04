import bcolz

from keras.preprocessing import image

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder

def plot_learning_curve(path):
    training_df = pd.read_csv(path)
    
    plt.subplot(211) # #row, cols, figure num
    plt.plot(training_df["acc"].values, 'r', training_df["val_acc"].values, 'b')
    training_patch = mpatches.Patch(color='red', label='training')
    validation_patch = mpatches.Patch(color='blue', label='validation')

    plt.legend(handles=[training_patch, validation_patch])

    plt.ylabel('accuracy')
    plt.xlabel('# epoch')

    plt.subplot(212)
    plt.plot(training_df["loss"].values, 'r', training_df["val_loss"].values, 'b')
    plt.ylabel('loss')
    plt.xlabel('# epoch')


    plt.show()
    
def save_array(fname, arr): c=bcolz.carray(arr, rootdir=fname, mode='w'); c.flush()
def load_array(fname): return bcolz.open(fname)[:]

def get_batches(dirname, gen=image.ImageDataGenerator(), shuffle=True, batch_size=4, class_mode='categorical',
                target_size=(244, 244)):
    return gen.flow_from_directory(dirname, target_size=target_size, 
                                   class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)

def get_data(path, target_size=(244, 244)):
    batches = get_batches(path, shuffle=False, batch_size=1, class_mode=None, target_size=target_size)
    return np.concatenate([batches.next() for i in range(batches.samples)])

def get_breed(labels_df, id):
    return labels_df.loc[labels_df["id"] == id]["breed"].values[0]

def onehot(x): return np.array(OneHotEncoder().fit_transform(x.reshape(-1,1)).todense())