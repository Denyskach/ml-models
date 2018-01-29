from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.models import Sequential
from keras.models import Model
from keras.optimizers import Adam

from keras.regularizers import l2

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