import numpy as np
from skimage.color import rgb2gray
import tensorflow as tf
import os
from skimage.io import imread
from skimage.util import crop
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from utils import *


FAL=-1000
#(164, 306, 3) after crop
img_path= '../Synthetic_Leopard_Circle/'
FAL=-1000
box = ((196,120), (154, 180),(0,0))
commom_size=(100, 160, 3)
def getOffset(img1,img2):
    '''
    Get offset of two given images

    :return: The degree of offset within the range (-45,45).
    '''
    dg1=int(img1.split('_')[1].split('.')[0])*3
    dg2=int(img2.split('_')[1].split('.')[0])*3
    fs=dg2-dg1
    if fs >= -45 and fs <= 45:
        return fs
    elif 360-abs(fs) >= -45 and 360-abs(fs) <= 45:
        if fs >0:
            return 360-abs(fs)
        else :
            return -1*(360-abs(fs))
    return FAL

def loadData( path ):
    '''
    The function gets images according to given path and find out all possible images pairs in it and construct them
    as following form: x:(image1,image2) : y{label:(1,0)}. Meanwhile, every pixels in the images are divided by 255 in
    order to normalize

    :param Path: A str. The path of data
    :return: Two np.narray object. Images pairs and labels mention above with shape(n, 2, 64, 64) and (n, )
    '''
    print('Loading data....')
    imageList=os.listdir(path)
    x=[]
    y=[]
    for i in range(120):
        img1 = imread(path + imageList[i])
        for j in range(120):
            offset=getOffset(imageList[i], imageList[j])
            if offset == FAL:
                continue
            img2=imread(path + imageList[j])
            x.append([img1,img2])
            y.append(offset/3+15)
    x=np.array(x)
    y=np.array(y)
    return x,y


def prePrec(data_x, data_y,gray=False ,parallel=False):
    '''
    This function can do pre-processing to the given dataset according to specific requirement. And this function will split the original
    dataset into training set validation set, test set in a ratio of 90:10:20.

    :param data_x: Image pairs in shape(n,2,648,480)
    :param data_y: The ground truth
    :param gray: A boolean value indicates if the images need to be convert to gray image
    :param parallel: A boolean value indicates the dataset need to be applied in a parallel network or not.(for the final task)
    :return: Training set validation set and test set
    '''
    print('Pre-precessing....')
    x=[]
    for pairs in data_x:
        if gray == True:
            sup=np.ones((commom_size[0],commom_size[1],1),dtype=np.float32)
            x.append(np.concatenate((resize(   (crop(rgb2gray(pairs[0]), (box[0],box[1])).astype(np.float32)-255./2) / (255./2), (commom_size[0],commom_size[1],1)),
                                     resize(   (crop(rgb2gray(pairs[1]), (box[0],box[1])).astype(np.float32)-255./2) / (255./2), (commom_size[0],commom_size[1],1)),sup), axis=2))
        elif parallel == True:
            x.append([ resize((crop(pairs[0],box).astype(np.float32)-255./2) / (255./2),  commom_size),  resize((crop(pairs[1],box).astype(np.float32)-255./2) / (255./2), commom_size)])
        else:
            x.append(np.concatenate((resize(crop(pairs[0],box).astype(np.float32)/255.,commom_size) , resize(crop(pairs[1],box).astype(np.float32)/255., commom_size)),axis=2))
    x=np.array(x)


    x_train, x_test, y_train, y_test = train_test_split(x, data_y, test_size=2./12., shuffle=True)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=1./10., shuffle=True)

    if parallel == True:
        x_train=np.transpose(x_train, (1, 0, 2, 3, 4))
        x_val = np.transpose(x_val, (1, 0, 2, 3, 4))
        x_test = np.transpose(x_test, (1, 0, 2, 3, 4))
    return x_train,y_train,x_val,y_val,x_test,y_test


def train(model,train_x,train_y,valid_x,valid_y,parameters,continue_from=0):
    '''
    This function to run training process according to given model and data. In this case, the optimizer of training is Adam
    and use a binary crossentropy or mse as loss function.

    :param model: The model need to be trained
    :param train_x: Training image pairs with shape (n, 2, 64, 64)
    :param train_y: Training labels list
    :param valid_x: Validation dataset split beforehand
    :param valid_y: Validation labels
    :param parameters: A dictionary that record hyper-parameters of different training task.
    :param continue_from: An integer used for continue training
    :return: A trained model object
    '''

    print('Training....')
    if parameters['type'] == 'regression':
        train_y=(train_y-15)/15.
        valid_y=(valid_y-15)/15.


    adam = tf.keras.optimizers.Adam( lr=parameters['lr'],clipnorm=1)
    model.compile(optimizer=adam,
                  loss=parameters['loss'],

                  metrics=parameters['metrics'])
    history=None
    if os.path.exists(parameters['save_path'] + '.index'):
        print('-------------load the model-----------------')
        model.load_weights(parameters['save_path'])
    else:
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=parameters['save_path'],save_weights_only=True,save_best_only=True)

        if parameters['type'] == 'parallel':
            train_x = [train_x[0], train_x[1]]
            valid_x = [valid_x[0], valid_x[1]]
        history = model.fit(train_x, train_y, batch_size=parameters['batch_size'], initial_epoch=continue_from,
                            epochs=parameters['epoch'], validation_data=(valid_x, valid_y), validation_freq=1,
                            callbacks=[cp_callback], shuffle=True)
        acc = history.history[parameters['metrics'][0]]
        val_acc = history.history['val_' + parameters['metrics'][0]]
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        plt.subplot(1, 2, 1)
        plt.plot(acc, label='Training '+parameters['metrics'][0])
        plt.plot(val_acc, label='Validation '+parameters['metrics'][0])
        plt.title('Training and Validation '+parameters['metrics'][0])
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.show()
    return model,history
