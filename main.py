import numpy as np
from skimage.color import rgb2gray
import tensorflow as tf
import os
from skimage.io import imread
from skimage.util import crop
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D,  MaxPool2D,  Flatten, Dense
from tensorflow.keras import Model
from sklearn.model_selection import train_test_split
from skimage.transform import resize
import argparse
from utils import *
from models import Net
img_path= '../Synthetic_Leopard_Circle/'
FAL=-1000
box = ((196,120), (154, 180),(0,0))
commom_size=(100, 160, 3)
parameter_set={
    'classification':{
        'type':'classification',
        'epoch':200,
        'batch_size':32,
        'lr':0.001,
        'loss':tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        'metrics':['sparse_categorical_accuracy'],
        'save_path':'./checkpoint/leopard_classification.ckpt'
    },
    'regression':{
        'type':'regression',
        'epoch': 50,
        'batch_size': 32,
        'lr': 0.0001,
        'loss': tf.keras.losses.mean_squared_error,
        'metrics': ['mse'],
        'save_path':'./checkpoint/leopard_regression.ckpt'
    },
    'mobilenet-v2_regression':{
        'type': 'regression',
        'epoch': 50,
        'batch_size': 16,
        'lr': 0.001,
        'loss': tf.keras.losses.mean_squared_error,
        'metrics': ['mse'],
        'save_path': './checkpoint/leopard_mobilenet-v2_regression.ckpt'
    },
    'mobilenet-v2_parallel':{
        'type': 'regression',
        'epoch': 20,
        'batch_size': 16,
        'lr': 0.0001,
        'loss': tf.keras.losses.mean_squared_error,
        'metrics': ['mse'],
        'save_path': './checkpoint/leopard_mobilenet-v2_parallel.ckpt'
    },
    'momobilenet-v2_finetune':{
        'type': 'regression',
        'epoch': 40,
        'batch_size': 16,
        'lr': 0.0001,
        'loss': tf.keras.losses.mean_squared_error,
        'metrics': ['mse'],
        'save_path': './checkpoint/leopard_mobilenet-v2_finetune.ckpt'

    }

}



if __name__=='__main__':

    tf.keras.layers.Conv2D()
    data_x,data_y=loadData(img_path)
    #Classification
    x_train, y_train, x_val, y_val, x_test, y_test = prePrec(data_x,data_y)
    model_clas=Net('classification')
    model_clas=train(model_clas,x_train,y_train,x_val,y_val, parameters=parameter_set['classification'])[0]
    print(model_clas.predict(x_test))
    print('Accuracy on testset:{}'.format(accuracy_score(y_test,model_clas.predict(x_test))))
    #Regression
    model_reg=Net('regression')
    model_reg=train(model_reg, x_train, y_train, x_val, y_val, parameters=parameter_set['regression'])[0]
    mse = tf.keras.losses.MeanSquaredError()
    loss = mse((y_val - 15) / 15., model_reg.predict(x_val).reshape(-1,))
    print('MSE on testset:{}'.format(loss))

    #######################MobileNet-v2############################
    #Use MobileNet-v2 without finetune
    x_train, y_train, x_val, y_val, x_test, y_test = prePrec(data_x, data_y,gray=True)
    base_model = tf.keras.applications.MobileNetV2(input_shape=commom_size,
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False
    global_average_layer = tf.keras.layers.GlobalMaxPooling2D()
    prediction_layer = tf.keras.layers.Dense(1,activation='tanh')

    inputs = tf.keras.Input(shape=commom_size)
    x = base_model(inputs)
    x = global_average_layer(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    train(model,x_train,y_train,x_val,y_val,parameters=parameter_set['mobilenet-v2_regression'])

    #Use MobileNet-v2 with parallel input
    x_train, y_train, x_val, y_val, x_test, y_test = prePrec(data_x, data_y,parallel=True)
    base_model = tf.keras.applications.MobileNetV2(input_shape=commom_size,
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False

    global_average_layer = tf.keras.layers.GlobalMaxPooling2D()
    d1=tf.keras.layers.Dense(120,activation='relu')
    d2=tf.keras.layers.Dense(84,activation='relu')
    prediction_layer = tf.keras.layers.Dense(1,activation='tanh')

    #Parallel input
    inputs1 = tf.keras.Input(shape=commom_size)
    inputs2 = tf.keras.Input(shape=commom_size)
    x1 = base_model(inputs1)
    x1 = tf.keras.Model(inputs1,x1)
    x2 = base_model(inputs2)
    x2 = tf.keras.Model(inputs2,x2)
    combine=K.concatenate([x1.output , x2.output])
    y=global_average_layer(combine)
    y=d1(y)
    y=d2(y)
    outputs = prediction_layer(y)

    model_mob = tf.keras.Model([x1.input,x2.input], outputs)
    model_mob=train(model_mob,x_train,y_train,x_val,y_val,parameters=parameter_set['mobilenet-v2_parallel'])


    #unfreeze the pretrained network
    base_model.trainable = True
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model_mob=train(model_mob[0],x_train,y_train,x_val,y_val,parameters=parameter_set['momobilenet-v2_finetune'],continue_from=model_mob[1].epoch[-1])
