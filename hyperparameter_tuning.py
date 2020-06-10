# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 10:12:10 2020

@author: Gaurav
"""

from tensorboard.plugins.hparams import api as hp
from sklearn import datasets
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input
from tensorflow.keras.models import Sequential

# import some data to play with
x = datasets.load_iris().data
y = datasets.load_iris().target

#%%

def scheduler(epoch):
    if epoch < 30:
        lr = 0.01
    if 30 < epoch < 100:
        lr = 0.005
    else:
        lr = 0.001
    tf.summary.scalar('learning rate', data=lr, step=epoch)
    return lr
lrs = tf.keras.callbacks.LearningRateScheduler(scheduler)

def model_run(hparams, log_seq):
    logdir= r'logs\t_{}'.format(log_seq)
    
    model = Sequential([Input((4,)),
                        Dense(hparams['num_units'], name='layer1_dense1'),
                        BatchNormalization(name='layer2_batch1'),
                        Dropout(hparams['dropout'], name='layer3_drop1'),
                        Dense(5, name='layer4_dense2'),
                        Dense(3, activation='softmax', name='layer5_sof1')])
    
    model.compile(hparams['optimizer'], loss='sparse_categorical_crossentropy', metrics=['acc'])
    
    model.fit(x, y, epochs=500, validation_split=0.2, shuffle=False,
              callbacks=[tf.keras.callbacks.TensorBoard(logdir),
                         hp.KerasCallback(logdir, hparams, trial_id=str(log_seq)),
                         lrs])

log_seq = 1
for num_units in [8, 16]:
    for dropout_rate in [0.1, 0.2]:
        for optimizer in ['rmsprop', 'adam', 'sgd']:
            hparams = {
                      'num_units': num_units,
                      'dropout': dropout_rate,
                      'optimizer': optimizer,
                  }
            model_run(hparams, log_seq)
            log_seq += 1
#to view logs, write following in the prompt: tensorboard --logdir logs
