'''
The implementation of  dataset1 training for semantic segmentation

'''
import random
import sys
import time
import os
import cv2
import datetime
import pickle
import tensorflow as tf
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import optimizers
from models.FCN import FCN8
from utils.utils import commonUtils, get_dataset_info, print_time_log
from utils.ImageDataGenerator import ImageDataGenerator
from config import VGG_Weights_path


if __name__ == '__main__':
    starttime = datetime.datetime.now()
    #commonUtils.GPUConfig()
    dataset_name = 'dataset1'
    #dataset_dir = os.path.abspath('/dataset/dataset1')E:\src\dataset\dataset1
    dataset_dir = os.path.abspath('E:\\src\\dataset\\dataset1')

    train_gen = ImageDataGenerator(random_crop=True,
                                   rotation_range=0,
                                   brightness_range=None,
                                   zoom_range=[0.5, 2],
                                   channel_shift_range=0,
                                   horizontal_flip=True,
                                   vertical_flip=False)

    valid_gen = ImageDataGenerator()
    train_image_names, train_label_names, valid_image_names, valid_label_names = get_dataset_info(
        dataset_name, dataset_dir)

    # config training parameters
    num_classes = 12
    train_batch_size = 16
    valid_batch_size = 16
    target_size = (224, 224)
    seed = 20191224
    data_aug_rate = 0
    num_valid_images = len(valid_image_names)

    # generator train and valid data
    train_generator = train_gen.flow(images_list=train_image_names,
                                     labels_list=train_label_names,
                                     num_classes=num_classes,
                                     batch_size=train_batch_size,
                                     target_size=target_size,
                                     pad_size=0.,
                                     shuffle=True,
                                     seed=seed,
                                     data_aug_rate=data_aug_rate)

    valid_generator = valid_gen.flow(images_list=valid_image_names,
                                     labels_list=valid_label_names,
                                     num_classes=num_classes,
                                     batch_size=valid_batch_size,
                                     target_size=target_size,
                                     pad_size=0.)
    # training and validation steps
    steps_per_epoch = len(train_image_names) // train_batch_size
    validation_steps = num_valid_images // valid_batch_size

    tensorboard = TensorBoard(
        log_dir='./logs/dataset1/FCN-dataset1-{}'.format(time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime())))
    model = FCN8(nClasses=12,
                 input_height=224,
                 input_width=224,
                 VGG_Weights_path=VGG_Weights_path)
    model.summary()

    sgd = optimizers.SGD(lr=1E-2, decay=5**(-4), momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy']
                  )
    best_weights_filepath = './data/FCN-dataset1-best_weights.hdf5'
    earlyStopping = EarlyStopping(
        monitor='val_loss', patience=30, verbose=2, mode='auto')
    saveBestModel = ModelCheckpoint(
        best_weights_filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10,
                                  verbose=2, mode='auto', epsilon=0.0001, cooldown=0, min_lr=1E-5)
    hist1 = model.fit_generator(generator=train_generator,
                                steps_per_epoch=steps_per_epoch,
                                validation_data=valid_generator,
                                validation_steps=validation_steps,
                                epochs=500,
                                verbose=1,
                                callbacks=[tensorboard, earlyStopping, saveBestModel, reduce_lr])
    with open('./data/FCN-dataset1.pickle', 'wb') as file_pi:
        pickle.dump(hist1.history, file_pi)
    endtime = datetime.datetime.now()
    print_time_log(starttime, endtime)
