import os
import sys
import numpy as np
import tensorflow as tf
from models.LiveIRNet import LiveIRNet_K0
from datasets.load_lmdb import *
from tensorflow.keras.optimizers import SGD
from keras.models import load_model


def main():
    model = LiveIRNet_K0(input_shape=INPUT_SHAPE, input_tensor=None, output_size=OUTPUT_SIZE)
    model.load_weights("E:/Data/AYC/spoofing/live_irs_tf/models/2/328.h5")
    print(model.summary())

    optimizer = SGD(lr=1e-2, decay=1e-6, momentum=0.9)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])

    train_lmdb = LMDB()
    train_lmdb_size = train_lmdb.set_lmdb_info("E:/Data/AYC/spoofing/database/LevelDB/Train_LiveDepth_go_10_20201018-lmdb")
    #Train_LiveDepth_go_03_20200818-lmdb
    train_step_batch = train_lmdb_size // TRAIN_BATCH_SIZE

    valid_lmdb = LMDB()
    valid_lmdb_size = valid_lmdb.set_lmdb_info("E:/Data/AYC/spoofing/database/LevelDB/Valid_LiveDepth_go_09_20201017-lmdb")
    #valid_LiveDepth_go_09_20201017-lmdb
    valid_step_batch = valid_lmdb_size // VALID_BATCH_SIZE

    val_acc = []
    val_loss = []

    print("training has been prepared.")
    for epoch in range(EPOCH_COUNT):
        train_acc = []
        train_loss = []
        for i in range(train_step_batch):
            train_x, train_y = train_lmdb.load_lmdb_on_batch(i, TRAIN_BATCH_SIZE)

            # acc = model.train_on_batch(x=train_x, y=train_y)
            acc = model.fit(train_x, train_y, epochs=1, verbose=0)

            train_acc.append(acc.history['acc'][0])
            train_loss.append(acc.history['loss'][0])

            sys.stdout.write('batch_count = {0} of {1} \r'.format(i, train_step_batch))
            sys.stdout.flush()

        train_acc = np.sum(np.asarray(train_acc)) * 100 / train_step_batch
        train_loss = np.sum(np.asarray(train_loss)) / train_step_batch
        print('train_acc: {0} \t train_loss: {1} on epoch {2}'.format(train_acc, train_loss, epoch))

        valid_acc = []
        valid_loss = []
        for i in range(valid_step_batch):
            valid_x, valid_y = valid_lmdb.load_lmdb_on_batch(i, VALID_BATCH_SIZE)

            loss, acc = model.evaluate(valid_x, valid_y, verbose=0)

            valid_acc.append(acc)
            valid_loss.append(loss)

            sys.stdout.write('batch_count = {0} of {1} \r'.format(i, valid_step_batch))
            sys.stdout.flush()

        valid_acc = np.sum(np.asarray(valid_acc)) * 100 / valid_step_batch
        valid_loss = np.sum(np.asarray(valid_loss)) / valid_step_batch
        print('valid_acc: {0} \t valid_loss: {1} on epoch {2}'.format(valid_acc, valid_loss, epoch))

        model.save("./models/{}.h5".format(epoch))


if __name__ == '__main__':
    EPOCH_COUNT = 1000
    OUTPUT_SIZE = 2
    IMAGE_SIZE = 128
    INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 1)
    TRAIN_BATCH_SIZE = 256
    VALID_BATCH_SIZE = 128
    main()
