import os
import sys
import numpy as np
import tensorflow as tf
from models.LiveIRNet import LiveIRNet_K0
from datasets.load_lmdb import load_lmdb_on_batch


def main():
    model = LiveIRNet_K0(input_shape=INPUT_SHAPE, input_tensor=None, output_size=OUTPUT_SIZE)
    print(model.summary())

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),
                  loss=tf.keras.losses.sparse_categorical_crossentropy(from_logits=True),
                  metrics=['accuracy'])

    train_lmdb_size = load_lmdb_on_batch("", BATCH_SIZE)
    train_step_batch = train_lmdb_size // BATCH_SIZE
    valid_lmdb_size = load_lmdb_on_batch("", BATCH_SIZE)
    valid_step_batch = valid_lmdb_size // BATCH_SIZE

    print("training has been prepared.")
    for epoch in range(EPOCH_COUNT):
        train_acc = []
        train_loss = []
        for i in range(train_step_batch):
            train_x, train_y = load_lmdb_on_batch(i, BATCH_SIZE)

            train_acc = model.fit(train_x, train_y, epochs=1, verbose=0)

            train_acc.append(train_acc.history['acc'][0])
            train_loss.append(train_acc.history['loss'][0])

            sys.stdout.write('batch_count = {0} of {1} \r'.format(i, train_step_batch))
            sys.stdout.flush()

        train_acc = np.sum(np.asarray(train_acc)) * 100 / train_step_batch
        train_loss = np.sum(np.asarray(train_loss)) * 100 / train_step_batch
        print('train_acc: {0} \t train_loss: {1} on epoch {2}'.format(train_acc, train_loss, epoch))

        valid_acc = []
        valid_loss = []
        for i in range(valid_step_batch):
            valid_x, valid_y = load_lmdb_on_batch(i, BATCH_SIZE)

            valid_acc = model.fit(valid_x, valid_y, epochs=1, verbose=0)

            valid_acc.append(valid_acc.history['acc'][0])
            valid_loss.append(valid_acc.history['loss'][0])

            sys.stdout.write('batch_count = {0} of {1} \r'.format(i, valid_step_batch))
            sys.stdout.flush()

        valid_acc = np.sum(np.asarray(valid_acc)) * 100 / valid_step_batch
        valid_loss = np.sum(np.asarray(valid_loss)) * 100 / valid_step_batch
        print('valid_acc: {0} \t valid_loss: {1} on epoch {2}'.format(valid_acc, valid_loss, epoch))


if __name__ == '__main__':
    EPOCH_COUNT = 10
    OUTPUT_SIZE = 2
    IMAGE_SIZE = 128
    INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 1)
    BATCH_SIZE = 128
    main()
