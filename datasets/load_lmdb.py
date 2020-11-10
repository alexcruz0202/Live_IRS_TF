
import numpy as np
import lmdb
import random
import cv2
from datasets import caffe_pb2


class LMDB():
    def __init__(self):
        self.CROP_SIZE = 128
        self.CROP_CHANNELS = 1

        self.g_lmdb_env = None
        self.g_lmdb_txn = None
        self.g_lmdb_cursor = None

        self.g_rand_generator = random
        self.g_flip_enable = True
        self.g_tf_input_enable = True
        self.g_tf_input_scale = 0.00390625

    def set_lmdb_info(self, path):
        self.g_lmdb_env = lmdb.open(path, readonly=True)
        self.g_lmdb_txn = self.g_lmdb_env.begin()
        self.g_lmdb_cursor = self.g_lmdb_txn.cursor()

        self.g_lmdb_cursor.first()
        lmdb_size = 0
        while len(self.g_lmdb_cursor.key()) != 0:
            lmdb_size += 1
            self.g_lmdb_cursor.next()

        return lmdb_size

    def load_lmdb_on_batch(self, batch_idx, batch_size):
        """Returns the i-th example."""

        st = batch_idx * batch_size
        en = (batch_idx + 1) * batch_size
        st_en = en - st

        # if en > self.db_size:
        #     en = self.db_size

        crop_size = self.CROP_SIZE

        batchX = np.empty((st_en, self.CROP_SIZE, self.CROP_SIZE, self.CROP_CHANNELS), dtype=np.float32)
        batchY = np.empty((st_en), dtype=int)
        if len(self.g_lmdb_cursor.key()) == 0:
            self.g_lmdb_cursor.first()

        for i in range(st_en):
            if len(self.g_lmdb_cursor.key()) == 0:
                self.g_lmdb_cursor.first()

            datum = caffe_pb2.Datum()
            datum.ParseFromString(self.g_lmdb_cursor.value())

            w = datum.width
            h = datum.height
            c = datum.channels
            y = datum.label

            assert (c == self.CROP_CHANNELS), "must equal channels."

            xint8 = np.fromstring(datum.data, dtype=np.uint8)
            xint8 = xint8.reshape(c, h, w)

            if self.g_rand_generator:
                top = random.randint(0, h - crop_size - 1)
                left = random.randint(0, w - crop_size - 1)
            else:
                top = (h - crop_size) // 2
                left = (w - crop_size) // 2

            bottom = top + crop_size
            right = left + crop_size
            xint8 = xint8[:, top:bottom, left:right]

            if self.g_flip_enable:
                if random.randint(0, 1):
                    xint8 = xint8[:, :, ::-1]

            xint8 = xint8.transpose(1, 2, 0)

            batchX[i, :, :, :] = xint8 * np.float32(self.g_tf_input_scale)
            batchY[i] = y

            self.g_lmdb_cursor.next()

        return (batchX, np.array(batchY, dtype=np.float32))