
import numpy as np
import lmdb
import random
import caffe_pb2

CROP_SIZE = 128
CROP_CHANNELS = 1

g_lmdb_env = None
g_lmdb_txn = None
g_lmdb_cursor = None

g_rand_generator = random
g_flip_enable = True
g_tf_input_enable = True
g_tf_input_scale = 0.00390625


def set_lmdb_info(path):
    g_lmdb_env = lmdb.open(path, readonly=True)
    g_lmdb_txn = g_lmdb_env.begin()
    g_lmdb_cursor = g_lmdb_txn.cursor()

    key = g_lmdb_cursor.key()
    lmdb_size = 0
    while len(key) != 0:
        lmdb_size += 1
        key = g_lmdb_cursor.key()
        g_lmdb_cursor.next()

    return lmdb_size


def load_lmdb_on_batch(batch_idx, batch_size):
    """Returns the i-th example."""

    st = batch_idx * batch_size
    en = (batch_idx + 1) * batch_size
    st_en = en - st

    # if en > self.db_size:
    #     en = self.db_size

    crop_size = CROP_SIZE

    batchX = np.empty((st_en, CROP_SIZE, CROP_SIZE, CROP_CHANNELS), dtype=np.float32)
    batchY = np.empty((st_en), dtype=int)

    for i in range(st_en):
        key = g_lmdb_cursor.key()


        datum = caffe_pb2.Datum()
        datum.ParseFromString(g_lmdb_cursor.value())

        w = datum.width
        h = datum.height
        c = datum.channels
        y = datum.label

        assert (c == CROP_CHANNELS), "must equal channels."

        xint8 = np.fromstring(datum.data, dtype=np.int8)
        xint8 = xint8.reshape(c, h, w)

        if g_rand_generator:
            top = random.randint(0, h - crop_size - 1)
            left = random.randint(0, w - crop_size - 1)
        else:
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2

        bottom = top + crop_size
        right = left + crop_size
        xint8 = xint8[:, top:bottom, left:right]

        if g_flip_enable:
            if random.randint(0, 1):
                xint8 = xint8[:, :, ::-1]

        if g_tf_input_enable:
            xint8.transpose(1, 2, 0)

        batchX[i, :, :, :] = xint8 * np.float32(g_tf_input_scale)
        batchY[i] = y

        g_lmdb_cursor.next()

    return (batchX, batchY)