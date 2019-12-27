import struct
import numpy as np


class FileManager:
    @staticmethod
    def read_file(filename):
        with open(filename, 'rb') as f:
            zero, data_type, dims = struct.unpack('>HBB', f.read(4))
            shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

    @staticmethod
    def load_data_set():
        raw_train_set = FileManager.read_file("train-images.idx3-ubyte")

        train_set = np.reshape(raw_train_set, (60000, 28 * 28))
        train_label = FileManager.read_file("train-labels.idx1-ubyte")

        return train_set, train_label
