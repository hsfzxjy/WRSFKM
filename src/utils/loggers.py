import h5py
import numpy as np


class Logger:

    def __init__(self, filename):

        if filename:
            self.file = h5py.File(filename, 'w')
            group = self.file
        else:
            group = None

        self.group = group
        self.middle_data = []
        self.final_data = None
        self.iterations = 0

    def log_middle(self, E, metric):

        if self.group is None:
            print(self.iterations, E, *metric)
        else:
            self.middle_data.append((E, *metric))

        self.iterations += 1

    def log_final(self, U, V, t, metric):

        if self.group is None:
            print('result:', t, *metric)
        else:
            self.U = U
            self.V = V
            self.result = (t, *metric)

    def close(self):

        if self.group is None:
            return

        self.group.create_dataset('middle', data=np.array(self.middle_data))
        self.group.create_dataset('U', data=np.array([]))  # self.U)
        self.group.create_dataset('V', data=np.array([]))  # self.V)
        self.group.create_dataset('result', data=np.array(self.result))

        self.file.close()
