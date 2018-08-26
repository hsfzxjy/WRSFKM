import numpy as np
import os.path as osp
from skimage.transform import resize


resolve = lambda *parts: osp.join(osp.dirname(__file__), '..', '..', 'data', *parts)  # noqa


def mnist_10k():

    import h5py
    import mnist
    import numpy as np
    C = 10
    directory = resolve('MNIST-10K')

    h5fn = osp.join(directory, 'data.h5')
    if osp.isfile(h5fn):

        f = h5py.File(h5fn, 'r')
        X = np.array(f.get('X'))
        labels = np.array(f.get('labels'))

        return X, C, labels

    images, labels = mnist.MNIST(resolve('MNIST-10K')).load_testing()
    len_ = 11
    ndim = len_ ** 2
    size = len(labels)

    X = np.empty((size, ndim))
    for i, x in enumerate(images):
        img = np.array(x).reshape((28, 28)) / 255
        X[i, :] = resize(img, (len_, len_)).reshape(ndim)

    f = h5py.File(h5fn, 'w')
    f.create_dataset('X', data=X)
    f.create_dataset('labels', data=labels)
    f.close()

    return X, C, labels


def coil_20():

    from scipy.misc import imread
    from glob import glob
    import re

    reg = re.compile(r'/obj(\d+)')

    directory = resolve('coil-20')
    classes = []
    imgs = []

    for fn in glob(resolve(directory, '*.png')):
        class_ = int(reg.findall(fn)[0])
        classes.append(class_)

        img = resize(imread(fn, mode='L'), (16, 16)).flatten()
        imgs.append(img)

    return np.array(imgs), 20, np.array(classes)


def load_dataset(name):

    return globals()[name]()
