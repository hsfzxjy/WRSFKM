import numpy as np
import os.path as osp

resolve = lambda *parts: osp.join(osp.dirname(__file__), '..', '..', 'data', *parts)  # noqa


def mnist_10k():

    import mnist
    import numpy as np

    images, labels = mnist.MNIST(resolve('MNIST-10K')).load_testing()
    ndim = 784
    size = len(labels)
    C = 10
    X = np.array(images).reshape((size, ndim)) / 255

    return X, C, labels


def coil_20():

    from scipy.misc import imread
    from skimage.transform import resize
    from glob import glob
    import re

    reg = re.compile(r'/obj(\d+)')

    directory = resolve('coil-20')
    classes = []
    imgs = []

    for fn in glob(resolve(directory, '*.png')):
        class_ = int(reg.findall(fn)[0])
        classes.append(class_)

        img = resize(imread(fn, mode='L'), (64, 64)).flatten()
        imgs.append(img)

    return np.array(imgs), 20, np.array(classes)


def load_dataset(name):

    return globals()[name]()
