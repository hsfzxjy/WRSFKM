import os.path as osp

resolve = lambda *parts: osp.join(osp.dirname(__file__), '..', 'data', *parts)  # noqa


def mnist_10k():

    import mnist
    import numpy as np

    images, labels = mnist.MNIST(resolve('MNIST-10K')).load_testing()
    ndim = 784
    size = len(labels)
    C = 10
    X = np.array(images).reshape((size, ndim)) / 255

    return X, C, labels


def load_dataset(name):

    return globals()[name]()
