#!/usr/bin/env python3

import os
import os.path as osp
import subprocess
import hashlib

current_dir = osp.dirname(__file__)
resolve = lambda *parts: osp.join(current_dir, *parts)  # noqa

download_dir = resolve('downloaded_content')


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download(url, filename, md5_value):

    filename = resolve(download_dir, filename)

    if osp.isfile(filename) and md5(filename) == md5_value:
        return filename

    subprocess.run(['wget', url, '-O', filename])
    return filename


def coil20(filename):

    subprocess.run(['rm', 'coil-20', '-rf'])
    subprocess.run(['unzip', filename, '-d', current_dir])
    subprocess.run(['mv', '-f', 'coil-20-unproc', 'coil-20', ])


def coil100(filename):

    subprocess.run(['rm', 'coil-100', '-rf'])
    subprocess.run(['unzip', filename, '-d', current_dir])
    subprocess.run(['mv', '-f', 'coil-100', 'coil-100'])


if __name__ == '__main__':

    os.makedirs(download_dir, exist_ok=True)

    datasets = [
        ('http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-unproc.zip', 'coil20.zip', 'a04fd3c91db987e5e634e7e0945a430a', coil20),
        # ('http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip', 'coil100.zip', '', coil100)
    ]

    for url, filename, md5_value, action in datasets:
        action(download(url, filename, md5_value))
