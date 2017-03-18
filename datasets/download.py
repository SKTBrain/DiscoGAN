"""
Modification of https://github.com/carpedm20/DCGAN-tensorflow/blob/master/download.py
Modification of https://github.com/faceteam/facescrub/download.py
Downloads the following:
- Celeb-A dataset
- Pix2Pix - Edges2Handbags dataset
- Pix2pix - Edges2Shoes dataset
- Facescrub dataset
"""

from __future__ import print_function
import os
from os.path import join, exists
import multiprocessing
import hashlib
import cv2
import sys
import zipfile
import argparse
from six.moves import urllib


parser = argparse.ArgumentParser(description='Download dataset for DiscoGAN.')
parser.add_argument('datasets', metavar='N', type=str, nargs='+', choices=['celebA', 'edges2handbags', 'edges2shoes', 'facescrub'],
                    help='name of dataset to download [celebA, edges2handbags, edges2shoes, facescrub]')


def download(url, path):
    filename = url.split('/')[-1]
    filepath = os.path.join(path, filename)
    u = urllib.request.urlopen(url)
    f = open(filepath, 'wb')
    filesize = int(u.headers["Content-Length"])
    print("Downloading: %s Bytes: %s" % (filename, filesize))

    downloaded = 0
    block_sz = 8192
    status_width = 70
    while True:
        buf = u.read(block_sz)
        if not buf:
            print('')
            break
        else:
            print('', end='\r')
        downloaded += len(buf)
        f.write(buf)
        status = (("[%-" + str(status_width + 1) + "s] %3.2f%%") %
                  ('=' * int(float(downloaded) / filesize * status_width) + '>', downloaded * 100. / filesize))
        print(status, end='')
        sys.stdout.flush()
    f.close()
    return filepath


def unzip(filepath):
    print("Extracting: " + filepath)
    dirpath = os.path.dirname(filepath)
    with zipfile.ZipFile(filepath) as zf:
        zf.extractall(dirpath)
    os.remove(filepath)


def download_celeb_a(dirpath):
    data_dir = 'celebA'
    if os.path.exists(os.path.join(dirpath, data_dir)):
        print('Found Celeb-A - skip')
        return
    url = 'https://drive.google.com/open?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM'
    filepath = download(url, dirpath)
    zip_dir = ''
    with zipfile.ZipFile(filepath) as zf:
        zip_dir = zf.namelist()[0]
        zf.extractall(dirpath)
    os.remove(filepath)
    os.rename(os.path.join(dirpath, zip_dir), os.path.join(dirpath, data_dir))

    attribute_url = 'https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AAB06FXaQRUNtjW9ntaoPGvCa?dl=0'
    filepath = download(attribute_url, dirpath)



def download_pix2pix(category):
    CMD = 'bash ./datasets/download_pix2pix.sh "%s"'
    res = os.system(CMD % (category))


def preprocess_facescrub(dirpath):
    data_dir = os.path.join(dirpath, 'facescrub')
    if os.path.exists(data_dir):
        print('Found Facescrub')
    else:
        os.mkdir(data_dir)
    files = ['./datasets/facescrub_actors.txt',
             './datasets/facescrub_actresses.txt']
    for f in files:
        with open(f, 'r') as fd:
            # strip first line
            fd.readline()
            names = []
            urls = []
            bboxes = []
            genders = []
            for line in fd.readlines():
                gender = f.split("_")[1].split(".")[0]
                components = line.split('\t')
                assert(len(components) == 6)
                name = components[0].replace(' ', '_')
                url = components[3]
                bbox = [int(_) for _ in components[4].split(',')]
                names.append(name)
                urls.append(url)
                bboxes.append(bbox)
                genders.append(gender)
        # every name gets a task
        last_name = names[0]
        task_names = []
        task_urls = []
        task_bboxes = []
        task_genders = []
        tasks = []
        for i in range(len(names)):
            if names[i] == last_name:
                task_names.append(names[i])
                task_urls.append(urls[i])
                task_bboxes.append(bboxes[i])
                task_genders.append(genders[i])
            else:
                tasks.append(
                    (data_dir, task_genders, task_names, task_urls, task_bboxes))
                task_names = [names[i]]
                task_urls = [urls[i]]
                task_bboxes = [bboxes[i]]
                task_genders = [genders[i]]
                last_name = names[i]
        tasks.append(
            (data_dir, task_genders, task_names, task_urls, task_bboxes))

        pool_size = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=pool_size, maxtasksperchild=2)
        pool.map(download_facescrub, tasks)
        pool.close()
        pool.join()


def download_facescrub((data_dir, genders, names, urls, bboxes)):
    """
        download from urls into folder names using wget
    """

    assert(len(names) == len(urls))
    assert(len(names) == len(bboxes))
    # download using external wget
    CMD = 'wget -c -t 1 -T 3 "%s" -O "%s"'
    for i in range(len(names)):
        directory = join(data_dir, genders[i])

        if not exists(directory):
            print(directory)
            os.mkdir(directory)
        fname = hashlib.sha1(urls[i]).hexdigest() + "_" + names[i] + '.jpg'
        dst = join(directory, fname)
        print("downloading", dst)
        if exists(dst):
            print("already downloaded, skipping...")
            continue
        else:
            res = os.system(CMD % (urls[i], dst))
        # get face
        face_directory = join(directory, 'face')
        if not exists(face_directory):
            os.mkdir(face_directory)
        img = cv2.imread(dst)
        if img is None:
            # no image data
            os.remove(dst)
        else:
            face_path = join(face_directory, fname)
            face = img[bboxes[i][1]:bboxes[i][3], bboxes[i][0]:bboxes[i][2]]
            cv2.imwrite(face_path, face)
            #write bbox to file
            with open(join(directory, '_bboxes.txt'), 'a') as fd:
                bbox_str = ','.join([str(_) for _ in bboxes[i]])
                fd.write('%s %s\n' % (fname, bbox_str))


if __name__ == '__main__':
    args = parser.parse_args()

    if 'celebA' in args.datasets:
        download_celeb_a('./datasets/')
    if 'edges2handbags' in args.datasets:
        download_pix2pix('edges2handbags')
    if 'edges2shoes' in args.datasets:
        download_pix2pix('edges2shoes')
    if 'facescrub' in args.datasets:
        preprocess_facescrub('./datasets/')
