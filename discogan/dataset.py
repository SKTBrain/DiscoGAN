import os
import cv2
import numpy as np
import pandas as pd
from scipy.misc import imresize
import scipy.io


dataset_path = './datasets/'
celebA_path = os.path.join(dataset_path, 'celebA')
handbag_path = os.path.join(dataset_path, 'edges2handbags')
shoe_path = os.path.join(dataset_path, 'edges2shoes')
facescrub_path = os.path.join(dataset_path, 'facescrub')
chair_path = os.path.join(dataset_path, 'rendered_chairs')
face_3d_path = os.path.join(dataset_path, 'PublicMM1', '05_renderings')
face_real_path = os.path.join(dataset_path, 'real_face')
car_path = os.path.join(dataset_path, 'data', 'cars')

def shuffle_data(da, db):
    a_idx = range(len(da))
    np.random.shuffle( a_idx )

    b_idx = range(len(db))
    np.random.shuffle(b_idx)

    shuffled_da = np.array(da)[ np.array(a_idx) ]
    shuffled_db = np.array(db)[ np.array(b_idx) ]

    return shuffled_da, shuffled_db

def read_images( filenames, domain=None, image_size=64):

    images = []
    for fn in filenames:
        image = cv2.imread(fn)
        if image is None:
            continue

        if domain == 'A':
            kernel = np.ones((3,3), np.uint8)
            image = image[:, :256, :]
            image = 255. - image
            image = cv2.dilate( image, kernel, iterations=1 )
            image = 255. - image
        elif domain == 'B':
            image = image[:, 256:, :]

        image = cv2.resize(image, (image_size,image_size))
        image = image.astype(np.float32) / 255.
        image = image.transpose(2,0,1)
        images.append( image )

    images = np.stack( images )
    return images

def read_attr_file( attr_path, image_dir ):
    f = open( attr_path )
    lines = f.readlines()
    lines = map(lambda line: line.strip(), lines)
    columns = ['image_path'] + lines[1].split()
    lines = lines[2:]

    items = map(lambda line: line.split(), lines)
    df = pd.DataFrame( items, columns=columns )
    df['image_path'] = df['image_path'].map( lambda x: os.path.join( image_dir, x ) )

    return df

def get_celebA_files(style_A, style_B, constraint, constraint_type, test=False, n_test=200):
    attr_file = os.path.join( celebA_path, 'list_attr_celeba.txt' )
    image_dir = os.path.join( celebA_path, 'img_align_celeba' )
    image_data = read_attr_file( attr_file, image_dir )

    if constraint:
        image_data = image_data[ image_data[constraint] == constraint_type]

    style_A_data = image_data[ image_data[style_A] == '1']['image_path'].values
    if style_B:
        style_B_data = image_data[ image_data[style_B] == '1']['image_path'].values
    else:
        style_B_data = image_data[ image_data[style_A] == '-1']['image_path'].values

    if test == False:
        return style_A_data[:-n_test], style_B_data[:-n_test]
    if test == True:
        return style_A_data[-n_test:], style_B_data[-n_test:]


def get_edge2photo_files(item='edges2handbags', test=False):
    if item == 'edges2handbags':
        item_path = handbag_path
    elif item == 'edges2shoes':
        item_path = shoe_path

    if test == True:
        item_path = os.path.join( item_path, 'val' )
    else:
        item_path = os.path.join( item_path, 'train' )

    image_paths = map(lambda x: os.path.join( item_path, x ), os.listdir( item_path ))

    if test == True:
        return [image_paths, image_paths]
    else:
        n_images = len( image_paths )
        return [image_paths[:n_images/2], image_paths[n_images/2:]]


def get_facescrub_files(test=False, n_test=200):
    actor_path = os.path.join(facescrub_path, 'actors', 'face' )
    actress_path = os.path.join( facescrub_path, 'actresses', 'face' )

    actor_files = map(lambda x: os.path.join( actor_path, x ), os.listdir( actor_path ) )
    actress_files = map(lambda x: os.path.join( actress_path, x ), os.listdir( actress_path ) )

    if test == False:
        return actor_files[:-n_test], actress_files[:-n_test]
    else:
        return actor_files[-n_test:], actress_files[-n_test:]


def get_chairs(test=False, half=None, ver=360, angle_info=False):
    chair_ids = os.listdir( chair_path )
    if test:
        current_ids = chair_ids[-10:]
    else:
        if half is None: current_ids = chair_ids[:-10]
        elif half == 'first': current_ids = chair_ids[:-10][:len(chair_ids)/2]
        elif half == 'last': current_ids = chair_ids[:-10][len(chair_ids)/2:]

    chair_paths = []

    for chair in current_ids:
        current_path = os.path.join( chair_path, chair, 'renders' )
        if not os.path.exists( current_path ): continue
        filenames = filter(lambda x: x.endswith('.png'), os.listdir( current_path ))

        for filename in filenames:
            angle = int(filename.split('_')[3][1:])
            filepath = os.path.join(current_path, filename)

            if ver == 180:
                if angle > 180 and angle < 360: chair_paths.append(filepath)
            if ver == 360:
                chair_paths.append(filepath)

    return chair_paths

def get_cars(test=False, ver=360, interval=1, half=None, angle_info=False, image_size=64, gray=True):
    car_files = map(lambda x: os.path.join(car_path, x), os.listdir( car_path ))
    car_files = filter(lambda x: x.endswith('.mat'), car_files)

    car_idx = map(lambda x: int(x.split('car_')[1].split('_mesh')[0]), car_files )
    car_df = pd.DataFrame( {'idx': car_idx, 'path': car_files}).sort_values(by='idx')

    car_files = car_df['path'].values

    if not test:
        car_files = car_files[:-14]
    else:
        car_files = car_files[-14:]

    car_images = []
    classes = []

    n_cars = len(car_files)
    car_idx = 0
    for car_file in car_files:
        if not car_file.endswith('.mat'): continue
        car_mat = scipy.io.loadmat(car_file)
        car_ims = car_mat['im']
        car_idx += 1

        if half == 'first':
            if car_idx > n_cars / 2:
                break
        elif half == 'last':
            if car_idx <= n_cars / 2:
                continue

        if ver == 360:
            for idx,i in enumerate(range(24)):
                car_image = car_ims[:,:,:,i,3]
                car_image = cv2.resize(car_image, (image_size,image_size))
                if gray:
                    car_image = cv2.cvtColor(car_image, cv2.COLOR_BGR2GRAY)
                    car_image = np.repeat(car_image[:,:,None], 3, 2)
                car_image = car_image.transpose(2,0,1)
                car_image = car_image.astype(np.float32)/255.
                car_images.append( car_image )
                if angle_info:
                    classes.append(idx)

        elif ver == 180:
            for idx,i in enumerate(range(5,-1,-1) + range(23,18,-1)):
                car_image = car_ims[:,:,:,i,3]
                car_image = cv2.resize(car_image, (image_size,image_size))
                if gray:
                    car_image = cv2.cvtColor(car_image, cv2.COLOR_BGR2GRAY)
                    car_image = np.repeat(car_image[:,:,None], 3, 2)
                car_image = car_image.transpose(2,0,1)
                car_image = car_image.astype(np.float32)/255.
                car_images.append( car_image )
		if angle_info:
		    classes.append( idx )

        elif ver == 90:
            for idx,i in enumerate(range(5,-1,-1)):
                car_image = car_ims[:,:,:,i,3]
                car_image = cv2.resize(car_image, (image_size,image_size))
                car_image = car_image.transpose(2,0,1)
                car_image = car_image.astype(np.float32)/255.
                car_images.append( car_image )
                if angle_info:
                    classes.append( idx )

    car_images = car_images[::interval]

    if angle_info:
        return np.stack(car_images), np.array(classes)

    return np.stack( car_images )

def get_faces_3d(test=False, half=None):
    files = os.listdir( face_3d_path )
    image_files = filter(lambda x: x.endswith('.png'), files)

    df = pd.DataFrame({'image_path': image_files})
    df['id'] = df['image_path'].map(lambda x: x.split('/')[-1][:20])
    unique_ids = df['id'].unique()

    if not test:
        if half is None:
            current_ids = unique_ids[:8]
        if half == 'first':
            current_ids = unique_ids[:4]
        if half == 'last':
            current_ids = unique_ids[4:8]
    else:
        current_ids = unique_ids[8:]

    groups = df.groupby('id')
    image_paths = []

    for current_id in current_ids:
        image_paths += groups.get_group(current_id)['image_path'].tolist()

    return image_paths
