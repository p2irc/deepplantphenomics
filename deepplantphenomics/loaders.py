import tensorflow.compat.v1 as tf
import xml.etree.ElementTree as Tree
import numpy as np
import random
import os
import datetime


def split_raw_data(images, labels, test_ratio=0, validation_ratio=0, moderation_features=None, augmentation_images=None,
                   augmentation_labels=None, split_labels=True, force_mask_creation=False):
    """Currently depends on test/validation_ratio being 0 when not using test/validation"""
    # serialize labels if they are lists (e.g. for regression)
    if isinstance(labels, list):
        if split_labels:
            labels = [' '.join(map(str, label)) for label in labels]

    # Get the mask that generates the train/test/val split of the dataset
    n_aug = len(augmentation_labels) if augmentation_images is not None and augmentation_labels is not None else 0
    mask = _get_split_mask(test_ratio, validation_ratio, len(labels), n_aug, force_mask_creation)

    # If we're using a training augmentation set, add them to the rest of the dataset
    if augmentation_images is not None and augmentation_labels is not None:
        images = images + augmentation_images
        labels = labels + augmentation_labels

    # create partitions, we set train/validation to None if they're not being used
    if test_ratio != 0 and validation_ratio != 0:
        train_images, test_images, val_images = tf.dynamic_partition(images, mask, 3)
        train_labels, test_labels, val_labels = tf.dynamic_partition(labels, mask, 3)
    elif test_ratio != 0 and validation_ratio == 0:
        train_images, test_images = tf.dynamic_partition(images, mask, 2)
        train_labels, test_labels = tf.dynamic_partition(labels, mask, 2)
        val_images, val_labels = None, None
    elif test_ratio == 0 and validation_ratio != 0:
        train_images, val_images = tf.dynamic_partition(images, mask, 2)
        train_labels, val_labels = tf.dynamic_partition(labels, mask, 2)
        test_images, test_labels = None, None
    else:
        # We are just training, but we still need partitions for rest of the code to interact with.
        # dynamic_partition returns a list, which is fine, but now it returns a list of length 1, so we index into it
        # with [0] to get what we want.
        train_images = tf.dynamic_partition(images, mask, 1)[0]
        train_labels = tf.dynamic_partition(labels, mask, 1)[0]
        test_images, test_labels = None, None
        val_images, val_labels = None, None

    # Also partition moderation features if present <-- NEEDS TO BE FIXED/IMPROVED
    train_mf, test_mf, val_mf = None, None, None
    if moderation_features is not None:
        train_mf, test_mf, val_mf = tf.dynamic_partition(moderation_features, mask, 2)

    return train_images, train_labels, train_mf, test_images, test_labels, test_mf, val_images, val_labels, val_mf


def _get_split_mask(test_ratio, validation_ratio, n_label, n_augmentation=0, force_mask_creation=False, mask_dir=None):
    if not mask_dir:
        mask_dir = os.path.curdir
    mask_name = os.path.join(mask_dir, "mask_ckpt.txt")

    # If there is a previously saved mask and we don't want to force a new one, load it from the current directory
    mask = []
    if not force_mask_creation:
        try:
            mask_file = open(mask_name, "r", encoding='utf-8-sig')
            with mask_file:
                for line in mask_file:
                    mask.append(int(line.rstrip()))
            print('{0}: {1}'.format(datetime.datetime.now().strftime("%I:%M%p"), "Loaded previous partition mask."))
        except FileNotFoundError:
            mask = []

    # If there is no previous mask or we're forcing it, we'll build one
    if not mask:
        print('{0}: {1}'.format(datetime.datetime.now().strftime("%I:%M%p"), 'Building new partition mask.'))
        mask = [0] * n_label
        val_mask_num = 1  # this changes depending on whether we are using testing or not
        val_start_idx = 0  # if no testing then we idx from beginning, else we change this if there is testing

        if test_ratio != 0:
            # creating a mask [1,1,1,...,0,0,0]
            num_test = int(n_label * test_ratio)
            mask[:num_test] = [1] * num_test
            val_mask_num = 2
            val_start_idx = num_test

        if validation_ratio != 0:
            # if test_ratio != 0 then val_num_mask = 2 and we will create a mask as [1,1,1,...,2,2,2,...,0,0,0,...]
            # otherwise we will only have train and validation thus creating a mask as [1,1,1,...,0,0,0]
            num_val = int(n_label * validation_ratio)
            mask[val_start_idx: val_start_idx + num_val] = [val_mask_num] * num_val

        # If we're using a training augmentation set, add them to the training portion
        if n_augmentation != 0:
            mask = mask + ([0] * n_augmentation)

        # make the split random <-- ESSENTIAL
        random.shuffle(mask)

        # save the mask file in current directory for future use
        with open(mask_name, 'w+', encoding='utf-8-sig') as mask_file:
            for entry in mask:
                mask_file.write(str(entry) + '\n')

    return mask


def label_string_to_tensor(x, batch_size, num_outputs=-1):
    sparse = tf.string_split(x, sep=' ')
    values = tf.string_to_number(sparse.values)
    dense = tf.reshape(values, [batch_size, num_outputs])
    return dense


def get_dir_images(dirname):
    return sorted([os.path.join(dirname, f) for f in os.listdir(dirname) if
                   os.path.isfile(os.path.join(dirname, f)) and f.endswith('.png')])


def read_csv_labels(file_name, column_number=False, character=','):
    f = open(file_name, 'r', encoding='utf-8-sig')
    labels = []

    for line in f:
        line = line.rstrip()

        if column_number is False:
            labels.append(line.split(character)[0])  # without [0], length 1 lists are added to labels
        else:
            temp = line.split(character)
            labels.append(temp[column_number])

    return labels


def read_csv_rows(file_name, column_number=False, character=','):
    """
    Reads the rows of a csv file and returns them as a list.

    read_csv_labels and its variants read column-wise, this function is needed for row-wise parsing
    """
    f = open(file_name, 'r', encoding='utf-8-sig')
    rows = []

    for line in f:
        line = line.rstrip()
        curr_row = line.split(character)
        rows.append(curr_row)

    return rows


def read_csv_labels_and_ids(file_name, column_number, id_column_number, character=','):
    f = open(file_name, 'r', encoding='utf-8-sig')
    labels = []
    ids = []

    for line in f:
        line = line.rstrip()

        temp = line.split(character)
        labels.append(temp[column_number])
        ids.append(temp[id_column_number])

    return labels, ids


def read_csv_multi_labels_and_ids(file_name, id_column_number, character=','):
    f = open(file_name, 'r', encoding='utf-8-sig')
    labels = []
    ids = []

    for line in f:
        line = line.rstrip()

        temp = line.split(character)
        ids.append(temp[id_column_number])

        temp.pop(id_column_number)

        labels.append(temp)

    return labels, ids


def string_labels_to_sequential(labels):
    unique = set([label.strip() for label in labels])
    num_labels = range(len(unique))
    seq_labels = dict(zip(unique, num_labels))

    return [seq_labels[label.strip()] for label in labels]


def indices_to_onehot_array(idx):
    onehot = np.zeros((idx.size, idx.max() + 1))
    onehot[np.arange(idx.size), idx] = 1

    return onehot


def read_single_bounding_box_from_pascal_voc(file_name):
    root = Tree.parse(file_name)

    filename = os.path.basename(root.find('path').text)

    e = root.find('object/bndbox')

    x_min = float(e.find('xmin').text)
    x_max = float(e.find('xmax').text)
    y_min = float(e.find('ymin').text)
    y_max = float(e.find('ymax').text)

    return filename, x_min, x_max, y_min, y_max


def pascal_voc_coordinates_to_pcv_coordinates(img_height, img_width, coords):
    """Converts bounding box coordinates defined in Pascal VOC format to x_adj, y_adj, w_adj, h_adj"""

    x_min = coords[0]
    x_max = coords[1]
    y_min = coords[2]
    y_max = coords[3]

    x_adj = int(x_min)
    y_adj = int(y_min)
    w_adj = int(x_max - img_width)
    h_adj = int(y_max - img_height)

    return [x_adj, y_adj, w_adj, h_adj]


def box_coordinates_to_pascal_voc_coordinates(coords):
    """Converts c1x,c1y,c2x,c2y... box coordinates to Pascal VOC format"""
    min_x = coords[0]
    max_x = coords[6]
    min_y = coords[1]
    max_y = coords[5]

    return min_x, max_x, min_y, max_y


def box_coordinates_to_xywh_coordinates(coords):
    """Converts x1,y1,x2,y2 to x,y,w,h where x,y is center point and w,h is width and height of the box"""
    x1 = int(coords[0])
    y1 = int(coords[1])
    x2 = int(coords[4])
    y2 = int(coords[5])

    w = x2 - x1
    h = y2 - y1
    x = int(w / 2 + x1)
    y = int(h / 2 + y1)

    return x, y, w, h
