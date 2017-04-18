import tensorflow as tf
import xml.etree.ElementTree as tree
import numpy as np
import random
import os


def split_raw_data(images, labels, ratio, moderation_features=None):
    # serialize labels if they are lists (e.g. for regression)
    if isinstance(labels, list):
        labels = [' '.join(map(str, label)) for label in labels]
        total_samples = len(labels)
    else:
        total_samples = labels.get_shape().as_list()[0]

    # calculate and perform random split
    num_training = int(total_samples * ratio)

    mask = [0] * total_samples
    mask[:num_training] = [1] * num_training
    random.shuffle(mask)

    train_images, test_images = tf.dynamic_partition(images, mask, 2)
    train_labels, test_labels = tf.dynamic_partition(labels, mask, 2)

    # Also partition moderation features if present
    if moderation_features is not None:
        train_mf, test_mf = tf.dynamic_partition(moderation_features, mask, 2)

        return train_images, train_labels, test_images, test_labels, train_mf, test_mf
    else:
        return train_images, train_labels, test_images, test_labels


def label_string_to_tensor(x, batch_size, num_outputs):
    sparse = tf.string_split(x, delimiter=' ')
    values = tf.string_to_number(sparse.values)
    dense = tf.reshape(values, (batch_size, num_outputs))

    return dense


def read_csv_labels(file_name, column_number=False, character=','):
    f = open(file_name, 'r')
    labels = []

    for line in f:
        line = line.rstrip()

        if column_number is False:
            labels.append(line.split(character))
        else:
            temp = line.split(character)
            labels.append(temp[column_number])

    return labels


def read_csv_labels_and_ids(file_name, column_number, id_column_number, character=','):
    f = open(file_name, 'r')
    labels = []
    ids = []

    for line in f:
        line = line.rstrip()

        temp = line.split(character)
        labels.append(temp[column_number])
        ids.append(temp[id_column_number])

    return labels, ids


def read_csv_multi_labels_and_ids(file_name, id_column_number, character=','):
    f = open(file_name, 'r')
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
    onehot = np.zeros((idx.size, idx.max()+1))
    onehot[np.arange(idx.size), idx] = 1

    return onehot


def read_single_bounding_box_from_pascal_voc(file_name):
    root = tree.parse(file_name)

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

    return (x_adj, y_adj, w_adj, h_adj)


def box_coordinates_to_pascal_voc_coordinates(coords):
    """Converts c1x,c1y,c2x,c2y... box coordinates to Pascal VOC format"""
    min_x = coords[0]
    max_x = coords[6]
    min_y = coords[1]
    max_y = coords[5]

    return (min_x, max_x, min_y, max_y)
