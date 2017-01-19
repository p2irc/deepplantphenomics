import tensorflow as tf
import xml.etree.ElementTree as tree
import random
import os


def splitRawData(images, labels, ratio):
    # serialize labels if they are lists (e.g. for regression)
    if isinstance(labels, list):
        labels = [' '.join(label) for label in labels]

    total_samples = len(labels)
    num_training = int(total_samples * ratio)

    partitions = [0] * total_samples
    partitions[:num_training] = [1] * num_training
    random.shuffle(partitions)

    train_images, test_images = tf.dynamic_partition(images, partitions, 2)
    train_labels, test_labels = tf.dynamic_partition(labels, partitions, 2)

    return train_images, train_labels, test_images, test_labels


def readCSVLabels(file_name, column_number, character=','):
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


def readCSVLabelsAndIds(file_name, column_number, id_column_number, character=','):
    f = open(file_name, 'r')
    labels = []
    ids = []

    for line in f:
        line = line.rstrip()

        temp = line.split(character)
        labels.append(temp[column_number])
        ids.append(temp[id_column_number])

    return labels, ids


def readCSVMultiLabelsAndIds(file_name, id_column_number, character=','):
    f = open(file_name, 'r')
    labels = []
    ids = []

    for line in f:
        line = line.rstrip()

        temp = line.split(character)
        ids.append(temp[id_column_number])

        temp.pop(id_column_number)

        ids.append(temp)

    return labels, ids


def stringLabelsToSequential(labels):
    unique = set([label.strip() for label in labels])
    num_labels = range(len(unique))
    seq_labels = dict(zip(unique, num_labels))

    return [seq_labels[label.strip()] for label in labels]


def readBoundingBoxFromPascalVOC(file_name):
    root = tree.parse(file_name)

    filename = os.path.basename(root.find('path').text)

    e = root.find('object/bndbox')

    x_min = e.find('xmin').text
    x_max = e.find('xmax').text
    y_min = e.find('ymin').text
    y_max = e.find('ymax').text

    return filename, x_min, x_max, y_min, y_max