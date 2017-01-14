import tensorflow as tf
import random


def splitRawData(images, labels, ratio):
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


def readCSVMultiLabelsandIds(file_name, id_column_number, character=','):
    f = open(file_name, 'r')
    labels = []
    ids = []

    for line in f:
        line = line.rstrip()

        temp = line.split(character)
        ids.append(temp[id_column_number])

        temp.pop(id_column_number)

        ids.append((temp, 0))

    return labels, ids


def stringLabelsToSequential(labels):
    unique = set([label.strip() for label in labels])
    num_labels = range(len(unique))
    seq_labels = dict(zip(unique, num_labels))

    return [seq_labels[label.strip()] for label in labels]
