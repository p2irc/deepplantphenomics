import shutil

import pytest
from unittest.mock import patch
import os
import numpy as np
from deepplantphenomics import loaders


@pytest.fixture(scope="module")
def test_data_dir():
    return os.path.join(os.path.dirname(__file__), 'test_data')


@pytest.fixture(scope='module')
def csv_data():
    # hard coding csv data ("answers") from 'test_data/test_csv*.csv' to test against
    # columns are: labels, col1, col2
    col2 = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    col1 = ['7', '8', '6', '7', '5', '8', '7', '8', '5', '6']
    labels = ['ara2013_plant001', 'ara2013_plant002', 'ara2013_plant003', 'ara2013_plant004', 'ara2013_plant005',
              'ara2013_plant006', 'ara2013_plant007', 'ara2013_plant008', 'ara2013_plant009', 'ara2013_plant010']
    return {'col2': col2, 'col1': col1, 'labels': labels}


def test_read_csv_labels(csv_data, test_data_dir):
    cvs_file = os.path.join(test_data_dir, 'test_csv.csv')
    labels = loaders.read_csv_labels(cvs_file)
    assert labels == csv_data['labels']
    labels = loaders.read_csv_labels(cvs_file, 1)
    assert labels == csv_data['col1']
    labels = loaders.read_csv_labels(cvs_file, 2)
    assert labels == csv_data['col2']


def test_read_csv_labels_and_ids(csv_data, test_data_dir):
    csv_file = os.path.join(test_data_dir, 'test_csv.csv')
    labels, ids = loaders.read_csv_labels_and_ids(csv_file, 1, 0)
    assert labels == csv_data['col1']
    assert ids == csv_data['labels']


def test_read_csv_multi_labels_and_ids(csv_data, test_data_dir):
    csv_file = os.path.join(test_data_dir, 'test_csv.csv')
    labels, ids = loaders.read_csv_multi_labels_and_ids(csv_file, 0)
    assert labels == [list(x) for x in zip(csv_data['col1'], csv_data['col2'])]
    assert ids == csv_data['labels']


# testing string_labels_to_sequential is difficult because it uses set() which is not consistent in what it returns


def test_indices_to_onehot():
    idx = np.array([0, 1, 2, 3])
    expected_output = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    onehot = loaders.indices_to_onehot_array(idx)
    assert np.array_equal(onehot, expected_output)

    idx = np.array([3, 4])
    expected_output = np.array([[0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
    onehot = loaders.indices_to_onehot_array(idx)
    assert np.array_equal(onehot, expected_output)


def test_get_split_mask():
    test_mask_name = os.path.join(os.path.curdir, 'mask_ckpt.txt')
    if os.path.exists(test_mask_name):
        os.remove(test_mask_name)

    # Make a new mask for no testing or validation
    mask = loaders._get_split_mask(0, 0, 10)
    assert os.path.exists(test_mask_name)
    assert mask.count(0) == 10 and mask.count(1) == 0 and mask.count(2) == 0

    # Try to make a new mask for some testing and validation, and get the previous mask instead
    mask = loaders._get_split_mask(0.2, 0.1, 10, 0, force_mask_creation=False)
    assert os.path.exists(test_mask_name)
    assert mask.count(0) == 10 and mask.count(1) == 0 and mask.count(2) == 0

    # Actually make a new mask for some testing and validation
    mask = loaders._get_split_mask(0.2, 0.1, 10, 0, force_mask_creation=True)
    assert os.path.exists(test_mask_name)
    assert mask.count(0) == 7 and mask.count(1) == 2 and mask.count(2) == 1

    # Make a new mask for some testing only with some augmentation data
    mask = loaders._get_split_mask(0.2, 0.0, 10, 2, force_mask_creation=True)
    assert os.path.exists(test_mask_name)
    assert mask.count(0) == 10 and mask.count(1) == 2 and mask.count(2) == 0

    # Make a new mask for some validation only
    mask = loaders._get_split_mask(0.0, 0.2, 10, 0, force_mask_creation=True)
    assert os.path.exists(test_mask_name)
    assert mask.count(0) == 8 and mask.count(1) == 2 and mask.count(2) == 0

    # Make a mask for 12 labels instead of 10, and have a new mask actually made regardless of force_mask_creation
    mask = loaders._get_split_mask(0.0, 0.25, 12, 0, force_mask_creation=False)
    assert os.path.exists(test_mask_name)
    assert mask.count(0) == 9 and mask.count(1) == 3 and mask.count(2) == 0

    os.remove(test_mask_name)


def test_get_dir_images():
    def make_fake_file(f_name):
        open(f_name, 'a').close()

    dir_name = os.path.join(os.path.curdir, 'fake_dir')
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)

    # Getting images from an empty directory
    os.mkdir(dir_name)
    ims = loaders.get_dir_images(dir_name)
    assert ims == []

    # Getting images when there are no images
    make_fake_file(os.path.join(dir_name, 'labels.csv'))
    os.mkdir(os.path.join(dir_name, 'patches'))
    ims = loaders.get_dir_images('fake_dir')
    assert ims == []

    # Getting images with different extensions
    make_fake_file(os.path.join(dir_name, 'im1.jpg'))
    make_fake_file(os.path.join(dir_name, 'im2.JPG'))
    make_fake_file(os.path.join(dir_name, 'im3.jpeg'))
    make_fake_file(os.path.join(dir_name, 'im4.png'))
    make_fake_file(os.path.join(dir_name, 'im5.tif'))
    ims = loaders.get_dir_images('fake_dir')
    assert ims == ['fake_dir/im1.jpg', 'fake_dir/im2.JPG', 'fake_dir/im3.jpeg', 'fake_dir/im4.png']

    shutil.rmtree(dir_name)
