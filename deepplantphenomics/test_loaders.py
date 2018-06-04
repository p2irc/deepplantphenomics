import pytest
import os
import numpy as np
from . import loaders

@pytest.fixture
def csv_data(scope="module"):
    # hard coding csv data ("answers") from 'test_data/test_csv*.csv' to test against
    # columns are: labels, col1, col2
    col2 = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    col1 = ['7', '8', '6', '7', '5', '8', '7', '8', '5', '6']
    labels = ['ara2013_plant001', 'ara2013_plant002', 'ara2013_plant003', 'ara2013_plant004', 'ara2013_plant005',
               'ara2013_plant006', 'ara2013_plant007', 'ara2013_plant008', 'ara2013_plant009', 'ara2013_plant010']
    return {'col2':col2, 'col1':col1, 'labels':labels}


def test_read_csv_labels(csv_data):
    labels = loaders.read_csv_labels(os.path.join('test_data', 'test_csv.csv'))
    assert labels == csv_data['labels']
    labels = loaders.read_csv_labels(os.path.join('test_data', 'test_csv.csv'), 1)
    assert labels == csv_data['col1']
    labels = loaders.read_csv_labels(os.path.join('test_data', 'test_csv.csv'), 2)
    assert labels == csv_data['col2']

def test_read_csv_labels_and_ids(csv_data):
    labels, ids = loaders.read_csv_labels_and_ids(os.path.join('test_data', 'test_csv.csv'), 1, 0)
    assert labels == csv_data['col1']
    assert ids == csv_data['labels']

def test_read_csv_multi_labels_and_ids(csv_data):
    labels, ids = loaders.read_csv_multi_labels_and_ids(os.path.join('test_data', 'test_csv.csv'), 0)
    assert labels == [list(x) for x in zip(csv_data['col1'], csv_data['col2'])]
    assert ids == csv_data['labels']

# testing string_labels_to_sequential is difficult because it uses set() which is not consistent in what it returns

def test_indices_to_onehot():
    idx = np.array([0, 1, 2, 3])
    expected_output = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]])
    onehot = loaders.indices_to_onehot_array(idx)
    assert np.array_equal(onehot, expected_output)
    idx = np.array([3, 4])
    expected_output = np.array([[0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
    onehot = loaders.indices_to_onehot_array(idx)
    assert np.array_equal(onehot, expected_output)

# pascal

