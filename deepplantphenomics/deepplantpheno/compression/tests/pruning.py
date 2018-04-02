import unittest


class BasicTest(unittest.TestCase):
    def test_true(self):
        self.assertTrue(True)


class TestLoader(unittest.TestCase):
    def setUp(self):
        self.model = None
    def test_basic_model(self):
        # load up a basic network and test pruning
