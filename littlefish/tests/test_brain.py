import unittest
import numpy as np
import littlefish.brain.brain as brain


class TestBrain(unittest.TestCase):
    def setup(self):
        pass

    def test_generate_minimal_brain(self):
        min_brain = brain.generate_minimal_brain()
        assert min_brain.get_postsynaptic_indices(0) == [1, 2]
        assert min_brain.get_presynaptic_indices(3) == [1, 2]


if __name__ == "__main__":
    test_brain = TestBrain()
    test_brain.test_generate_minimal_brain()
