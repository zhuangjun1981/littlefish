import os
import h5py
import numpy as np
from littlefish.brain.base import Neuron
from littlefish.brain.eyes import Eye
from littlefish.brain.functional import load_neuron_from_h5_group


def test_neuron_io():
    curr_folder = os.path.dirname(os.path.abspath(__file__))
    temp_path = os.path.join(curr_folder, "temp_file.h5")

    neuron = Neuron()
    f_temp = h5py.File(temp_path, "a")
    h5_grp = f_temp.create_group("neuron")
    neuron.to_h5_group(h5_grp)
    neuron2 = load_neuron_from_h5_group(h5_grp)
    f_temp.close()

    assert neuron.baseline_rate == neuron2.baseline_rate
    assert neuron.type == neuron2.type
    assert neuron.refractory_period == neuron2.refractory_period

    os.remove(temp_path)


def test_eye_io():
    curr_folder = os.path.dirname(os.path.abspath(__file__))
    temp_path = os.path.join(curr_folder, "temp_file.h5")

    eye = Eye()
    f_temp = h5py.File(temp_path, "a")
    h5_grp = f_temp.create_group("neuron")
    eye.to_h5_group(h5_grp)

    eye2 = load_neuron_from_h5_group(h5_grp)
    f_temp.close()

    assert eye.type == eye2.type
    assert eye.baseline_rate == eye2.baseline_rate
    assert eye.refractory_period == eye2.refractory_period
    assert eye.gain == eye2.gain
    assert eye.input_type == eye2.input_type
    assert np.array_equal(eye.eye_position, eye2.eye_position)
    assert np.array_equal(eye.rf_positions, eye2.rf_positions)
    assert np.array_equal(eye.rf_weights, eye2.rf_weights)

    os.remove(temp_path)


if __name__ == "__main__":
    test_neuron_io()
    test_eye_io()
