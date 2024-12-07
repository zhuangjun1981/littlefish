import numpy as np


class Connection:
    """
    synaptic connection between two neurons
    """

    def __init__(self, latency=3, amplitude=0.001, rise_time=5, decay_time=10):
        """

        :param latency: int, temporal latency from presynaptic neuron action to the postsynaptic effect onset, number
                        of time units
        :param amplitude: float, peak change of the firing rate in the postsynaptic neuron, probablity of a action per
                          time unit. can be positive (excitatiory) or negative (inhibitory)
        :param rise_time: int, temporal duration from onset to peak, number of time units
        :param decay_time: int, temporal duration from peak to baseline, number of time units
        """

        self.latency = int(latency)
        self.amplitude = float(amplitude)
        self.rise_time = int(rise_time)
        self.decay_time = int(decay_time)
        self.psp = self.generate_psp()
        self.type = "littlefish.brain.connection.Connection"

    def __str__(self):
        return f"{self.type} object"

    def set_latency(self, new_latency):
        self.latency = int(new_latency)
        self.psp = self.generate_psp()

    def set_ampletude(self, new_amplitude):
        self.amplitude = float(new_amplitude)
        self.psp = self.generate_psp()

    def set_rise_time(self, new_rise_time):
        self.rise_time = int(new_rise_time)
        self.psp = self.generate_psp()

    def set_decay_time(self, new_decay_time):
        self.decay_time = int(new_decay_time)
        self.psp = self.generate_psp()

    def generate_psp(self):
        """
        generate post synaptic probability wave form
        """

        psp = np.zeros(self.latency + self.rise_time + self.decay_time)
        psp[self.latency : self.latency + self.rise_time] = (
            self.amplitude
            * (np.arange(self.rise_time) + 1).astype(np.float32)
            / float(self.rise_time)
        )

        psp[-self.decay_time :] = (
            self.amplitude
            * (np.arange(self.decay_time, 0, -1) - 1).astype(np.float32)
            / float(self.decay_time)
        )

        return psp

    def set_params(self, latency=None, amplitude=None, rise_time=None, decay_time=None):
        """
        set new parameters and regenerate psp waveform

        :param latency: int, number of time units for time delay
        :param amplitude: float, peak probability
        :param rise_time: int, number of time units to rise to peak
        :param decay_time: int, number of time units to decay to baseline
        """

        changed = False

        if latency is not None:
            self.latency = int(latency)
            changed = True
        if amplitude is not None:
            self.amplitude = float(amplitude)
            changed = True
        if rise_time is not None:
            self.rise_time = int(rise_time)
            changed = True
        if decay_time is not None:
            self.decay_time = int(decay_time)
            changed = True

        if changed:
            self.psp = self.generate_psp()

    def act(
        self, t_point: int, postsynaptic_index: int, psp_waveforms: np.ndarray
    ) -> None:
        """
        if the presynaptic neuron fires at the 'time_point', a psp wave form will be generated and add to the
        input waveform of postsynaptic neuron defined by postsynaptic_index

        :param t_point: int, current time point as the index of time unit axis
        :param postsynaptic_index: uint, the index of postsynaptic neuron
        :param psp_waveforms: 2-d array, float 32, the psp waveforms of all neurons in the brain, neuron id x t-point,
                              the generated psp will be added to the postsynaptic_index th line of the array
        :return:
        """

        psp_end = t_point + len(self.psp)
        if psp_end <= psp_waveforms.shape[1]:
            psp_waveforms[postsynaptic_index, t_point:psp_end] += self.psp
        else:
            psp_waveforms[postsynaptic_index, t_point:] += self.psp[
                : psp_waveforms.shape[1] - t_point
            ]
