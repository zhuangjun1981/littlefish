import timeit

setup = '''
import random
import numpy as np
import littlefish.fish.brain as brain

neuron = brain.Neuron()

def neuron_act1(simulation_length):
    for i in range(simulation_length):
        neuron.act(i, probability_base=random.random())

def neuron_act2(simulation_length):
    base = np.random.rand(simulation_length)
    for i in range(simulation_length):
        neuron.act(i, probability_base=base[i])
'''

timer_random = timeit.Timer('neuron_act1(10000000)',setup=setup)
timer_numpy = timeit.Timer('neuron_act2(10000000)',setup=setup)

print('random timing:', timer_random.repeat(3, 1))
print('numpy timing:', timer_numpy.repeat(3, 1))