import matplotlib.pyplot as plt
import numpy as np
import littlefish.core.utilities as util


def plot_brain(input_brain, plot_axis=None, cmap='RdBu_r', bl_range=(-0.1, 0.1), ca_lw_cap=5., ca_range=(-1., 1.),
               neuron_kws=None, connection_kws=None, layer_label_kws=None, neuron_label_kws=None):
    """
    plot the neurons and connections of a given brain, neuron color indicates its baseline firing rate, connection
    color indicates connection amplitude

    :param input_brain: littlefish.fish.Brain object
    :param plot_axis: matplotlib.Axes object
    :param cmap: matplotlib.cm object
    :param bl_range: tuple of two floats, plotting range of neuron baseline firing rate, to be mapped to the cmap
    :param ca_lw_cap: positive float, cap line width to indicating connection amplitude
    :param ca_range: tuple of two floats, plotting range of connection amplitude, to be mapped to the cmap and
                     line width
    :param neuron_kws: dict, keyword arguments to plot neuron, matplotlib.Axis.plot inputs
    :param connection_kws: dict, keyword arguments to plot connection, matplotl.Axis.plot inputs
    :param layer_label_kws: dict, keyword arguments, to print text
    :return:
    """

    if neuron_kws is None:
        neuron_kws = {'mew': 4, 'ms': 80.}

    if layer_label_kws is None:
        layer_label_kws = {'fontsize': 15, 'va': 'center', 'ha': 'center', 'color': '#333333',
                           'weight': 'semibold'}

    if neuron_label_kws is None:
        neuron_label_kws = {'fontsize': 15, 'va': 'center', 'ha': 'center', 'color': '#dddddd',
                            'family': 'monospace', 'weight': 'semibold'}

    if connection_kws is None:
        connection_kws = {}

    if plot_axis is None:
        f = plt.figure(figsize=(10, 8))
        plot_axis = f.add_axes([0.05, 0.05, 0.9, 0.9])

    plot_axis.get_figure().set_facecolor('#808080')

    layer_num = input_brain.layer_num
    layer_x_pos = np.linspace(0.1, 0.9, layer_num, endpoint=True)
    neurons_df = input_brain.get_neurons().copy()

    neuron_y_pos = []
    for layer_ind in range(layer_num):
        neuron_num = len(neurons_df[neurons_df['layer'] == float(layer_ind)])
        neuron_y_pos.append(np.linspace(0.85, 0.1, neuron_num, endpoint=True))

    xy_pos = [[[layer_x_pos[l], neuron_y_pos[l][n]] for n in range(len(neuron_y_pos[l]))] for l in range(layer_num)]
    xy_pos = np.concatenate([np.array(p) for p in xy_pos], axis=0)
    neurons_df['plot_y'] = xy_pos[:, 1]
    neurons_df['plot_x'] = xy_pos[:, 0]

    # print layer type
    for layer in range(layer_num):
        layer_type = input_brain.get_layer_type(layer)
        plot_axis.text(layer_x_pos[layer], 0.92, layer_type.upper(), **layer_label_kws)

    # plot connections
    for gap, gap_conn in input_brain._connections.items():
        for i in range(len(gap_conn.index)):
            for j in range(len(gap_conn.columns)):
                pre_ind = gap_conn.columns[j]
                post_ind = gap_conn.index[i]
                curr_conn = gap_conn.iloc[i, j]
                pre_x = neurons_df.loc[pre_ind, 'plot_x']
                pre_y = neurons_df.loc[pre_ind, 'plot_y']
                post_x = neurons_df.loc[post_ind, 'plot_x']
                post_y = neurons_df.loc[post_ind, 'plot_y']
                amp_raw = curr_conn.get_amplitude()
                amp_n = util.normalized_to_range(var=amp_raw, input_range=ca_range, output_range=(0., 1.))
                amp_c = util.value_2_rgb(value=amp_n, cmap=cmap)
                amp_lw = util.normalized_to_range(var=abs(amp_raw), input_range=[0, max([abs(c) for c in ca_range])],
                                                  output_range=[0., ca_lw_cap])
                plot_axis.plot([pre_x, post_x], [pre_y, post_y], color=amp_c, lw=amp_lw, **connection_kws)

    # plotting neurons
    for neuron_ind, neuron in neurons_df.iterrows():
        bl = util.normalized_to_range(var=neuron['neuron'].get_baseline_rate(),
                                      input_range=bl_range, output_range=(0., 1.))
        bl_c = util.value_2_rgb(value=bl, cmap=cmap)
        if neuron['neuron'].get_neuron_type() == 'neuron':
            plot_axis.plot(neuron['plot_x'], neuron['plot_y'], '.', mfc=bl_c, mec='#444444', **neuron_kws)
        elif neuron['neuron'].get_neuron_type() == 'muscle':
            plot_axis.plot(neuron['plot_x'], neuron['plot_y'], '.', mfc=bl_c, mec='#444444', **neuron_kws)
            plot_axis.text(neuron['plot_x'], neuron['plot_y'],
                           util.short(neuron['neuron'].get_direction()), **neuron_label_kws)
        elif neuron['neuron'].get_neuron_type() == 'eye':
            if neuron['neuron'].get_input_type() == 'terrain':
                plot_axis.plot(neuron['plot_x'], neuron['plot_y'], '.', mfc=bl_c, mec='#065535', **neuron_kws)
            elif neuron['neuron'].get_input_type() == 'food':
                plot_axis.plot(neuron['plot_x'], neuron['plot_y'], '.', mfc=bl_c, mec='#800000', **neuron_kws)
            plot_axis.text(neuron['plot_x'], neuron['plot_y'],
                           util.short(neuron['neuron'].get_direction()), **neuron_label_kws)

    plot_axis.set_axis_off()
    plot_axis.set_xlim([0, 1])
    plot_axis.set_ylim([0, 1])

    return plot_axis

