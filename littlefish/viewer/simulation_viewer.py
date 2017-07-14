import sys

import h5py
import littlefish.core.fish as fi
import littlefish.core.plotting as pt
import littlefish.core.utilities as util
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QApplication, QTableWidgetItem, QSizePolicy
from littlefish.viewer.simulation_viewer_ui import Ui_SimulationViewer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

LAND_RGB = np.array([21, 174, 21], dtype=np.uint8)
SEA_RGB = np.array([21, 174, 225], dtype=np.uint8)
FISH_RGB = np.array([255, 255, 0], dtype=np.uint8)
FOOD_RGB = np.array([153, 51, 0], dtype=np.uint8)
PLOT_STEP = 10


def get_terrain_map_rgb(terrain_map_binary):

    terrain_map_rgb = np.zeros((terrain_map_binary.shape[0], terrain_map_binary.shape[1], 3), dtype=np.uint8)
    land_rgb = LAND_RGB
    sea_rgb = SEA_RGB
    terrain_map_rgb[terrain_map_binary==1, :] = land_rgb
    terrain_map_rgb[terrain_map_binary==0, :] = sea_rgb

    return terrain_map_rgb


def add_fish_rgb(terrain_map_rgb, body_position):

    fish_rgb = FISH_RGB
    show_map_rgb = np.array(terrain_map_rgb)
    show_map_rgb[body_position[0] - 1: body_position[0] + 2,
                 body_position[1] - 1: body_position[1] + 2,
                 0] = fish_rgb[0]
    show_map_rgb[body_position[0] - 1: body_position[0] + 2,
                 body_position[1] - 1: body_position[1] + 2,
                 1] = fish_rgb[1]
    show_map_rgb[body_position[0] - 1: body_position[0] + 2,
                 body_position[1] - 1: body_position[1] + 2,
                 2] = fish_rgb[2]
    return show_map_rgb


def add_foods_rgb(show_map_rgb, food_poss):

    food_rgb = FOOD_RGB
    for food_pos in food_poss:
        show_map_rgb[food_pos[0], food_pos[1], :] = food_rgb


class MatplotlibCavas(FigureCanvas):

    def __init__(self, parent=None):
        fig = Figure(dpi=100)
        self.axes = fig.add_axes([0., 0., 1., 1.])
        self.axes.set_aspect('equal')
        self.axes.set_axis_off()
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def plot_rgb(self, rgb):
        self.axes.cla()
        self.axes.set_axis_off()
        self.axes.imshow(rgb)
        self.draw()

    def clear(self):
        self.axes.cla()
        self.axes.set_axis_off()
        self.draw()


class SimulationViewer(Ui_SimulationViewer):
    def __init__(self, dialog):
        Ui_SimulationViewer.__init__(self)

        self.setupUi(dialog)
        self.MovieCanvas = MatplotlibCavas()
        self.MovieLayout.addWidget(self.MovieCanvas)
        self.PlayTimer = QTimer(self.MovieCanvas)

        self.ChooseFileButton.clicked.connect(self.get_file)
        self.SimulationComboBox.activated[str].connect(self._load_simulation)
        self.PlotBrainButton.clicked.connect(self._plot_brain)
        self.ClearFileButton.clicked.connect(self.clear_loaded_file)
        self.PlayPauseButton.clicked.connect(self._play_pause)
        self.PlayTimer.timeout.connect(self._show_next_frame)
        self.PlaySlider.sliderMoved.connect(self._slide_to_t)

        self.clear_loaded_file()

    def get_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        f_path, _ = QFileDialog.getOpenFileName(caption="QFileDialog.getOpenFileName()",
                                                directory="C:/little_fish_simulation_logs",
                                                filter="hdf Files (*.hdf5 *.h5);;")
        self.clear_loaded_file()
        self.PlotBrainButton.setEnabled(True)
        self.FilePathBrowser.setText(f_path)
        self._file = h5py.File(f_path, 'r')
        self._max_health = self._file['fish/max_health'].value
        self._show_fish_params()
        self._set_simulation_list()

    def _plot_brain(self):
        try:
            fish = fi.Fish.from_h5_group(self._file['fish'])
            fish_n = fish.name
            ax = pt.plot_brain(fish.get_brain())
            ax.set_title(fish_n)
            plt.show()
        except Exception as e:
            print (e)

    def _show_fish_params(self):
        try:
            self.FishTableWidget.setItem(0, 0, QTableWidgetItem('name'))
            self.FishTableWidget.setItem(1, 0, QTableWidgetItem('mother_name'))
            self.FishTableWidget.setItem(2, 0, QTableWidgetItem('max_health'))
            self.FishTableWidget.setItem(3, 0, QTableWidgetItem('food_rate'))
            self.FishTableWidget.setItem(4, 0, QTableWidgetItem('health_decay_rate'))
            self.FishTableWidget.setItem(5, 0, QTableWidgetItem('land_penalty_rate'))
            self.FishTableWidget.setItem(0, 1, QTableWidgetItem(util.decode(self._file['fish/name'].value)))
            self.FishTableWidget.setItem(1, 1, QTableWidgetItem(util.decode(self._file['fish/mother_name'].value)))
            self.FishTableWidget.setItem(2, 1, QTableWidgetItem(str(self._file['fish/max_health'].value)))
            self.FishTableWidget.setItem(3, 1, QTableWidgetItem(str(self._file['fish/food_rate_per_pixel'].value)))
            self.FishTableWidget.setItem(4, 1, QTableWidgetItem(str(self._file['fish/health_decay_rate_per_tu'].value)))
            self.FishTableWidget.setItem(5, 1, QTableWidgetItem(str(self._file['fish/land_penalty_rate_per_pixel_tu'].value)))
        except Exception as e:
            print(e)

    def _set_simulation_list(self):
        try:
            self.SimulationComboBox.addItems([s for s in self._file.keys() if s[0: 10] == 'simulation'])
            self._load_simulation(self.SimulationComboBox.itemText(0))
        except Exception as e:
            print (e)

    def _show_simulation_results(self, sim_grp):
        try:
            last_t = sim_grp['simulation_log/last_time_point'].value
            try:
               total_move = sim_grp['simulation_log/total_moves'].value
            except KeyError:
                total_move = None
            max_length = sim_grp['simulation_length'].value
            ending_time = util.decode(sim_grp['ending_time'].value)
            self.SimulationTableWidget.setItem(0, 0, QTableWidgetItem('last_time_point'))
            self.SimulationTableWidget.setItem(1, 0, QTableWidgetItem('total_moves'))
            self.SimulationTableWidget.setItem(2, 0, QTableWidgetItem('max_length'))
            self.SimulationTableWidget.setItem(3, 0, QTableWidgetItem('ending_time'))
            self.SimulationTableWidget.setItem(0, 1, QTableWidgetItem(str(last_t)))
            self.SimulationTableWidget.setItem(1, 1, QTableWidgetItem(str(total_move)))
            self.SimulationTableWidget.setItem(2, 1, QTableWidgetItem(str(max_length)))
            self.SimulationTableWidget.setItem(3, 1, QTableWidgetItem(ending_time))
        except Exception as e:
            print (e)

    def _show_terrain_params(self, sim_grp):
        try:
            terr_shape = sim_grp['simulation_log/terrain_map'].shape
            food_num = sim_grp['simulation_log/food_pos_history'].shape[1]
            terrain_map = sim_grp['simulation_log/terrain_map'].value
            sea_potion = 1. - (np.sum(terrain_map.flat) / float(terrain_map.shape[0] * terrain_map.shape[1]))
            self.TerrainTableWidget.setItem(0, 0, QTableWidgetItem('terrain_shape'))
            self.TerrainTableWidget.setItem(1, 0, QTableWidgetItem('food_number'))
            self.TerrainTableWidget.setItem(2, 0, QTableWidgetItem('sea_potion'))
            self.TerrainTableWidget.setItem(0, 1, QTableWidgetItem(str(terr_shape)))
            self.TerrainTableWidget.setItem(1, 1, QTableWidgetItem(str(food_num)))
            self.TerrainTableWidget.setItem(2, 1, QTableWidgetItem(str(sea_potion)))
        except Exception as e:
            print (e)

    def _load_simulation(self, simulation_key):

        self._is_playing = False
        self._curr_t_point = 0

        if simulation_key in self._file.keys():
            sim_grp = self._file[simulation_key]
            self._show_terrain_params(sim_grp)
            self._show_simulation_results(sim_grp)
            self._terrain_map_rgb = get_terrain_map_rgb(sim_grp['simulation_log/terrain_map'].value)
            self._health_history = sim_grp['simulation_log/health'].value
            self._fish_pos_history =  sim_grp['simulation_log/position_history'].value
            self._food_pos_history = sim_grp['simulation_log/food_pos_history'].value
            self._total_t_point = sim_grp['simulation_log/last_time_point'].value - 1
            self.PlayPauseButton.setEnabled(True)
            self.PlaySlider.setEnabled(True)
            self.TimeTextBrowser.setEnabled(True)
            self.HealthTextBrowser.setEnabled(True)
            self.PlaySlider.setRange(0, self._total_t_point)
            self._show_curr_map()

    def _show_curr_map(self):
        try:
            curr_fish_pos = self._fish_pos_history[self._curr_t_point]
            curr_food_poss = self._food_pos_history[self._curr_t_point]
            curr_health = self._health_history[self._curr_t_point]
            curr_map_rgb = add_fish_rgb(terrain_map_rgb=self._terrain_map_rgb, body_position=curr_fish_pos)
            add_foods_rgb(show_map_rgb=curr_map_rgb, food_poss=curr_food_poss)
            self.TimeTextBrowser.setText('{:7d} / {:7d}'.format(self._curr_t_point, self._total_t_point))
            self.HealthTextBrowser.setText('{:5.2f} / {:5.2f}'.format(curr_health, self._max_health))
            self.MovieCanvas.plot_rgb(curr_map_rgb)
        except Exception as e:
            print (e)

    def _show_next_frame(self, step=PLOT_STEP):
        try:
            self._curr_t_point = (self._curr_t_point + step) % self._total_t_point
            self.PlaySlider.setSliderPosition(self._curr_t_point)
            self._show_curr_map()
        except Exception as e:
            print(e)

    def _play_pause(self):
        if not self._is_playing:
            self._is_playing = True
            self.PlayPauseButton.setText('Pause')
            self.PlayTimer.start(40)
        else:
            self._is_playing = False
            self.PlayTimer.stop()
            self.PlayPauseButton.setText('Play')

    def _slide_to_t(self):
        self._curr_t_point = int(self.PlaySlider.value())

    def clear_loaded_file(self):

        self._is_playing = False
        self.PlayTimer.stop()
        self.PlayPauseButton.setText('Play')

        self._file = None
        self._curr_t_point = None
        self._terrain_map_rgb = None
        self._fish_pos_history = None
        self._health_history = None
        self._food_pos_history = None
        self._max_health = None
        self._total_t_point = None

        self.MovieCanvas.clear()

        self.PlayPauseButton.setEnabled(False)
        self.PlaySlider.setEnabled(False)
        self.SimulationComboBox.clear()
        self.TerrainTableWidget.clear()
        self.SimulationTableWidget.clear()
        self.FilePathBrowser.clear()
        self.PlotBrainButton.setEnabled(False)
        self.FishTableWidget.clear()
        self.PlotBrainButton.setEnabled(False)
        self.TimeTextBrowser.clear()
        self.TimeTextBrowser.setEnabled(False)
        self.HealthTextBrowser.clear()
        self.HealthTextBrowser.setEnabled(False)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    dialog = QMainWindow()
    prog = SimulationViewer(dialog)
    dialog.show()
    sys.exit(app.exec_())