import sys
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

import littlefish.core.fish as fi
import littlefish.core.utilities as util
import littlefish.log_analysis.simulation_log as sl
from littlefish.brain.functional import plot_brain_connections
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QApplication,
    QTableWidgetItem,
    QSizePolicy,
)
from littlefish.viewer.simulation_viewer_ui import Ui_SimulationViewer
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

LAND_RGB = np.array([46, 204, 113], dtype=int)
SEA_RGB = np.array([52, 152, 219], dtype=int)
FISH_RGB = np.array([241, 196, 15], dtype=int)
FOOD_RGB = np.array([157, 32, 45], dtype=int)
PLOT_STEP = 1


def get_terrain_map_rgb(terrain_map_binary):
    terrain_map_rgb = np.zeros(
        (terrain_map_binary.shape[0], terrain_map_binary.shape[1], 3), dtype=int
    )
    land_rgb = LAND_RGB
    sea_rgb = SEA_RGB
    terrain_map_rgb[terrain_map_binary == 1, :] = land_rgb
    terrain_map_rgb[terrain_map_binary == 0, :] = sea_rgb

    return terrain_map_rgb


def add_fish_rgb(terrain_map_rgb, body_position):
    fish_rgb = FISH_RGB
    show_map_rgb = np.array(terrain_map_rgb)
    show_map_rgb[
        body_position[0] - 1 : body_position[0] + 2,
        body_position[1] - 1 : body_position[1] + 2,
        0,
    ] = fish_rgb[0]
    show_map_rgb[
        body_position[0] - 1 : body_position[0] + 2,
        body_position[1] - 1 : body_position[1] + 2,
        1,
    ] = fish_rgb[1]
    show_map_rgb[
        body_position[0] - 1 : body_position[0] + 2,
        body_position[1] - 1 : body_position[1] + 2,
        2,
    ] = fish_rgb[2]
    return show_map_rgb


def add_foods_rgb(show_map_rgb, food_poss):
    food_rgb = FOOD_RGB
    for food_pos in food_poss:
        show_map_rgb[food_pos[0], food_pos[1], :] = food_rgb


class MatplotlibCavas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(dpi=100)
        self.axes = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        self.axes.set_aspect("equal")
        self.axes.set_frame_on(False)
        self.axes.set_axis_off()
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
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
        self.MovieToolbar = NavigationToolbar(
            self.MovieCanvas, self.ToolbarWidget, coordinates=True
        )
        self.PlayTimer = QTimer(self.MovieCanvas)

        self.ChooseFileButton.clicked.connect(self.get_file)
        self.SimulationComboBox.activated[str].connect(self._load_simulation)
        self.PlotBrainButton.clicked.connect(self._plot_brain)
        self.ClearFileButton.clicked.connect(self.clear_loaded_file)
        self.PlayPauseButton.clicked.connect(self._play_pause)
        self.PlayTimer.timeout.connect(self._show_next_frame)
        # self.PlaySlider.sliderMoved.connect(self._slide_to_t)
        self.PlaySlider.valueChanged.connect(self._slide_to_t)

        self._saved_directory = None
        self._file = None
        self.clear_loaded_file()

    def get_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        f_path, _ = QFileDialog.getOpenFileName(
            caption="QFileDialog.getOpenFileName()",
            directory=self._saved_directory,
            filter="hdf Files (*.hdf5 *.h5);;",
        )

        if f_path:
            self.clear_loaded_file()
            self.PlotBrainButton.setEnabled(True)
            self.FilePathBrowser.setText(f_path)
            self._saved_directory = os.path.dirname(f_path)
            self._file = h5py.File(f_path, "r")
            self._set_simulation_list()

    def _plot_brain(self):
        """this needs to be reimplemented"""
        f = plot_brain_connections(self.fish.brain)
        f.suptitle(self.fish_name)
        plt.tight_layout()
        plt.show()

    def _show_fish_params(self):
        try:
            self.FishTableWidget.setItem(0, 0, QTableWidgetItem("name"))
            self.FishTableWidget.setItem(1, 0, QTableWidgetItem("mother_name"))
            self.FishTableWidget.setItem(2, 0, QTableWidgetItem("max_health"))
            self.FishTableWidget.setItem(3, 0, QTableWidgetItem("food_rate"))
            self.FishTableWidget.setItem(4, 0, QTableWidgetItem("health_decay_rate"))
            self.FishTableWidget.setItem(5, 0, QTableWidgetItem("land_penalty_rate"))
            self.FishTableWidget.setItem(6, 0, QTableWidgetItem("move_penalty_rate"))
            self.FishTableWidget.setItem(7, 0, QTableWidgetItem("firing_penalty_rate"))
            self.FishTableWidget.setItem(8, 0, QTableWidgetItem("start_generation"))
            self.FishTableWidget.setItem(9, 0, QTableWidgetItem("current_generation"))
            self.FishTableWidget.setItem(10, 0, QTableWidgetItem("total_moves"))
            self.FishTableWidget.setItem(11, 0, QTableWidgetItem("total_firings"))
            self.FishTableWidget.setItem(12, 0, QTableWidgetItem("mean_firing_rate"))
            self.FishTableWidget.setItem(0, 1, QTableWidgetItem(self.fish.name))
            self.FishTableWidget.setItem(1, 1, QTableWidgetItem(self.fish.mother_name))
            self.FishTableWidget.setItem(
                2, 1, QTableWidgetItem(str(self.fish.max_health))
            )
            self.FishTableWidget.setItem(
                3, 1, QTableWidgetItem(str(self.fish.food_rate))
            )
            self.FishTableWidget.setItem(
                4, 1, QTableWidgetItem(f"{self.fish.health_decay_rate:.6f}")
            )
            self.FishTableWidget.setItem(
                5, 1, QTableWidgetItem(str(self.fish.land_penalty_rate))
            )
            self.FishTableWidget.setItem(
                6, 1, QTableWidgetItem(f"{self.fish.move_penalty_rate:.6f}")
            )
            self.FishTableWidget.setItem(
                7, 1, QTableWidgetItem(f"{self.fish.firing_penalty_rate:.8f}")
            )

            # this should be in simulation_log, which needs to be implemented
            if "generations" in self._file:
                self.FishTableWidget.setItem(
                    8, 1, QTableWidgetItem(str(self._file["generations"][0]))
                )
                self.FishTableWidget.setItem(
                    9, 1, QTableWidgetItem(str(self._file["generations"][-1]))
                )

            self.FishTableWidget.setItem(
                10,
                1,
                QTableWidgetItem(
                    str(self.sim_log.get_fish_total_moves(self.fish_name))
                ),
            )
            firing_total, firing_mean = self.sim_log.get_fish_firing_stats(
                self.fish_name
            )
            self.FishTableWidget.setItem(11, 1, QTableWidgetItem(str(firing_total)))
            self.FishTableWidget.setItem(12, 1, QTableWidgetItem(f"{firing_mean:.4f}"))

        except Exception as e:
            print(e)

    def _set_simulation_list(self):
        try:
            self.SimulationComboBox.addItems(
                [s for s in self._file.keys() if s.startswith("simulation_")]
            )
            self._load_simulation(self.SimulationComboBox.itemText(0))
        except Exception as e:
            print(e)

    def _show_simulation_results(self):
        try:
            self.SimulationTableWidget.setItem(
                0, 0, QTableWidgetItem("last_time_point")
            )
            self.SimulationTableWidget.setItem(1, 0, QTableWidgetItem("max_length"))
            self.SimulationTableWidget.setItem(2, 0, QTableWidgetItem("ending_time"))
            self.SimulationTableWidget.setItem(3, 0, QTableWidgetItem("random_seed"))
            self.SimulationTableWidget.setItem(
                4, 0, QTableWidgetItem("numpy_random_seed")
            )
            self.SimulationTableWidget.setItem(
                0, 1, QTableWidgetItem(str(self.sim_log.last_time_point))
            )
            self.SimulationTableWidget.setItem(
                1, 1, QTableWidgetItem(str(self.sim_log.max_simulation_length))
            )
            self.SimulationTableWidget.setItem(
                2, 1, QTableWidgetItem(self.sim_log.ending_time)
            )
            self.SimulationTableWidget.setItem(
                3, 1, QTableWidgetItem(str(self.sim_log.random_seed))
            )
            self.SimulationTableWidget.setItem(
                4, 1, QTableWidgetItem(str(self.sim_log.numpy_random_seed))
            )
        except Exception as e:
            print(e)

    def _show_terrain_params(self):
        try:
            terrain_map = self.sim_log.terrain_map
            sea_portion = 1.0 - (
                np.sum(terrain_map.flat)
                / float(terrain_map.shape[0] * terrain_map.shape[1])
            )
            self.TerrainTableWidget.setItem(0, 0, QTableWidgetItem("terrain_shape"))
            self.TerrainTableWidget.setItem(1, 0, QTableWidgetItem("food_number"))
            self.TerrainTableWidget.setItem(2, 0, QTableWidgetItem("sea_portion"))
            self.TerrainTableWidget.setItem(
                0, 1, QTableWidgetItem(str(self.sim_log.terrain_shape))
            )
            self.TerrainTableWidget.setItem(
                1, 1, QTableWidgetItem(str(self.sim_log.food_num))
            )
            self.TerrainTableWidget.setItem(2, 1, QTableWidgetItem(str(sea_portion)))
        except Exception as e:
            print(e)

    def _load_simulation(self, simulation_key):
        self._is_playing = False
        self._curr_t_point = 0

        if simulation_key in self._file.keys():
            self.sim_log = sl.SimulationLog(log=self._file[simulation_key])
            self.fish_name = self.sim_log.fish_names[
                0
            ]  # currently only look at the first fish
            self.fish = fi.load_fish_from_h5_group(self._file[f"fish_{self.fish_name}"])
            self._show_fish_params()
            self._show_terrain_params()
            self._show_simulation_results()
            self._total_t_point = self.sim_log.last_time_point - 1
            self._terrain_map_rgb = get_terrain_map_rgb(self.sim_log.terrain_map)
            self._food_pos_history = self.sim_log.get_food_position_history()
            self._health_history = self.sim_log.get_fish_health_history(self.fish_name)
            self._fish_pos_history = self.sim_log.get_fish_position_history(
                self.fish_name
            )

            self.PlayPauseButton.setEnabled(True)
            self.PlaySlider.setEnabled(True)
            self.TimeTextBrowser.setEnabled(True)
            self.HealthTextBrowser.setEnabled(True)
            self.PlaySlider.setRange(0, self._total_t_point)
            self.PlaySlider.setValue(0)
            self._slide_to_t()

    def _show_curr_map(self):
        try:
            curr_fish_pos = self._fish_pos_history[self._curr_t_point]
            curr_food_poss = self._food_pos_history[self._curr_t_point]
            curr_health = self._health_history[self._curr_t_point]
            curr_map_rgb = add_fish_rgb(
                terrain_map_rgb=self._terrain_map_rgb, body_position=curr_fish_pos
            )
            add_foods_rgb(show_map_rgb=curr_map_rgb, food_poss=curr_food_poss)
            self.TimeTextBrowser.setText("Time: {:7d}".format(self._curr_t_point))
            self.HealthTextBrowser.setText("HP: {:5.2f}".format(curr_health))
            self.MovieCanvas.plot_rgb(curr_map_rgb)
        except Exception as e:
            print(e)

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
            self.PlayPauseButton.setText("Pause")
            self.PlayTimer.start(40)
        else:
            self._is_playing = False
            self.PlayTimer.stop()
            self.PlayPauseButton.setText("Play")

    def _slide_to_t(self):
        self._curr_t_point = int(self.PlaySlider.value())
        if self._file is not None and not self._is_playing:
            self._show_curr_map()

    def clear_loaded_file(self):
        self._is_playing = False
        self.PlaySlider.setValue(0)
        self._slide_to_t()
        self.PlayTimer.stop()
        self.PlayPauseButton.setText("Play")

        self._file = None
        self.sim_log = None
        self.fish_name = None
        self.fish = None

        if hasattr(self, "_brain_figure"):
            if self._brain_figure is not None:
                plt.close(self._brain_figure)
        self._brain_figure = None

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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dialog = QMainWindow()
    prog = SimulationViewer(dialog)
    dialog.show()
    sys.exit(app.exec_())
