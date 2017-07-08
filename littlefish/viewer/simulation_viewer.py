import sys
import os
import h5py
import matplotlib.pyplot as plt
import littlefish.core.fish as fi
import littlefish.viewer.plotting_tools as pt
import littlefish.core.utilities as util
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QApplication, QTableWidgetItem
from simulation_viewer_ui import Ui_SimulationViewer


class SimulationViewer(Ui_SimulationViewer):
    def __init__(self, dialog):
        Ui_SimulationViewer.__init__(self)

        self.setupUi(dialog)

        self._file = None

        self.ChooseFileButton.clicked.connect(self.get_file)
        self.SimulationComboBox.activated[str].connect(self._load_simulation)
        self.PlotBrainButton.setEnabled(False)
        self.PlotBrainButton.clicked.connect(self._plot_brain)

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
            self.FishTableWidget.setRowCount(6)
            self.FishTableWidget.setColumnCount(2)
            self.FishTableWidget.setColumnWidth(0, 150)
            self.FishTableWidget.setColumnWidth(1, 150)
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
            self.SimulationTableWidget.setRowCount(4)
            self.SimulationTableWidget.setColumnCount(2)
            self.SimulationTableWidget.setColumnWidth(0, 150)
            self.SimulationTableWidget.setColumnWidth(1, 150)
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
            self.TerrainTableWidget.setRowCount(2)
            self.TerrainTableWidget.setColumnCount(2)
            self.TerrainTableWidget.setColumnWidth(0, 150)
            self.TerrainTableWidget.setColumnWidth(1, 150)
            self.TerrainTableWidget.setItem(0, 0, QTableWidgetItem('terrain_shape'))
            self.TerrainTableWidget.setItem(1, 0, QTableWidgetItem('food_number'))
            self.TerrainTableWidget.setItem(0, 1, QTableWidgetItem(str(terr_shape)))
            self.TerrainTableWidget.setItem(1, 1, QTableWidgetItem(str(food_num)))
        except Exception as e:
            print (e)

    def _load_simulation(self, simulation_key):

        is_playing = False

        if simulation_key:
            sim_grp = self._file[simulation_key]
            self._show_terrain_params(sim_grp)
            self._show_simulation_results(sim_grp)

    def clear_loaded_file(self):
        self.PlotBrainButton.setEnabled(False)
        self.FishTableWidget.clear()
        self.TerrainTableWidget.clear()
        self.SimulationTableWidget.clear()
        self.SimulationComboBox.clear()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    dialog = QMainWindow()
    prog = SimulationViewer(dialog)
    dialog.show()
    sys.exit(app.exec_())