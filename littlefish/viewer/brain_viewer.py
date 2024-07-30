import os
import sys

import h5py
import littlefish.core.fish as fi
import littlefish.core.plotting as pt
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QApplication
from brain_viewer_ui import Ui_BrainViewerMainWindow


class BrainViewer(Ui_BrainViewerMainWindow):
    def __init__(self, dialog):
        Ui_BrainViewerMainWindow.__init__(self)

        self.setupUi(dialog)

        self.ChooseFileButton.clicked.connect(self.plot_brain)

    def get_file_path(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        f_path, _ = QFileDialog.getOpenFileName(caption="QFileDialog.getOpenFileName()", directory="",
                                                filter="All Files (*);;hdf Files (*.hdf5, *.h5)", options=options)
        return f_path

    def plot_brain(self):

        f_path = self.get_file_path()

        try:
            ff = h5py.File(f_path, "a")
            brain = fi.Brain.from_h5_group(ff['fish/brain'])
            ax = pt.plot_brain(brain)
            fish_n = os.path.splitext(os.path.split(f_path)[1])[0]
            print(fish_n)
            ax.set_title(fish_n)
            plt.show()
        except Exception as e:
            print(e)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    dialog = QMainWindow()
    prog = BrainViewer(dialog)
    dialog.show()
    sys.exit(app.exec_())