from PyQt4.QtCore import *
from PyQt4.QtGui import *

import pyqtgraph as pg


class ParamPicker(QWidget):

    def __init__(self, name, params, parent=None):
        super().__init__(parent)
        self._params = list(params)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self._name = QLabel(name)
        self._name.setAlignment(Qt.AlignCenter)
        self._layout.addWidget(self._name)
        self._labels = []
        self._edits = []
        for i in range(0, len(self._params)):
            horizontal_layout = QHBoxLayout()
            label = QLabel(self._params[i])
            label.setAlignment(Qt.AlignCenter)
            edit = QLineEdit()
            # edit.setMaximumHeight(20)
            edit.setAlignment(Qt.AlignCenter)
            self._labels.append(label)
            self._edits.append(edit)
            horizontal_layout.addWidget(label)
            horizontal_layout.addWidget(edit)
            self._layout.addLayout(horizontal_layout)
        self._layout.addStretch()

    def get_params(self, mapping):
        res = {}
        for i in range(len(self._params)):
            res[self._params[i]] = mapping[i](self._edits[i].text())
        return res


class MainWindow(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.resize(640, 480)

        self._axis_pen = pg.mkPen('r')
        self._layout = QVBoxLayout(self)
        self._tab_view = QTabWidget()
        self._tab_view.setUsesScrollButtons(False)
        self._harmonic = ParamPicker(
            'Harmonic motion', ["ω=", "x(0)=", "x'(0)="])
        self._damping = ParamPicker(
            'Damping motion', ["ω=", "δ=", "x(0)=", "x'(0)="])
        self._forced = ParamPicker(
            'Forced motion', ["ω=", "δ=", "x(0)=", "x'(0)=", "f(x)="])
        self._tab_view.insertTab(0, self._harmonic, 'Harmonic')
        self._tab_view.insertTab(1, self._damping, 'Damping')
        self._tab_view.insertTab(2, self._forced, 'Forced')
        self._layout.addWidget(self._tab_view)
        self._graph = pg.PlotWidget()
        self._graph.getPlotItem().addLine(x=0, pen=self._axis_pen)
        self._graph.getPlotItem().addLine(y=0, pen=self._axis_pen)
        self._layout.addWidget(self._graph)
        self._button = QPushButton("PUSH TO TEMPER")
        self._layout.addWidget(self._button)

    def submitContact(self):
        name = self.nameLine.text()

        if name == "":
            QMessageBox.information(self, "Empty Field",
                                    "Please enter a name and address.")
            return
        else:
            QMessageBox.information(self, "Success!", "Hello %s!" % name)


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    screen = MainWindow()
    screen.show()
    sys.exit(app.exec_())
