from PyQt4.QtCore import *
from PyQt4.QtGui import *
import numpy as np
from numpy.random import normal
from scipy.integrate import odeint
from sympy import Symbol, lambdify
from sympy.parsing.sympy_parser import parse_expr
import pyqtgraph as pg
import pandas as pd

from lso import lso

CONST_OMEGA = 'ω'
CONST_DELTA = 'δ'
CONST_X0 = "x(0)"
CONST_DX0 = "x'(0)"
CONST_F = "f(t)"
CONST_N = "Number of measures"
CONST_SIGMA = "σ"
CONST_PRECISION = "Precision"

t_symbol = Symbol('t')

CONST_TIME_SIMULATING = 300
CONST_N_LINSPACE = 1000
CONST_h_VALUE = CONST_TIME_SIMULATING / CONST_N_LINSPACE

CONST_FIRST_WIDGET_NAME = "Forced"
CONST_SECOND_WIDGET_NAME = "MNKO diff"
CONST_THIRD_WIDGET_NAME = "MNKO"


def round_sigfigs(num, sig_figs=None):
    if sig_figs is None:
        return num
    elif num != 0:
        return round(num, -int(np.floor(np.log10(abs(num))) - (sig_figs - 1)))
    else:
        return 0


def round_sigfigs_array(x, sig_figs=None):
    if not sig_figs:
        return
    x_flat = x.reshape(-1)
    for i in range(len(x_flat)):
        x_flat[i] = round_sigfigs(x_flat[i], sig_figs=sig_figs)


def d_f(f_x_prev, f_x_next, h):
    return (f_x_next - f_x_prev) / (2 * h)


def d2_f(f_x_prev, f_x_curr, f_x_next, h):
    return (f_x_next - 2 * f_x_curr + f_x_prev) / (h ** 2)


def make_f(f, a):
    _a = np.array(a)

    def dec_f(y, t):
        res = np.roll(y, -1)
        res[-1] = f(t) - _a.dot(y)
        return res

    return dec_f


def unpack_range_to_type(to_type):
    def res(str_range):
        return [to_type(i) for i in str_range.split(',')]

    return res


def zero(*args, **kwargs):
    return 0


class FilePicker(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self._x_file = ""
        self._y_file = ""

        self._layout = QVBoxLayout(self)
        self._first_layer = QHBoxLayout()
        self._x_file_label = QLabel("No file picked")
        self._x_file_button = QPushButton("Pick X file")

        self._first_layer.addWidget(
            self._x_file_label, 0, Qt.AlignRight | Qt.AlignHCenter)
        self._first_layer.addWidget(
            self._x_file_button, 0, Qt.AlignLeft | Qt.AlignHCenter)
        self._layout.addLayout(self._first_layer)

        self._second_layer = QHBoxLayout()
        self._y_file_label = QLabel("No file picked")
        self._y_file_button = QPushButton("Pick y file")

        self._second_layer.addWidget(
            self._y_file_label, 0, Qt.AlignRight | Qt.AlignHCenter)
        self._second_layer.addWidget(
            self._y_file_button, 0, Qt.AlignLeft | Qt.AlignHCenter)
        self._layout.addLayout(self._second_layer)

        self._layout.addStretch()

        self._x_file_button.clicked.connect(self.x_file_button_event)
        self._y_file_button.clicked.connect(self.y_file_button_event)

    def x_file_button_event(self):
        file_name = QFileDialog.getOpenFileName(
            self, 'Open file', '', 'CSV (*.csv);;Excel (*.xls *.xlsx)')
        if file_name != '':
            self._x_file = file_name
            self._x_file_label.setText(file_name)

    def y_file_button_event(self):
        file_name = QFileDialog.getOpenFileName(
            self, 'Open file', '', 'CSV (*.csv);;Excel (*.xls *.xlsx)')
        if file_name != '':
            self._y_file = file_name
            self._y_file_label.setText(file_name)

    def get_x_file_name(self):
        return self._x_file

    def get_y_file_name(self):
        return self._y_file


class ParamPicker(QWidget):

    def __init__(self, name, params, defaults, mapping, child_widgets=None, parent=None):
        super().__init__(parent)
        self._name = str(name)
        self._params = list(params)
        self._defaults = list(defaults)
        self._mapping = list(mapping)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self._labels = []
        self._edits = []
        for param, default in zip(self._params, self._defaults):
            horizontal_layout = QHBoxLayout()
            label = QLabel(param)
            label.setAlignment(Qt.AlignCenter)
            edit = QLineEdit()
            edit.setText(default)
            edit.setAlignment(Qt.AlignCenter)
            self._labels.append(label)
            self._edits.append(edit)
            horizontal_layout.addWidget(
                label, 0, Qt.AlignRight | Qt.AlignHCenter)
            horizontal_layout.addWidget(
                edit, 0, Qt.AlignLeft | Qt.AlignHCenter)
            self._layout.addLayout(horizontal_layout)
        if child_widgets is not None:
            for widget in child_widgets:
                self._layout.addWidget(widget)
        self._layout.addStretch()

    def get_params(self):
        res = {'widget_name': self._name}
        for i in range(len(self._params)):
            res[self._params[i]] = self._mapping[i](self._edits[i].text())
        return res


class MainWindow(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self._axis_pen = pg.mkPen('r')
        self._default_line_pen = pg.mkPen(color='#0C27D9', width=2)
        self._plot_background = pg.mkBrush(color='#bfbfbf')
        self._error_box = QErrorMessage()
        self._t = np.linspace(0, CONST_TIME_SIMULATING, CONST_N_LINSPACE)

        self.resize(640, 480)

        self._layout = QVBoxLayout(self)
        self._tab_view = QTabWidget()
        self._tab_view.setUsesScrollButtons(False)

        # adding tabs
        self._forced = ParamPicker(
            CONST_FIRST_WIDGET_NAME,
            [CONST_OMEGA, CONST_DELTA, CONST_X0, CONST_DX0, CONST_F],
            ['1', '0.01', '5', '6', '2*cos(3*t+4)'],
            [float, float, float, float, str]
        )
        self._MNKO = ParamPicker(
            CONST_SECOND_WIDGET_NAME,
            [CONST_OMEGA, CONST_DELTA, CONST_X0, CONST_DX0,
             CONST_N, CONST_SIGMA, CONST_PRECISION],
            ['1', '0.01', '5', '6', '50', '0.01', '6'],
            [float, float, float, float, int, float, int]
        )

        self._file_picker = FilePicker()

        self._lin_reg_MNKO = ParamPicker(
            CONST_THIRD_WIDGET_NAME,
            [CONST_SIGMA, CONST_PRECISION],
            ['0.1', '5'],
            [float, int],
            [self._file_picker]
        )
        self._tab_view.insertTab(0, self._forced, CONST_FIRST_WIDGET_NAME)
        self._tab_view.insertTab(1, self._MNKO, CONST_SECOND_WIDGET_NAME)
        self._tab_view.insertTab(
            2, self._lin_reg_MNKO, CONST_THIRD_WIDGET_NAME)

        self._layout.addWidget(self._tab_view)
        self._graph_layout = QHBoxLayout()
        self._graph_left = pg.PlotWidget(background=self._plot_background)
        self._graph_left.getPlotItem().addLine(x=0, pen=self._axis_pen)
        self._graph_left.getPlotItem().addLine(y=0, pen=self._axis_pen)
        self._graph_right = pg.PlotWidget(background=self._plot_background)
        self._graph_right.getPlotItem().addLine(x=0, pen=self._axis_pen)
        self._graph_right.getPlotItem().addLine(y=0, pen=self._axis_pen)
        self._graph_layout.addWidget(self._graph_left)
        self._graph_layout.addWidget(self._graph_right)
        self._layout.addLayout(self._graph_layout)
        self._button = QPushButton("Go Go Go")
        self._layout.addWidget(self._button, 0, Qt.AlignHCenter)

        self._button.clicked.connect(self.__submit_button__)
        self.setWindowTitle('Motions')

    def __submit_button__(self):
        try:
            params = self._tab_view.currentWidget().get_params()
        except Exception as e:
            self._error_box.showMessage(str(e))
            return
        self.__clear_plots__()
        if params['widget_name'] == CONST_FIRST_WIDGET_NAME:
            expr = parse_expr(params[CONST_F])
            f = make_f(lambdify((t_symbol), expr, modules='numpy'),
                       [params[CONST_OMEGA] ** 2, 2 * params[CONST_DELTA]])
            x0 = [params[CONST_X0], params[CONST_DX0]]
            x_res = odeint(f, x0, self._t).T
            self.__update_plot_left__(self._t, x_res[0])
            self.__update_plot_right__(x_res[0], x_res[1])
        elif params['widget_name'] == CONST_SECOND_WIDGET_NAME:
            x0 = [params[CONST_X0], params[CONST_DX0]]
            N = params[CONST_N]
            sigma = params[CONST_SIGMA]
            significant_digits = params[CONST_PRECISION]

            f = make_f(zero, [params[CONST_OMEGA]**2, 2 * params[CONST_DELTA]])
            _x = odeint(f, x0, self._t).T[0]

            _X = []
            _Y = []
            step = int(CONST_N_LINSPACE / (N + 1))
            _vals = []
            for i in range(step, N * step, step):
                _vals.append(_x[i - 1:i + 2])

            for i in _vals:
                x_prev, x_curr, x_next = i
                _X.append([x_curr, -d_f(x_prev, x_next, CONST_h_VALUE)])
            if sigma > 0:
                for i in _vals:
                    x_prev, x_curr, x_next = i
                    _Y.append(d2_f(x_prev, x_curr, x_next,
                                   CONST_h_VALUE) + normal(loc=0.0, scale=sigma))
            else:
                for i in _vals:
                    x_prev, x_curr, x_next = i
                    _Y.append(d2_f(x_prev, x_curr, x_next, CONST_h_VALUE))

            _X = np.array(_X)
            _Y = np.array(_Y)
            round_sigfigs_array(_X, significant_digits)
            round_sigfigs_array(_Y, significant_digits)

            params = None
            for params in lso(_X, _Y):
                pass

            msg = QMessageBox()
            msg.setText("Estimated params " +
                        str(params[0]) + '\nrss=' + str(params[1]))
            msg.exec()
        elif params['widget_name'] == CONST_THIRD_WIDGET_NAME:
            x_file = self._file_picker.get_x_file_name()
            y_file = self._file_picker.get_y_file_name()

            try:
                _X = pd.read_csv(x_file).as_matrix()
            except Exception:
                self._error_box.showMessage('First file does not exist')
                return
            try:
                _Y = pd.read_csv(y_file).as_matrix().flatten()
            except Exception:
                self._error_box.showMessage('Second file does not exist')
                return

            sigma = params[CONST_SIGMA]
            significant_digits = params[CONST_PRECISION]

            _Y_with_noise = _Y
            if sigma > 0:
                noise = normal(loc=0.0, scale=sigma, size=len(_Y))
                _Y_with_noise = _Y + noise



            round_sigfigs_array(_X, significant_digits)
            round_sigfigs_array(_Y_with_noise, significant_digits)

            n = _X.shape[1]

            rss = []
            cp = []
            fpe = []
            s = np.int32(1)
            for _,curr_rss in lso(_X, _Y_with_noise):
                rss.append(curr_rss)
                cp.append(curr_rss + 2 * s)
                fpe.append(curr_rss * (n + s) / (n - s))
                s += 1
            pen_yellow = pg.mkPen(color='#FFFF00', width=2)
            pen_green = pg.mkPen(color='#00FF00', width=2)
            pen_blue = pg.mkPen(color='#0000FF', width=2)

            series = np.arange(1,n+1,1)

            self.__update_plot_left__(series,rss,pen=pen_yellow)
            self.__update_plot_left__(series,cp,pen=pen_green)
            self.__update_plot_left__(series,fpe,pen=pen_blue)


    def __clear_plots__(self):
        self._graph_left.clear()
        self._graph_left.getPlotItem().addLine(x=0, pen=self._axis_pen)
        self._graph_left.getPlotItem().addLine(y=0, pen=self._axis_pen)
        self._graph_right.clear()
        self._graph_right.getPlotItem().addLine(x=0, pen=self._axis_pen)
        self._graph_right.getPlotItem().addLine(y=0, pen=self._axis_pen)

    def __update_plot_left__(self, x, y, pen=None):
        if pen:
            self._graph_left.addItem(pg.PlotDataItem(x, y, pen=pen))
        else:
            self._graph_left.addItem(pg.PlotDataItem(
                x, y, pen=self._default_line_pen))

    def __update_plot_right__(self, x, y, pen=None):
        if pen:
            self._graph_right.addItem(pg.PlotDataItem(x, y, pen=pen))
        else:
            self._graph_right.addItem(pg.PlotDataItem(
                x, y, pen=self._default_line_pen))


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    screen = MainWindow()
    screen.show()
    sys.exit(app.exec_())
