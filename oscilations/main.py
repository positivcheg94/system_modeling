from math import sqrt

from PyQt4.QtCore import *
from PyQt4.QtGui import *
import numpy as np
from scipy.integrate import odeint
from sympy import Symbol
from sympy.parsing.sympy_parser import parse_expr
import pyqtgraph as pg

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

CONST_MNKO_WINDOW = 3
CONST_TIME_SIMULATING = 300
CONST_N_LINSPACE = 1000
CONST_h_VALUE = CONST_TIME_SIMULATING / CONST_N_LINSPACE


def d_f(f_x_prev, f_x_next, h):
    return (f_x_next - f_x_prev) / (2 * h)


def d2_f(f_x_prev, f_x_curr, f_x_next, h):
    return (f_x_next - 2 * f_x_curr + f_x_prev) / (h**2)


def make_f(f, a):
    _a = np.array(a)

    def dec_f(y, t):
        res = np.roll(y, -1)
        res[-1] = f(t) - _a.dot(y)
        return res
    return dec_f


def zero(*args, **kwargs):
    return 0


class ParamPicker(QWidget):

    def __init__(self, name, params, defaults, mapping, parent=None):
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
        self._line_pen = pg.mkPen(color='#0C27D9', width=2)
        self._plot_background = pg.mkBrush(color='#bfbfbf')
        self._error_box = QErrorMessage()
        self._t = np.linspace(0, CONST_TIME_SIMULATING, CONST_N_LINSPACE)

        self.resize(640, 480)

        self._layout = QVBoxLayout(self)
        self._tab_view = QTabWidget()
        self._tab_view.setUsesScrollButtons(False)

        # adding tabs
        self._forced = ParamPicker(
            'Forced',
            [CONST_OMEGA, CONST_DELTA, CONST_X0, CONST_DX0, CONST_F],
            ['1', '0.01', '5', '6', '2*cos(3*t+4)'],
            [float, float, float, float, str]
        )
        self._MNKO = ParamPicker(
            'MNKO',
            [CONST_OMEGA, CONST_DELTA, CONST_X0, CONST_DX0,
                CONST_N, CONST_SIGMA, CONST_PRECISION],
            ['1', '0.01', '5', '6', '50', '0.01-1', '1-5'],
            [float, float, float, float, int, str, str]
        )
        self._tab_view.insertTab(0, self._forced, 'Forced')
        self._tab_view.insertTab(1, self._MNKO, 'MNKO')

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
        params = None
        try:
            params = self._tab_view.currentWidget().get_params()
        except Exception as e:
            self._error_box.showMessage(str(e))
            return

        if(params['widget_name'] == 'Forced'):
            expr = parse_expr(params[CONST_F])
            f = make_f(lambda arg: expr.evalf(subs={t_symbol: arg}), [
                       params[CONST_OMEGA]**2, 2 * params[CONST_DELTA]])
            x0 = [params[CONST_X0], params[CONST_DX0]]
            x_res = odeint(f, x0, self._t).T
            self.__update_plot__(self._t, x_res[0], 'left')
            self.__update_plot__(x_res[0], x_res[1], 'right')
        elif(params['widget_name'] == 'MNKO'):
            x0 = [params[CONST_X0], params[CONST_DX0]]
            N = params[CONST_N]

            f = make_f(zero, [params[CONST_OMEGA]**2, 2 * params[CONST_DELTA]])

            x_res = odeint(f, x0, self._t).T
            _x = x_res[0]


            step = int(CONST_N_LINSPACE / (params[CONST_N] + 1))

            _X = []
            _Y = []
            for i in range(step, CONST_N_LINSPACE, step):
                x_prev, x_curr, x_next = _x[i - 1:i + 2]
                _X.append([-x_curr, -d_f(x_prev, x_next, CONST_h_VALUE)])
                _Y.append(d2_f(x_prev, x_curr, x_next, CONST_h_VALUE))

            _X = np.array(_X)
            _Y = np.array(_Y)

            params = list(lso(_X,_Y))
            pp = params[-1]
            pp[0] = sqrt(pp[0])
            pp[1] = pp[1]/2
            print(pp[0])
            print(pp[1])


    def __update_plot__(self, x, y, plot='left'):
        if plot == 'left':
            plt = self._graph_left
        else:
            plt = self._graph_right
        plt.clear()
        plt.getPlotItem().addLine(x=0, pen=self._axis_pen)
        plt.getPlotItem().addLine(y=0, pen=self._axis_pen)
        plt.addItem(pg.PlotDataItem(x, y, pen=self._line_pen))


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    screen = MainWindow()
    screen.show()
    sys.exit(app.exec_())
