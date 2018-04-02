import csv
import matplotlib.pyplot as plt
import pandas as pd

from numpy import genfromtxt
from numpy import set_printoptions


def plot_data(x_data, y_data, color):
    plt.plot(x_data, y_data, color)
    plt.show()

csv_headers = [
    "Time", "FP1", "F7",
    "F3", "C3", "P3", "P7",
    "O1", "O2", "FP2", "F8",
    "F4", "C4", "P4", "P8",
    "Pz", "T8", "Sampling Rate"
]

csv_file = """\
/Users/tudoramariei\
/Work/dev/Python/\
disertatie/ex_01/\
res_disertatie/\
Tibi1_rest1_[2017.06.14-11.25.26].csv\
"""

data = pd.read_csv(
    csv_file, delimiter=';'
    )

# plot_data(data["Time"], data["FP1"], 'r')
