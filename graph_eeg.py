from scipy import signal

import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_pandas_data_set(selected_data_set):
    """
    Passes in the index of the data set we want to use
    The list is statically built based on the files we have in ./resurse/
    """

    csv_sets = [
        "Tibi1_rest1_[2017.06.14-11.25.26].csv",
        "tibi1_rest1_cpy.csv",
        "TUI04_2back_[2017.06.14-11.11.18].csv",
        "TUI4_VTE_1Random_[2017.06.14-11.15.03].csv",
        "TUI4_VTE_Boxes_[2017.06.14-11.20.56].csv",
        "TUI4_VTE_Complex_[2017.06.14-11.22.59].csv",
        "TUI4_VTE_Rest_[2017.06.14-11.07.20].csv",
        "TUI5_3Back-[2017.05.24-15.48.50].csv",
        "TUI5_3Back-modded.csv"
    ]

    csv_file = (
        "resurse/"
        "{0}".format(csv_sets[selected_data_set])
    )

    df = pd.read_csv(
        csv_file,
        delimiter=';'
    )

    return df


def plot_power_spectrum_periodogram(data):
    """
    Create a power spectrum using the periodogram method
    When plotting, we are limiting the printed frequency to 256 for clarity
    The selected frequency is half of the sampling rate
    """

    f, pxx = signal.periodogram(
        data,
        fs=512.0,
        window='flattop',
        scaling='spectrum'
        )

    plt.semilogy(f, pxx)
    plt.xlim(0, 256)
    plt.show()


def plot_data(x_data, y_data, color):
    """
    A simple method to quickly plot basic data
    It helps with not forgetting to plot() and show()
    """

    plt.plot(x_data, y_data, color)
    plt.show()


data_set = get_pandas_data_set(3)

FP1_data = data_set["FP1"]

plot_power_spectrum_periodogram(FP1_data)
