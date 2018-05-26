from scipy import signal
from math import log

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sampling_rate = 0

def plot_data(x_data, y_data, color='r'):
    """
    A simple method to quickly plot basic data
    It helps with not forgetting to plot() and show()
    """

    plt.plot(x_data, y_data, color)
    plt.show()


def plot_freq_bands(
    _time_df,
    _df_beta1, _df_alpha1,
    _df_delta1, _df_theta1,
    _channel):
    """
    Mostly debugging purposes, it prints the four frequency bands
    """

    plt.figure(1)
    plt.plot(_time_df, _df_beta1[_channel], 'y')
    plt.plot(_time_df, _df_alpha1[_channel], 'b')
    plt.plot(_time_df, _df_delta1[_channel], 'g')
    plt.plot(_time_df, _df_theta1[_channel], 'r')

    plt.figure(2)
    plt.plot(_time_df, _df_alpha1[_channel], 'b')
    plt.plot(_time_df, _df_delta1[_channel], 'g')
    plt.plot(_time_df, _df_theta1[_channel], 'r')

    plt.figure(3)
    plt.plot(_time_df, _df_delta1[_channel], 'g')
    plt.plot(_time_df, _df_theta1[_channel], 'r')

    plt.figure(4)
    plt.plot(_time_df, _df_theta1[_channel], 'r')
    plt.show()


def get_sampling_rate():
    return sampling_rate


def drop_column(dtf, column_name):
    """
    Drop a column from a dataframe if it exists
    """

    if column_name in dtf.columns:
        dtf = dtf.drop(columns=[column_name])
    return dtf


def drop_zero_columns(dtf):
    """
    Drop all columns from a dataframe if they are composed of only zeros
    """

    return dtf.loc[:, (dtf != 0).any(axis=0)]


def get_column(dtf, column_name):
    """
    Get a column from a dataframe if it exists
    """

    if column_name in dtf.columns:
        return dtf[column_name]
    return None


def get_headers(dtf):
    """
    Passes a dataframe and returns the _headers for it
    """

    return dtf.columns.values


def data_normalization(dtf):
    """
    Bring the dataframe values in the range [0, 1]
    Needed in order for the date to work in a butterworth filter
    """

    return dtf.apply(lambda x: (x - x.min()) / (x.max() - x.min()))


def get_pandas_data_set(selected_data_set):
    """
    Passes in the index of the data set we want to use
    The list is statically built based on the files we have in ./resurse/
    """

    global sampling_rate

    _csv_sets = [
        "Tibi1_rest1_[2017.06.14-11.25.26].csv",
        "TUI04_2back_[2017.06.14-11.11.18].csv",
        "TUI4_VTE_1Random_[2017.06.14-11.15.03].csv",
        "TUI4_VTE_Boxes_[2017.06.14-11.20.56].csv",
        "TUI4_VTE_Complex_[2017.06.14-11.22.59].csv",
        "TUI4_VTE_Rest_[2017.06.14-11.07.20].csv",
        "TUI5_3Back-[2017.05.24-15.48.50].csv",
        "Irina-1-14.04.18.22.08.52.csv",
        "Irina-2-14.04.18.22.12.13.csv",
        "Irina-2back-14.04.18.22.19.03.csv",
        "Irina-3back-14.04.18.22.21.01.csv"
    ]

    _csv_file = (
        "resurse/"
        "{0}".format(_csv_sets[selected_data_set])
    )

    _delimiter = ','
    sampling_rate = 128
    if ';' in open(_csv_file).read():
        sampling_rate = 512
        _delimiter = ';'

    # skip first 10 seconds from file
    _dtf = pd.read_csv(
        _csv_file,
        delimiter=_delimiter,
        skiprows=range(1, 5120)
    )

    if ',' is _delimiter:
        # if the data is from the epoc
        _time = get_column(_dtf, 'TIME_STAMP_ms')

        _dtf = drop_zero_columns(_dtf)
        _dtf = drop_column(_dtf, 'TIME_STAMP_ms')
        _dtf = drop_column(_dtf, 'TIME_STAMP_s')
        _dtf = drop_column(_dtf, 'COUNTER')
    else:
        # if data is from the test data set
        _time = get_column(_dtf, 'Time (s)')

        _dtf = drop_column(_dtf, 'Time (s)')
        _dtf = drop_column(_dtf, 'Sampling Rate')
        _dtf = drop_column(_dtf, 'Reference')

    print(get_headers(_dtf))
    # _dtf = _dtf[['FP1', 'FP2', 'F7', 'F8', 'C3', 'C4', 'P3', 'P4', 'P7', 'P8', 'O1', 'O2', 'Pz', 'T8']]
    # _dtf = _dtf[['FP1', 'F7','P3', 'P7', 'FP2','P4', 'F8', 'P8']]
    _dtf = _dtf[['FP1', 'F7', 'F3', 'FP2', 'F8', 'F4']]

    return _dtf, _time


def get_power_spectrum(data_vector):
    """
    Create a power spectrum using the periodogram method
    """

    f, pxx = signal.periodogram(
        data_vector,
        fs=get_sampling_rate(),
        window='boxcar',
        scaling='spectrum'
    )

    return f, pxx


# def get_sample_freq_for_ps(dtf):
#     f, _ = get_power_spectrum(dtf[1])
#     return f


def get_df_power_spectrum(dtf):
    _headers = get_headers(dtf)

    plt.figure(1)
    df_pxx = pd.DataFrame(columns=_headers)

    for selected_header in _headers:
        f, df_pxx[selected_header] = get_power_spectrum(dtf[selected_header])

        df_pxx[selected_header] = df_pxx[selected_header] * (10 ** 6)
        # plt.semilogy(f, df_pxx[selected_header])

    return f, df_pxx


def bandpass_filter(filter, data_vector, low_freq, high_freq):
    """
    Filters the signal with the desired bandpass filter
    """

    _order = 4
    _nyq = 0.5 * get_sampling_rate()
    _low = low_freq/_nyq
    _high = high_freq/_nyq

    if filter is 'cheby1':
        b, a = signal.cheby1(
            rp=5,
            N=_order,
            Wn=[_low, _high],
            btype='bandpass'
        )
    elif filter is 'butter':
        b, a = signal.butter(
            N=_order,
            Wn=[_low, _high],
            btype='bandpass'
        )

    y = signal.lfilter(b, a, data_vector)

    return y


def get_filtered_df(filter, dtf, low, high):
    """
    Filters the dataframe with bandpass filter
    """

    _lowcut_freq = low
    _highcut_freq = high

    for column in dtf.columns.values:
        df_column = dtf[column]
        dtf[column] = bandpass_filter(
            filter,
            df_column,
            _lowcut_freq,
            _highcut_freq
        )

    return dtf


def get_frequency_bands(dtf):
    """
    Returns the four frequency bands
        delta [0.5, 4]
        theta [4, 7]
        alpha [7, 12]
        beta [12, 30]
    """

    _delta_lf = 0.5
    _delta_hf = 4

    _theta_lf = 4
    _theta_hf = 7

    _alpha_lf = 7
    _alpha_hf = 12

    _beta_lf = 12
    _beta_hf = 30

    filter = 'cheby1'
    _delta1 = get_filtered_df(filter, dtf, _delta_lf, _delta_hf)
    _theta1 = get_filtered_df(filter, dtf, _theta_lf, _theta_hf)
    _alpha1 = get_filtered_df(filter, dtf, _alpha_lf, _alpha_hf)
    _beta1 = get_filtered_df(filter, dtf, _beta_lf, _beta_hf)

    _headers = get_headers(dtf)
    _df_delta1 = pd.DataFrame(data=_delta1, columns=_headers)
    _df_theta1 = pd.DataFrame(data=_theta1, columns=_headers)
    _df_alpha1 = pd.DataFrame(data=_alpha1, columns=_headers)
    _df_beta1 = pd.DataFrame(data=_beta1, columns=_headers)

    return _df_delta1, _df_theta1, _df_alpha1, _df_beta1


def get_frontal_assymetry(dtf):
    band_l = (dtf['FP1'] + dtf['F7'] + dtf['F3']) / 3
    band_r = (dtf['FP2'] + dtf['F8'] + dtf['F4']) / 3
    band_math = abs(band_l - band_r)/(band_l + band_r)
    band_assym = [log(y) for y in band_math]

    return band_assym

df, time_df = get_pandas_data_set(3)
df = get_filtered_df('cheby1', df, 0.5, 100)
df_delta1, df_theta1, df_alpha1, df_beta1 = get_frequency_bands(df)

f, pf_delta1 = get_df_power_spectrum(df_delta1)
_, pf_theta1 = get_df_power_spectrum(df_theta1)
_, pf_alpha1 = get_df_power_spectrum(df_alpha1)
_, pf_beta1 = get_df_power_spectrum(df_beta1)

delta_assym = get_frontal_assymetry(pf_delta1)
theta_assym = get_frontal_assymetry(pf_theta1)
alpha_assym = get_frontal_assymetry(pf_alpha1)
beta_assym = get_frontal_assymetry(pf_beta1)

plt.plot(f, beta_assym)
plt.xlim(0.5, 30)
plt.show()
