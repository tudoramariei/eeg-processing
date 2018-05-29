from math import log
from pyentrp import entropy as ent
from sampen import sampen2 as sp2
from scipy import signal as sig

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
        _df_delta1, _df_theta1,
        _df_alpha1, _df_beta1,
        _channel='F3'):
    """
    Mostly debugging purposes, it prints the four frequency bands
    """

    # plt.figure(1)
    ef, axarr = plt.subplots(2, 2)
    axarr[0, 0].plot(_time_df, _df_theta1[_channel], 'g')
    # axarr[0, 0].set_xlim(_delta_lf, _delta_hf)
    axarr[0, 0].set_xlim(0.5, 30)

    axarr[0, 1].plot(_time_df, _df_delta1[_channel], 'r')
    # axarr[0, 1].set_xlim(_theta_lf, _theta_hf)
    axarr[0, 1].set_xlim(0.5, 30)

    axarr[1, 0].plot(_time_df, _df_alpha1[_channel], 'b')
    # axarr[1, 0].set_xlim(_alpha_lf, _alpha_hf)
    axarr[1, 0].set_xlim(0.5, 30)

    axarr[1, 1].plot(_time_df, _df_beta1[_channel], 'y')
    # axarr[1, 1].set_xlim(_beta_lf, _beta_hf)
    axarr[1, 1].set_xlim(0.5, 30)
    ef.subplots_adjust(hspace=0.3)

    # plt.figure(2)
    # plt.plot(_time_df, _df_beta1[_channel], 'y')
    # plt.plot(_time_df, _df_alpha1[_channel], 'b')
    # plt.plot(_time_df, _df_delta1[_channel], 'g')
    # plt.plot(_time_df, _df_theta1[_channel], 'r')

    # plt.figure(3)
    # plt.plot(_time_df, _df_alpha1[_channel], 'b')
    # plt.plot(_time_df, _df_delta1[_channel], 'g')
    # plt.plot(_time_df, _df_theta1[_channel], 'r')

    # plt.figure(4)
    # plt.plot(_time_df, _df_delta1[_channel], 'g')
    # plt.plot(_time_df, _df_theta1[_channel], 'r')

    # plt.figure(5)
    # plt.plot(_time_df, _df_theta1[_channel], 'r')
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
    if _delimiter not in open(_csv_file).read():
        sampling_rate = 512
        _delimiter = ';'

    # skip first ~10 seconds from file
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

    _headers_used = 'front'

    if _headers_used is 'full':
        _dtf = _dtf[[
            'FP1', 'FP2', 'F7', 'F8',
            'C3', 'C4', 'P3', 'P4',
            'P7', 'P8', 'O1', 'O2',
            'Pz', 'T8']]
    elif _headers_used is 'front':
        _dtf = _dtf[['FP1', 'F7', 'F3', 'FP2', 'F8', 'F4']]

    print(get_headers(_dtf))

    return _dtf, _time


def get_power_spectrum(data_vector):
    """
    Create a power spectrum using the periodogram method
    """

    f, pxx = sig.periodogram(
        data_vector,
        fs=get_sampling_rate(),
        window='boxcar',
        scaling='spectrum'
    )

    return f, pxx


def get_df_power_spectrum(dtf):
    """
    Returns the power spectrum of a whole dataframe
    Goes through each header of a dtf and gets the power spectrum for it
    """
    _headers = get_headers(dtf)

    df_pxx = pd.DataFrame(columns=_headers)

    print("get_df_power_spectrum 1")

    for selected_header in _headers:
        f, df_pxx[selected_header] = get_power_spectrum(dtf[selected_header])

        df_pxx[selected_header] = df_pxx[selected_header] * (10 ** 6)

    print("get_df_power_spectrum 2")

    return f, df_pxx


def bandpass_filter(filter, data_vector, low_freq, high_freq):
    """
    Filters the signal with the desired bandpass filter
    """

    _order = 4
    _nyq = 0.5 * get_sampling_rate()
    _low = low_freq/_nyq
    _high = high_freq/_nyq
    _btype = 'bandpass'

    if filter is 'cheby1':
        b, a = sig.cheby1(
            rp=5,
            N=_order,
            Wn=[_low, _high],
            btype=_btype
        )
    elif filter is 'butter':
        b, a = sig.butter(
            N=_order,
            Wn=[_low, _high],
            btype=_btype
        )

    y = sig.lfilter(b, a, data_vector)
    plot_data(range(0, y.size), y)

    return y


def get_filtered_df(filter, dtf, low, high):
    """
    Filters the dataframe with bandpass filter
    """

    _lowcut_freq = low
    _highcut_freq = high

    _headers = get_headers(dtf)
    _data = pd.DataFrame(columns=_headers)

    print("get_filtered_df 1")

    for header in _headers:
        df_column = dtf[header]
        _data[header] = bandpass_filter(
            filter,
            df_column,
            _lowcut_freq,
            _highcut_freq
        )

    print("get_filtered_df 2")
    return _data


def get_frequency_bands(dtf, time_df):
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

    print("get_frequency_bands 1")
    filter = 'butter'
    _delta1 = get_filtered_df(filter, dtf, _delta_lf, _delta_hf)
    _theta1 = get_filtered_df(filter, dtf, _theta_lf, _theta_hf)
    _alpha1 = get_filtered_df(filter, dtf, _alpha_lf, _alpha_hf)
    _beta1 = get_filtered_df(filter, dtf, _beta_lf, _beta_hf)

    print("get_frequency_bands 2")
    plot_freq_bands(time_df, _delta1, _theta1, _alpha1, _beta1)

    # _delta1.plot()
    # _theta1.plot()
    # _alpha1.plot()
    # _beta1.plot()
    # plt.show()

    return _delta1, _theta1, _alpha1, _beta1


def get_frontal_assymetry(dtf):
    """
    Returns the frontal assymetry for a dataframe
    The power difference between left and right hemispheres
    is divided by the total power of both hemispheres
    """
    band_l = (dtf['FP1'] + dtf['F7'] + dtf['F3']) / 3
    band_r = (dtf['FP2'] + dtf['F8'] + dtf['F4']) / 3
    band_math = abs(band_l - band_r)/(band_l + band_r)
    band_assym = [log(y) for y in band_math]

    return band_assym

df, time_df = get_pandas_data_set(3)
df = get_filtered_df('cheby1', df, 0.5, 35)

print("main 1")

df_delta1, df_theta1, df_alpha1, df_beta1 = get_frequency_bands(df, time_df)

print("main 2")

f, pf_delta1 = get_df_power_spectrum(df_delta1)
_, pf_theta1 = get_df_power_spectrum(df_theta1)
_, pf_alpha1 = get_df_power_spectrum(df_alpha1)
_, pf_beta1 = get_df_power_spectrum(df_beta1)

print("main 3")

delta_assym = get_frontal_assymetry(pf_delta1)
theta_assym = get_frontal_assymetry(pf_theta1)
alpha_assym = get_frontal_assymetry(pf_alpha1)
beta_assym = get_frontal_assymetry(pf_beta1)

print("main 4")
print(delta_assym)
samp_en = ent.sample_entropy(delta_assym, 5)

# samp_en = sp2(delta_assym)

print("main 5")

print(samp_en)

# _delta_lf = 0.5
# _delta_hf = 4

# _theta_lf = 4
# _theta_hf = 7

# _alpha_lf = 7
# _alpha_hf = 12

# _beta_lf = 12
# _beta_hf = 30

# ef, axarr = plt.subplots(2, 2)
# axarr[0, 0].plot(f, delta_assym, 'g')
# # axarr[0, 0].set_xlim(_delta_lf, _delta_hf)
# axarr[0, 0].set_xlim(0.5, 30)

# axarr[0, 1].plot(f, theta_assym, 'r')
# # axarr[0, 1].set_xlim(_theta_lf, _theta_hf)
# axarr[0, 1].set_xlim(0.5, 30)

# axarr[1, 0].plot(f, alpha_assym, 'b')
# # axarr[1, 0].set_xlim(_alpha_lf, _alpha_hf)
# axarr[1, 0].set_xlim(0.5, 30)

# axarr[1, 1].plot(f, beta_assym, 'y')
# # axarr[1, 1].set_xlim(_beta_lf, _beta_hf)
# axarr[1, 1].set_xlim(0.5, 30)

# plt.show()
