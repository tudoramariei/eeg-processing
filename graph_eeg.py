from copy import copy as cpy
from matplotlib import pyplot as plt
from math import log as ln
from scipy import signal as sig

import numpy as np
import pandas as pd

_delta_lf = 0.5
_delta_hf = 4.0

_theta_lf = 4.0
_theta_hf = 7.0

_alpha_lf = 7.0
_alpha_hf = 12.0

_beta_lf = 12.0
_beta_hf = 30.0


def get_headers(dtf):
    """
    Passes a dataframe and returns the _headers for it
    """

    return dtf.columns.values


def get_column(dtf, column_name):
    """
    Get a column from a dataframe if it exists
    """

    if column_name in dtf.columns:
        return dtf[column_name]
    return None


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


def get_s_rate():
    return s_rate


def get_pandas_data_set(
    sel_user,
    sel_game,
    wf,
    start_row,
    num_rows=None,
    skipped_seconds=0
):
    """
    Passes in the index of the data set we want to use
    The list is built from the files we have in ./resurse/
    """

    global s_rate
    _csv_file = wf

    _delimiter = ','
    s_rate = 128
    if ';' in open(_csv_file).read():
        _delimiter = ';'
        s_rate = 512

    # skip first 10 seconds from file
    _dtf = pd.read_csv(
        _csv_file,
        delimiter=_delimiter,
        skiprows=range(1, s_rate * skipped_seconds + start_row),
        nrows=num_rows
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

    _dtf = _dtf[['FP1', 'F7', 'F3', 'FP2', 'F8', 'F4']]

    return _dtf, _time


def get_bandpass_filter(df_filter, low, high, order):
    if df_filter is 'cheby1':
        b, a = sig.cheby1(
            rp=5,
            N=order,
            Wn=[low, high],
            btype='bandpass'
        )
    elif df_filter is 'butter':
        b, a = sig.butter(
            N=order,
            Wn=[low, high],
            btype='bandpass'
        )

    return b, a


def bandpass_filter(df_filter, data_vector, low_freq, high_freq):
    """
    Filters the signal with the desired bandpass df_filter
    """

    _order = 4
    _nyq = 0.5 * get_s_rate()
    _low = low_freq/_nyq
    _high = high_freq/_nyq

    b, a = get_bandpass_filter(df_filter, _low, _high, _order)

    y = sig.lfilter(b, a, data_vector)

    return y


def get_filtered_df(df_filter, dtf, _lowcut_freq, _highcut_freq):
    """
    Filters the dataframe with bandpass df_filter
    """

    dtf_aux = dtf.copy()

    for column in dtf.columns.values:
        df_column = cpy(dtf[column])
        dtf_aux[column] = bandpass_filter(
            df_filter,
            df_column,
            _lowcut_freq,
            _highcut_freq
        )

    return dtf_aux


def get_frequency_bands(df_filter, dtf):
    """
    Returns the four frequency bands
        delta [0.5, 4]
        theta [4, 7]
        alpha [7, 12]
        beta [12, 30]
    """

    _delta1 = get_filtered_df(df_filter, dtf, _delta_lf, _delta_hf)
    _theta1 = get_filtered_df(df_filter, dtf, _theta_lf, _theta_hf)
    _alpha1 = get_filtered_df(df_filter, dtf, _alpha_lf, _alpha_hf)
    _beta1 = get_filtered_df(df_filter, dtf, _beta_lf, _beta_hf)

    return _delta1, _theta1, _alpha1, _beta1


def subplot_freq_bands(
    freq, delta, theta, alpha, beta, text, mode, fig_num, save_fig
):
    if mode is True:
        fig, axes = plt.subplots(nrows=4, ncols=6)
    else:
        fig, axes = plt.subplots(2, 2)

    axes[0, 0].plot(freq, delta, 'r')
    axes[0, 0].set_xlim(_delta_lf, _delta_hf)
    axes[0, 0].set_title(("delta [0.5, 4] {}").format(text))

    axes[0, 1].plot(freq, theta, 'r')
    axes[0, 1].set_xlim(_theta_lf, _theta_hf)
    axes[0, 1].set_title(("theta [4, 7] {}").format(text))

    axes[1, 0].plot(freq, alpha, 'r')
    axes[1, 0].set_xlim(_alpha_lf, _alpha_hf)
    axes[1, 0].set_title(("alpha [7, 12] {}").format(text))

    axes[1, 1].plot(freq, beta, 'r')
    axes[1, 1].set_xlim(_beta_lf, _beta_hf)
    axes[1, 1].set_title(("beta [12, 30] {}").format(text))

    fig.subplots_adjust(hspace=0.4)
    fig.subplots_adjust(wspace=0.2)

    if save_fig:
        plt.savefig("subplots/subplot{0}.png".format(format(fig_num, '0>3')))
        plt.close()


def get_power_spectrum(data_vector):
    """
    Create a power spectrum for a single channel using the periodogram method
    """

    return sig.periodogram(
        data_vector,
        fs=get_s_rate(),
        window='boxcar',
        scaling='spectrum'
    )


def get_df_power_spectrum(dtf):
    """
    Get the power spectrum for the whole dataframe
    """

    _headers = get_headers(dtf)

    df_pxx = pd.DataFrame(columns=_headers)

    for selected_header in _headers:
        f, df_pxx[selected_header] = get_power_spectrum(dtf[selected_header])
        df_pxx[selected_header] = df_pxx[selected_header] * (10 ** 6)

    return f, df_pxx


def get_frontal_assymetry(dtf):
    """
    Calculates the frontal assymetry of the dataset
    """

    band_l = (dtf['FP1'] + dtf['F7'] + dtf['F3']) / 3
    band_r = (dtf['FP2'] + dtf['F8'] + dtf['F4']) / 3
    band_math = abs(band_l - band_r)/(band_l + band_r)
    band_assym = [ln(y) for y in band_math]

    return np.abs(band_assym)


def get_assym(work_file, st_row, num_rows=None):
    """
    Processes the work_file to get a number of num_row
    starting with row 'st_row'
    """

    df, _ = get_pandas_data_set(
        sel_user=1, sel_game='A',
        wf=work_file, start_row=st_row, num_rows=num_rows
    )

    df_filt = 'cheby1'
    f_df = get_filtered_df(df_filt, df, 0.5, 100)

    fb_filt = 'cheby1'
    df_delta1, df_theta1, df_alpha1, df_beta1 = get_frequency_bands(
        fb_filt,
        f_df)

    f, pf_delta1 = get_df_power_spectrum(df_delta1)
    _, pf_theta1 = get_df_power_spectrum(df_theta1)
    _, pf_alpha1 = get_df_power_spectrum(df_alpha1)
    _, pf_beta1 = get_df_power_spectrum(df_beta1)

    delta_assym = get_frontal_assymetry(pf_delta1)
    theta_assym = get_frontal_assymetry(pf_theta1)
    alpha_assym = get_frontal_assymetry(pf_alpha1)
    beta_assym = get_frontal_assymetry(pf_beta1)

    return f, delta_assym, theta_assym, alpha_assym, beta_assym


def get_selected_file_name(
    sel_user, sel_game
):
    """
    Returns the name of the file which will be used
    """

    print("Working with file: USER{0}_game_{1}".format(sel_user, sel_game))
    _csv_file = (
        "resurse/"
        "USER{0}_game_{1}.csv".format(sel_user, sel_game)
    )

    return _csv_file


def get_values_between_l_h(
    use_arr, ref_arr,
    low, high
):
    """
    Gets the value from the use_arr
    based on values between low and high from ref_arr
    """

    indices = np.where(np.logical_and(ref_arr > low, ref_arr < high))[0]
    return np.take(use_arr, indices)


_num_rows = 1024
_work_file = get_selected_file_name(sel_user=1, sel_game='A')

delta_max_list = []
theta_max_list = []
alpha_max_list = []
beta_max_list = []

num_sequences = 3

for i in range(0, num_sequences):
    f, d_a, t_a, a_a, b_a = get_assym(_work_file, _num_rows*i, _num_rows)

    subplot_freq_bands(
        freq=f, delta=d_a, theta=t_a, alpha=a_a, beta=b_a,
        text="asym", mode=False, fig_num=i,
        save_fig=True)

    delta_max_list.append(
        np.amax(
            get_values_between_l_h(
                d_a, f, _delta_lf, _delta_hf)))

    theta_max_list.append(
        np.amax(
            get_values_between_l_h(
                t_a, f, _theta_lf, _theta_hf)))

    alpha_max_list.append(
        np.amax(
            get_values_between_l_h(
                a_a, f, _alpha_lf, _alpha_hf)))

    beta_max_list.append(
        np.amax(
            get_values_between_l_h(
                b_a, f, _beta_lf, _beta_hf)))
