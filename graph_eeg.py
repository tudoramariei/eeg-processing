from copy import copy as cpy
from math import log
from scipy.signal import butter, cheby1, freqz, lfilter, periodogram

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def get_sampling_rate():
    return sampling_rate


def get_pandas_data_set(selected_data_set, first_row, phase=0):
    """
    Passes in the index of the data set we want to use
    The list is built from the files we have in ./resurse/
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
    print(_csv_sets[selected_data_set])
    _csv_file = (
        "resurse/"
        "{0}".format(_csv_sets[selected_data_set])
    )

    _delimiter = ','
    sampling_rate = 128
    if ';' in open(_csv_file).read():
        _delimiter = ';'
        sampling_rate = 512

    # skip first 10 seconds from file
    _dtf = pd.read_csv(
        _csv_file,
        delimiter=_delimiter,
        skiprows=range(first_row + phase, sampling_rate * 10),
        nrows=1000
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

    # _dtf = _dtf[['FP1', 'FP2', 'F7', 'F8', 'C3', 'C4', 'P3', 'P4',
    # 'P7', 'P8', 'O1', 'O2', 'Pz', 'T8']]
    # _dtf = _dtf[['FP1', 'F7','P3', 'P7', 'FP2','P4', 'F8', 'P8']]
    _dtf = _dtf[['FP1', 'F7', 'F3', 'FP2', 'F8', 'F4']]
    print(get_headers(_dtf))

    return _dtf, _time


def get_bandpass_filter(df_filter, low, high, order):
    if df_filter is 'cheby1':
        b, a = cheby1(
            rp=5,
            N=order,
            Wn=[low, high],
            btype='bandpass'
        )
    elif df_filter is 'butter':
        b, a = butter(
            N=order,
            Wn=[low, high],
            btype='bandpass'
        )

    return b, a


def bandpass_freqz(df_filter, data_vector, low_freq, high_freq):
    _order = 4
    _nyq = 0.5 * get_sampling_rate()
    _low = low_freq/_nyq
    _high = high_freq/_nyq

    b, a = get_bandpass_filter(df_filter, _low, _high, _order)
    w, h = freqz(b, a, worN=len(data_vector))

    data_w = (data_vector * 0.5 / np.pi) * w
    data_h = abs(h)

    plt.plot(data_w, data_h)
    plt.show()


def bandpass_filter(df_filter, data_vector, low_freq, high_freq):
    """
    Filters the signal with the desired bandpass df_filter
    """

    # fft1 = np.fft.fft(data_vector)
    # cfft = fft1 * np.conjugate(fft1)

    _order = 4
    _nyq = 0.5 * get_sampling_rate()
    _low = low_freq/_nyq
    _high = high_freq/_nyq

    b, a = get_bandpass_filter(df_filter, _low, _high, _order)

    y = lfilter(b, a, data_vector)

    return y


def get_filtered_df(df_filter, dtf, _lowcut_freq, _highcut_freq):
    """
    Filters the dataframe with bandpass df_filter
    """

    dtf_aux = dtf.copy()
    # dtf_aux.iloc[:, :]

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

    _delta_lf = 0.5
    _delta_hf = 4.0

    _theta_lf = 4.0
    _theta_hf = 7.0

    _alpha_lf = 7.0
    _alpha_hf = 12.0

    _beta_lf = 12.0
    _beta_hf = 30.0

    _delta1 = get_filtered_df(df_filter, dtf, _delta_lf, _delta_hf)
    _theta1 = get_filtered_df(df_filter, dtf, _theta_lf, _theta_hf)
    _alpha1 = get_filtered_df(df_filter, dtf, _alpha_lf, _alpha_hf)
    _beta1 = get_filtered_df(df_filter, dtf, _beta_lf, _beta_hf)

    return _delta1, _theta1, _alpha1, _beta1


def subplot_freq_bands(
    f, delta, theta, alpha, beta, text, mode=True
):
    if mode is True:
        fig, axes = plt.subplots(nrows=4, ncols=6)
    else:
        fig, axes = plt.subplots(2, 2)

    low_limit = 0.5
    high_limit = 30

    if mode is True:
        for i in range(0, 6):
            axes[0, i].plot(f, delta[delta.columns[i]], 'r')
            axes[0, i].set_xlim(low_limit, high_limit)
            bd = delta.columns[i]
            axes[0, i].set_title(("delta [0.5, 4] {0}-{1}").format(text, bd))
    else:
        axes[0, 0].plot(f, delta, 'r')
        axes[0, 0].set_xlim(low_limit, high_limit)
        axes[0, 0].set_title(("delta [0.5, 4] {}").format(text))

    # if mode is True:
    # else:
    #     delta.plot(ax=axes[0, 0])

    if mode is True:
        for i in range(0, 6):
            axes[1, i].plot(f, theta[theta.columns[i]], 'r')
            axes[1, i].set_xlim(low_limit, high_limit)
            bd = theta.columns[i]
            axes[1, i].set_title(("theta [4, 7] {0}-{1}").format(text, bd))
    else:
        axes[0, 1].plot(f, theta, 'r')
        axes[0, 1].set_xlim(low_limit, high_limit)
        axes[0, 1].set_title(("theta [4, 7] {}").format(text))

    # if mode is True:
    #     axes[0, 1].plot(f, theta, 'g')
    # else:
    #     theta.plot(ax=axes[0, 1])

    if mode is True:
        for i in range(0, 6):
            axes[2, i].plot(f, alpha[alpha.columns[i]], 'r')
            axes[2, i].set_xlim(low_limit, high_limit)
            bd = alpha.columns[i]
            axes[2, i].set_title(("alpha [7, 12] {0}-{1}").format(text, bd))
    else:
        axes[1, 0].plot(f, alpha, 'r')
        axes[1, 0].set_xlim(low_limit, high_limit)
        axes[1, 0].set_title(("alpha [7, 12] {}").format(text))

    # if mode is True:
    #     axes[1, 0].plot(f, alpha, 'b')
    # else:
    #     alpha.plot(ax=axes[1, 0])

    if mode is True:
        for i in range(0, 6):
            axes[3, i].plot(f, beta[beta.columns[i]], 'r')
            axes[3, i].set_xlim(low_limit, high_limit)
            bd = beta.columns[i]
            axes[3, i].set_title(("beta [12, 30] {0}-{1}").format(text, bd))
    else:
        axes[1, 1].plot(f, beta, 'r')
        axes[1, 1].set_xlim(low_limit, high_limit)
        axes[1, 1].set_title(("beta [12, 30] {}").format(text))

    # if mode is True:
    #     axes[1, 1].plot(f, beta, 'y')
    # else:
    #     beta.plot(ax=axes[1, 1])

    fig.subplots_adjust(hspace=0.4)
    fig.subplots_adjust(wspace=0.2)


def get_power_spectrum(data_vector):
    """
    Create a power spectrum using the periodogram method
    """

    f, pxx = periodogram(
        data_vector,
        fs=get_sampling_rate(),
        window='boxcar',
        scaling='spectrum'
    )

    return f, pxx


def get_df_power_spectrum(dtf):
    _headers = get_headers(dtf)

    plt.figure(1)
    df_pxx = pd.DataFrame(columns=_headers)

    for selected_header in _headers:
        f, df_pxx[selected_header] = get_power_spectrum(dtf[selected_header])

        df_pxx[selected_header] = df_pxx[selected_header] * (10 ** 6)
        # plt.semilogy(f, df_pxx[selected_header])

    return f, df_pxx


def get_frontal_assymetry(dtf):
    band_l = (dtf['FP1'] + dtf['F7'] + dtf['F3']) / 3
    band_r = (dtf['FP2'] + dtf['F8'] + dtf['F4']) / 3
    band_math = abs(band_l - band_r)/(band_l + band_r)
    band_assym = [log(y) for y in band_math]

    # abs_assym = abs(element) for element in band_assym
    return np.abs(band_assym)


def get_assym(first_row, phase=0):
    df, _ = get_pandas_data_set(3, first_row, phase)
    # df.plot(title="Original Dataset")

    df_filt = 'cheby1'
    f_df = get_filtered_df(df_filt, df, 0.5, 100)

    # f_df.plot(title="{} filtering".format(df_filt))

    fb_filt = 'cheby1'
    df_delta1, df_theta1, df_alpha1, df_beta1 = get_frequency_bands(
        fb_filt,
        f_df)
    # subplot_freq_bands(
    #     time_df,
    #     df_delta1, df_theta1,
    #     df_alpha1, df_beta1,
    #     "freq band", True)

    f, pf_delta1 = get_df_power_spectrum(df_delta1)
    _, pf_theta1 = get_df_power_spectrum(df_theta1)
    _, pf_alpha1 = get_df_power_spectrum(df_alpha1)
    _, pf_beta1 = get_df_power_spectrum(df_beta1)

    # subplot_freq_bands(
    #     f,
    #     pf_delta1, pf_theta1,
    #     pf_alpha1, pf_beta1,
    #     "ps", True)

    delta_assym = get_frontal_assymetry(pf_delta1)
    theta_assym = get_frontal_assymetry(pf_theta1)
    alpha_assym = get_frontal_assymetry(pf_alpha1)
    beta_assym = get_frontal_assymetry(pf_beta1)

    subplot_freq_bands(
        f,
        delta_assym, theta_assym,
        alpha_assym, beta_assym,
        "asym", False)


_first_row = 1
get_assym(_first_row, 0)
get_assym(_first_row, 1000)
get_assym(_first_row, 2000)
get_assym(_first_row, 3000)

plt.show()
