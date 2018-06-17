from copy import copy as cpy
from matplotlib import pyplot as plt
from math import log as ln
from math import floor as fl
from scipy import signal as sig

import numpy as np
import pandas as pd
import time

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


def subplot_bands(
    y, delta, theta, alpha, beta,
    ylim_l, ylim_h,
    folder, fig_num,
    save_fig, is_active,
    xlab, ylab
):
    if is_active:
        fig, axes = plt.subplots(2, 2)

        axes[0, 0].plot(y, delta, 'r')
        axes[0, 0].set_xlim(_delta_lf, _delta_hf)
        axes[0, 0].set_ylim(ylim_l, ylim_h)
        axes[0, 0].set_title(("delta [0.5, 4] {}").format(folder))
        axes[0, 0].set(xlabel=xlab, ylabel=ylab)

        axes[0, 1].plot(y, theta, 'r')
        axes[0, 1].set_xlim(_theta_lf, _theta_hf)
        axes[0, 1].set_ylim(ylim_l, ylim_h)
        axes[0, 1].set_title(("theta [4, 7] {}").format(folder))
        axes[0, 1].set(xlabel=xlab, ylabel=ylab)

        axes[1, 0].plot(y, alpha, 'r')
        axes[1, 0].set_xlim(_alpha_lf, _alpha_hf)
        axes[1, 0].set_ylim(ylim_l, ylim_h)
        axes[1, 0].set_title(("alpha [7, 12] {}").format(folder))
        axes[1, 0].set(xlabel=xlab, ylabel=ylab)

        axes[1, 1].plot(y, beta, 'r')
        axes[1, 1].set_xlim(_beta_lf, _beta_hf)
        axes[1, 1].set_ylim(ylim_l, ylim_h)
        axes[1, 1].set_title(("beta [12, 30] {}").format(folder))
        axes[1, 1].set(xlabel=xlab, ylabel=ylab)

        fig.subplots_adjust(hspace=0.4)
        fig.subplots_adjust(wspace=0.2)

        if save_fig is True:
            plt.savefig("subplots/{1}/sub_{1}_{0}.png".format(
                format(fig_num+1, '0>3'),
                folder))

        plt.close(fig)


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

    return band_assym


def ln_arr(arr):
    ln_of_arr = [ln(y) for y in arr]
    return np.array(ln_of_arr)


def get_log_frontal_assymetry(dtf):
    """
    Calculates the frontal assymetry of the dataset
    """

    band_l = (dtf['FP1'] + dtf['F7'] + dtf['F3']) / 3
    band_r = (dtf['FP2'] + dtf['F8'] + dtf['F4']) / 3

    up_math = band_l.apply(np.log) - band_r.apply(np.log)
    dn_math = band_l.apply(np.log) + band_r.apply(np.log)

    math = up_math / dn_math

    return math


def get_assym(dtf, i, num_rows):
    """
    Processes the work_file to get a number of num_row
    starting with row 'st_row'
    """
    st_row = num_rows * i
    _df = dtf.iloc[st_row:(st_row+num_rows)]

    _df_filt = 'cheby1'
    _f_df = get_filtered_df(_df_filt, _df, 0.5, 100)

    _fb_filt = 'cheby1'
    df_delta1, df_theta1, df_alpha1, df_beta1 = get_frequency_bands(
        _fb_filt,
        _f_df)

    f, pf_delta1 = get_df_power_spectrum(df_delta1)
    _, pf_theta1 = get_df_power_spectrum(df_theta1)
    _, pf_alpha1 = get_df_power_spectrum(df_alpha1)
    _, pf_beta1 = get_df_power_spectrum(df_beta1)

    subplot_bands(
        y=f, delta=pf_delta1, theta=pf_theta1,
        alpha=pf_alpha1, beta=pf_beta1,
        ylim_l=None, ylim_h=None,
        folder="ps", fig_num=i,
        is_active=True, save_fig=True,
        xlab='epocs', ylab='index')

    delta_assym = get_log_frontal_assymetry(pf_delta1)
    theta_assym = get_log_frontal_assymetry(pf_theta1)
    alpha_assym = get_log_frontal_assymetry(pf_alpha1)
    beta_assym = get_log_frontal_assymetry(pf_beta1)

    return f, delta_assym, theta_assym, alpha_assym, beta_assym


def get_values_between_l_h(
    use_arr, ref_arr,
    low, high
):
    """
    Gets the value from the use_arr
    based on values between low and high from ref_arr
    """

    indices = np.where(np.logical_and(ref_arr > low, ref_arr < high))[0]
    return np.take(use_arr.values, indices)


def get_selected_file_name(
    sel_user, sel_game
):
    """
    Returns the name of the file which will be used
    """

    if sel_game in ('A', 'B', 'C', 'D', 'E', 'F'):
        print("Working with file: USER{0}_game_{1}".format(sel_user, sel_game))
        _csv_file = (
            "resurse/"
            "USER{0}_game_{1}.csv".format(sel_user, sel_game)
        )
    else:
        print("Working with file: USER{0}_{1}".format(sel_user, sel_game))
        _csv_file = (
            "resurse/"
            "USER{0}_{1}.csv".format(sel_user, sel_game)
        )

    return _csv_file


def get_dataframe(work_file, skip_rows=0):
    """
    Gets the dataframe from the selected work_file
    """

    global s_rate

    _delimiter = ','
    if ';' in open(work_file).read():
        print("Delimiter is ';' so please replace")
        exit(1)

    if skip_rows is 0:
        skip_r = None
    else:
        skip_r = range(1, skip_rows)

    _dtf = pd.read_csv(
        work_file,
        delimiter=_delimiter,
        skiprows=skip_r
    )

    if ',' is _delimiter:
        # if the data is from the epoc
        # _time = get_column(_dtf, 'TIME_STAMP_ms')

        _dtf = drop_zero_columns(_dtf)
        _dtf = drop_column(_dtf, 'TIME_STAMP_ms')
        _dtf = drop_column(_dtf, 'TIME_STAMP_s')
        _dtf = drop_column(_dtf, 'COUNTER')
    else:
        # if data is from the user test data set
        # _time = get_column(_dtf, 'Time (s)')
        # _time = get_column(_dtf, 'Timestamp')

        # _dtf = drop_column(_dtf, 'Time (s)')
        _dtf = drop_column(_dtf, 'Timestamp')
        _dtf = drop_column(_dtf, 'Sampling Rate')
        _dtf = drop_column(_dtf, 'Reference')

    _dtf = _dtf[['FP1', 'F7', 'F3', 'FP2', 'F8', 'F4']]

    return _dtf

_sel_user = 4
_sel_game = 'B'
_work_file = get_selected_file_name(
    sel_user=_sel_user,
    sel_game=_sel_game
)
_dataframe = get_dataframe(_work_file)

# global variables declaration
s_rate = 512
_num_rows = 1024
_num_sequences = 20
_dtf_len = len(_dataframe)
# /global variables declaration

if _num_sequences is 'max':
    _num_sequences = fl(_dtf_len / _num_rows)
elif (_num_sequences * _num_rows) > _dtf_len:
    print("Data requested exceeds the accessible data.")
    print("Requsted sequences: {}".format(_num_sequences))
    print("Length of sequence: {}".format(_num_rows))
    print("\tRows requested: {}".format(_num_sequences*_num_rows))
    print("\tData available: {}".format(_dtf_len))

    print("If you want to use the full dataframe, set _num_sequences to 'max'")
    exit(1)

print("Data requested can be obtained from the current data.")
print("Requsted sequences: {}".format(_num_sequences))
print("Length of sequence: {}".format(_num_rows))
print("\tRows requested: {}".format(_num_sequences*_num_rows))
print("\tData available: {}".format(_dtf_len))

delta_max_list = []
theta_max_list = []
alpha_max_list = []
beta_max_list = []

for i in range(0, _num_sequences):
    print("Sequence {0}/{1}".format(i+1, _num_sequences))
    f, d_a, t_a, a_a, b_a = get_assym(_dataframe, i, _num_rows)

    subplot_bands(
        y=f, delta=d_a, theta=t_a,
        alpha=a_a, beta=b_a,
        ylim_l=-0.2, ylim_h=0.2,
        folder="asym", fig_num=i,
        is_active=True, save_fig=True,
        xlab='epocs', ylab='index')

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

plt.close("all")
fig, axes = plt.subplots(2, 2)

axes[0, 0].plot(delta_max_list)
axes[0, 0].set_title("Max DELTA")
axes[0, 0].set(xlabel='epocs', ylabel='index')

axes[0, 1].plot(theta_max_list)
axes[0, 1].set_title("Max THETA")
axes[0, 1].set(xlabel='epocs', ylabel='index')

axes[1, 0].plot(alpha_max_list)
axes[1, 0].set_title("Max ALPHA")
axes[1, 0].set(xlabel='epocs', ylabel='index')

axes[1, 1].plot(beta_max_list)
axes[1, 1].set_title("Max BETA")
axes[1, 1].set(xlabel='epocs', ylabel='index')

fig.subplots_adjust(hspace=0.6)
fig.subplots_adjust(wspace=0.5)

plt.savefig("subplots/max/sub_max_USR{0}_{1}_#{2}.png".format(
    _sel_user, _sel_game, _num_sequences))
plt.close('all')
