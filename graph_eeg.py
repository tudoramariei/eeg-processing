import csv
import matplotlib.pyplot as plt
import pandas as pd


def plot_data(x_data, y_data, color):
    plt.plot(x_data, y_data, color)
    plt.show()


def get_headers(df):
    return df.column.values


def get_elements_from_rows(df, row_start, row_stop):
    return df.iloc[row_start:row_stop]


def get_elements_for_header_from_rows(df, header, row_start, row_stop):
    return df[header][row_start:row_stop]


csv_headers = [
    "Time", "FP1", "F7", "F3", "C3", "P3", "P7",
    "O1", "O2", "FP2", "F8", "F4", "C4", "P4", "P8",
    "Pz", "T8", "Sampling Rate"
]

csv_file = """\
resurse/\
TUI5_3Back-modded.csv\
"""

df = pd.read_csv(
    csv_file,
    delimiter=';'
    )

x = "Time"
y = "FP1"
z = "F7"

x_data = df["Time"]
y_data = df["FP1"]
z_data = df["F7"]

df.head()
df.plot(
    x=x,
    subplots=True,
    legend=False
)
plt.show()

# print(data.column.values)

# print(get_headers(data))

# start_row = 0
# stop_row = 5

# print(get_elements_for_header_from_rows(data, "Time", start_row, stop_row))
# print(get_elements_for_header_from_rows(data, "FP1", start_row, stop_row))

# plot_data(
#     get_elements_for_header_from_rows(data, "Time", start_row, stop_row),
#     get_elements_for_header_from_rows(data, "FP1", start_row, stop_row),
#     'r')
