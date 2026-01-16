"""
Plot Radar Cross Section (RCS) results from output/rcs_results.csv.

Command-line options:
  --unit   : Select plotting unit ('dbsm' or 'm2'), default is 'dbsm'
  --plot   : Enable or disable plot saving/display (True/False), default True
  --scans  : Enable saving averaged RCS data for parameter scans, default False

Examples:
  python3 util/plotRCS.py
  python3 util/plotRCS.py --unit=m2
  python3 util/plotRCS.py --plot=False --scans=True --unit=m2
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import csv



# Argument parsing


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected (True/False).')


parser = argparse.ArgumentParser(description="Plot RCS in dBsm or m^2")

parser.add_argument(
    '--unit',
    choices=['dbsm', 'm2'],
    default='dbsm',
    help="Plotting unit: 'dbsm' (default) or 'm2'"
)

parser.add_argument(
    '--plot',
    type=str2bool,
    default=True,
    help="Enable or disable plotting and saving figures (default: True)"
)

parser.add_argument(
    '--scans',
    type=str2bool,
    default=False,
    help="Save averaged RCS values for parameter scans (default: False)"
)

args = parser.parse_args()

# Derived settings

plot_dbsm = args.unit == 'dbsm'
save_plots = args.plot
save_data_scans = args.scans

print()
print(f"Plot unit      : {args.unit}")
print(f"Save plots     : {save_plots}")
print(f"Save scan data : {save_data_scans}")
print()

# Helper functions
def get_next_filename(base_path, filename_base, extension):
    i = 1
    while True:
        candidate = os.path.join(base_path, f"{filename_base}_{i}.{extension}")
        if not os.path.exists(candidate):
            return candidate
        i += 1

# Load metadata

folder_csv = "."

file_csv = folder_csv + "/rcs_results.csv"

metadata = []
with open(file_csv, 'r') as f:
    for line in f:
        if line.startswith('#'):
            metadata.append(line.replace('#', '').strip())
        else:
            break

freq_str = metadata[0].replace("Frequency: ", "").replace(" Hz", "")
frequency = float(freq_str)

# Load data

df = pd.read_csv(file_csv, comment='#')
df = df.sort_values(by='phi')

# Plot setup
fig, ax = plt.subplots(figsize=(14, 7))
plt.subplots_adjust(right=0.8)

# Plotting logic

if plot_dbsm:
    plot_field = df['rcs_dbsm']
    label_str = 'RCS (dBsm)'
    unit_str = 'dBsm'
else:
    plot_field = df['rcs_m2']
    label_str = 'RCS ($m^2$)'
    unit_str = '$m^2$'

    avg_rcs = plot_field.mean()
    ax.axhline(
        y=avg_rcs,
        color='red',
        linestyle='--',
        linewidth=1.5,
        label=f'Average: {avg_rcs:.4f} {unit_str}'
    )


ax.plot(
    df['phi'],
    plot_field,
    color='#1f77b4',
    linewidth=2,
    label=f'Monostatic {label_str}'
)

# Styling

ax.set_title('Monostatic Radar Cross Section', fontsize=16)
ax.set_xlabel('Phi (Degrees)', fontsize=12)
ax.set_ylabel(label_str, fontsize=12)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='upper left')

# Metadata box
info_text = "Simulation Parameters:\n" + "\n".join(metadata)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(
    1.05,
    0.95,
    info_text,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment='top',
    bbox=props
)

# Save / show plot

if save_plots:
    if not os.path.exists("figures"):
        os.makedirs("figures")

    output_path = get_next_filename("figures", "rcs_plot", "png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"- - - - - Plot saved to {output_path} - - - - -")
else:
    plt.close()

# Scan data output

if not plot_dbsm:
    print(f"Computed Average RCS: {avg_rcs:.4f} m^2")
    print()

    if save_data_scans:
        filename_scan = folder_csv + "/results_freq_scans.csv"
        with open(filename_scan, "a", newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(["Frequency", "Avg_RCS"])
            writer.writerow([frequency, avg_rcs])

if save_plots:
    plt.show()