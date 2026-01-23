"""
Plot 2D Radar Cross Section (RCS) results from output/rcs_results.csv.

Command-line options:
  --unit   : Select plotting unit ('dbsm' or 'm2'), default is 'dbsm'
  --plot   : Enable or disable plot saving/display (True/False), default True
  --scans  : Enable saving averaged RCS data for parameter scans, default False

Examples:
  python3 util/plotRCS_2D.py
  python3 util/plotRCS_2D.py --unit=m2
  python3 util/plotRCS_2D.py --plot=False --scans=True --unit=m2
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import csv


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "STIXGeneral", "DejaVu Serif"],   
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})


# Argument parsing

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected (True/False).')


parser = argparse.ArgumentParser(description="Plot 2D RCS in dBsm or m^2")

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

# Check if theta column exists
if 'theta' not in df.columns:
    print("Warning: 'theta' column not found in CSV. Using default theta=90 for all points.")
    df['theta'] = 90.0

# Get unique values
phi_vals = np.sort(df['phi'].unique())
theta_vals = np.sort(df['theta'].unique())

# Create meshgrid
phi_grid, theta_grid = np.meshgrid(phi_vals, theta_vals)

# Select data field
if plot_dbsm:
    data_field = 'rcs_dbsm'
    label_str = 'RCS (dBsm)'
    unit_str = 'dBsm'
    cmap = 'plasma'
else:
    data_field = 'rcs_m2'
    label_str = 'RCS ($m^2$)'
    unit_str = '$m^2$'
    cmap = 'hot'

# Clip numeric RCS values to avoid extreme color scaling in the plot
vmin, vmax = -20, 63
df['rcs_clipped'] = df[data_field].clip(lower=vmin, upper=vmax)

# Pivot data for 2D plotting
rcs_grid = df.pivot_table(values=data_field, index='theta', columns='phi', aggfunc='mean')
rcs_array = rcs_grid.values

# Calculate average RCS
avg_rcs = df[data_field].mean()

# Plot setup
fig, ax = plt.subplots(figsize=(11, 5))

# Create contour plot
im = ax.pcolormesh(phi_grid, theta_grid, rcs_array, shading='auto', cmap=cmap)
# im.set_clim(vmin=-10, vmax=45)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label(label_str, fontsize=12)
# cbar.set_clim([-10, 45])

# Styling
# ax.set_title('2D Monostatic Radar Cross Section', fontsize=16)
ax.set_xlabel(r'Angle $\phi$ (deg)', fontsize=14)
ax.set_ylabel(r'Angle $\theta$ (deg)', fontsize=14)
ax.grid(True, linestyle='--', alpha=0.3, color='white')

plt.xticks(np.arange(0, 360, 15))
plt.yticks(np.arange(0, 180, 15))

plt.tight_layout()

# Save / show plot

if save_plots:
    if not os.path.exists("figures"):
        os.makedirs("figures")

    output_path = get_next_filename("figures", "rcs_2d_plot", "pdf")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"--------- Plot saved to {output_path} ---------")
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