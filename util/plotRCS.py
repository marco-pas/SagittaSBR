import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import csv

save_data_scans = True # only if we are doing parameter scans

parser = argparse.ArgumentParser(description="Plot RCS in dBsm or m^2")
parser.add_argument(
    'mode', 
    nargs='?',  # optional positional argument
    choices=['dbsm', 'm2'],  # allowed values
    default='dbsm',  # default if nothing is provided
    help="Choose 'dbsm' to plot in dBsm (default) or 'm2' to plot in m^2"
)

args = parser.parse_args()

# Determine the plotting mode
plot_dbsm = args.mode == 'dbsm'

print()

# Print the mode
if plot_dbsm:
    print("Plotting RCS in dBsm")
else:
    print("Plotting RCS in m^2")

def get_next_filename(base_path, filename_base, extension):
    i = 1
    while True:
        candidate = os.path.join(base_path, f"{filename_base}_{i}.{extension}")
        if not os.path.exists(candidate):
            return candidate
        i += 1

# 1. Load Metadata
metadata = []
with open('rcs_results.csv', 'r') as f:
    for line in f:
        if line.startswith('#'):
            metadata.append(line.replace('#', '').strip())
        else:
            break
freq_str = metadata[0].replace("Frequency: ", "").replace(" Hz", "")
frequency = float(freq_str)

# 2. Load Data
df = pd.read_csv('rcs_results.csv', comment='#')
df = df.sort_values(by='phi')

fig, ax = plt.subplots(figsize=(14, 7))
plt.subplots_adjust(right=0.8) 

# 3. Plotting Logic
if plot_dbsm:
    plot_field = df['rcs_dbsm']
    label_str = 'RCS (dBsm)'
    unit_str = 'dBsm'
else:
    plot_field = df['rcs_m2']
    label_str = 'RCS ($m^2$)'
    unit_str = '$m^2$'
    
    # Calculate and plot the average line
    avg_rcs = plot_field.mean()
    ax.axhline(y=avg_rcs, color='red', linestyle='--', linewidth=1.5, 
               label=f'Average: {avg_rcs:.4f} {unit_str}')

# Main Data Plot
ax.plot(df['phi'], plot_field, color='#1f77b4', linewidth=2, label=f'Monostatic {label_str}')

# 4. Styling
ax.set_title('Monostatic Radar Cross Section', fontsize=16)
ax.set_xlabel('Phi (Degrees)', fontsize=12)
ax.set_ylabel(label_str, fontsize=12)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='upper left')

# 5. Metadata Box
info_text = "Simulation Parameters:\n" + "\n".join(metadata)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(1.05, 0.95, info_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

# 6. Save and Show
if not os.path.exists("results_sphere"):
    os.makedirs("results_sphere")

output_path = get_next_filename("results_sphere", "rcs_plot", "png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')

print()
if not plot_dbsm:
    print(f"Computed Average RCS: {avg_rcs:.4f} m^2")
    print()

    if save_data_scans:
        filename_scan = "results_freq_scans.csv"
        with open(filename_scan, "a", newline='') as f:
            writer = csv.writer(f)
            # Check if file is empty to write header
            if f.tell() == 0:
                writer.writerow(["Frequency", "Avg_RCS"])
            writer.writerow([frequency, avg_rcs])

print(f"- - - - - Plot saved to {output_path} - - - - -")
print()

plt.show()