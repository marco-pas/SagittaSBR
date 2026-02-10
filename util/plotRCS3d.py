# Plot 3D Radar Cross Section (RCS) results from output/rcs_results.csv

"""
Interactive 3D RCS plot in dBsm using Plotly.
Generates HTML files with automatic numbering: rcs_3d_1.html, rcs_3d_2.html, etc.
Handles poles properly to avoid spikes/artifacts.
"""

# Creates an .html file

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

# Specify the .csv file location
folder_csv = "."
file_csv = os.path.join(folder_csv, "rcs_results.csv")

# Specify the saving location
output_folder = "figures"
output_base = "rcs_3d"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to generate next available filename
def get_next_filename(base_path, filename_base, extension):
    i = 1
    while True:
        candidate = os.path.join(base_path, f"{filename_base}_{i}.{extension}")
        if not os.path.exists(candidate):
            return candidate
        i += 1

# Load CSV
df = pd.read_csv(file_csv, comment='#')

# Ensure theta column exists
if 'theta' not in df.columns:
    df['theta'] = 90.0

# Select data field (always dbsm)
data_field = 'rcs_dbsm'
colorscale = 'Plasma'

# Pivot table
rcs_grid = df.pivot_table(values=data_field, index='theta', columns='phi', aggfunc='mean')
rcs_array = rcs_grid.values

# rcs_array = np.maximum(rcs_array, 61)
# rcs_array = np.minimum(rcs_array, -25)

theta_grid, phi_grid = np.meshgrid(rcs_grid.index.values, rcs_grid.columns.values, indexing='ij')

# Pole-safe adjustments
epsilon = 1e-5
theta_grid = np.clip(theta_grid, epsilon, 180 - epsilon)  # avoid exact poles

# Mask redundant φ values at poles to avoid spikes
rcs_array[0, 1:] = np.nan     # North pole
rcs_array[-1, 1:] = np.nan    # South pole

# Spherical -> Cartesian
theta_rad = np.deg2rad(theta_grid)
phi_rad   = np.deg2rad(phi_grid)
r = 1.0

X = r * np.sin(theta_rad) * np.cos(phi_rad)
Y = r * np.sin(theta_rad) * np.sin(phi_rad)
Z = r * np.cos(theta_rad)

# Plotly Surface
fig = go.Figure(data=[go.Surface(
    x=X,
    y=Y,
    z=Z,
    surfacecolor=rcs_array,   # raw dBsm values
    colorscale=colorscale,
    showscale=True,
    colorbar=dict(title='RCS (dBsm)'),
    cmin=np.nanmin(rcs_array),
    cmax=np.nanmax(rcs_array)
)])

fig.update_layout(
    title='Interactive 3D Radar Cross Section (dBsm)',
    scene=dict(
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
        zaxis=dict(showticklabels=False),
        aspectmode='data'
    )
)


# Save HTML
output_file = get_next_filename(output_folder, output_base, "html")
fig.write_html(output_file)
print(f"Interactive 3D plot saved to {output_file}")
print("Open this HTML file in a browser to rotate and zoom the sphere.")
