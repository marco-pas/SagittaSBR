import numpy as np
import matplotlib.pyplot as plt
import os
import csv

from scipy.special import spherical_jn, spherical_yn


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "STIXGeneral", "DejaVu Serif"],   
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})

def normalized_rcs(rel_freq):
    """
    Calculate the monostatic RCS of a PEC sphere normalized by πr².

    Parameters:
    relative_frequency (float): The relative frequency (2πr/λ), unitless.

    Returns:
    float: Normalized RCS (σ / πr²), unitless.
    """

    if rel_freq <= 0:
        return 0.0

    if rel_freq >= 1e3:
        return 1.0

    # determine the maximum number of terms in the series
    N_max = int(np.floor(rel_freq + 4 * rel_freq ** (1 / 3) + 2))
    total_sum = 0.0j

    for n in range(1, N_max + 1):
        # compute spherical Bessel (jn) and Neumann (yn) functions
        jn = spherical_jn(n, rel_freq)  # j_n(x)
        jn_deriv = spherical_jn(n, rel_freq, True)  # j_n'(x)

        yn = spherical_yn(n, rel_freq)  # y_n(x)
        yn_deriv = spherical_yn(n, rel_freq, True)  # y_n'(x)

        # spherical Hankel function of the first kind (hn1)
        hn1 = jn + 1j * yn  # h_n^(1)(x)
        hn1_deriv = jn_deriv + 1j * yn_deriv  # [h_n^(1)]'(x)

        # calculate (x * jn)' and (x * hn1)'
        x_jn_deriv = jn + rel_freq * jn_deriv
        x_hn1_deriv = hn1 + rel_freq * hn1_deriv

        # Mie coefficients for PEC sphere
        a_n = -x_jn_deriv / x_hn1_deriv
        b_n = -jn / hn1

        # series term
        term = ((-1) ** n) * (n + 0.5) * (a_n - b_n)
        total_sum += term

    # compute the normalized RCS
    if rel_freq == 0:
        return 0.0
    rcs_normalized = (4.0 / rel_freq**2) * (np.abs(total_sum) ** 2)

    return rcs_normalized.real

c = 299792458
r = 1

# Read data from CSV file
frequencies = []
results = []
std = []

with open("results_freq_scans_PAPER2.csv", "r") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        frequencies.append(float(row["Frequency"]))
        results.append(float(row["Avg_RCS"]))
        std.append(float(row["Std_RCS"]))

# Convert to numpy arrays
f = np.array(frequencies)
results_norm = np.array(results) / (np.pi * r**2)  # Normalize simulation results
std_norm = np.array(std) / (np.pi * r**2)

relative_freq = (2 * np.pi * r) * f / c
relative_freq_analytical = np.logspace(np.log10(min(relative_freq)), 
                                       np.log10(max(relative_freq)), 1000)

plt.figure(figsize=(11, 6))

plt.plot(relative_freq, results_norm, 'b-', markersize=8, label='SagittaSBR', linewidth=2)

results_analyt = [normalized_rcs(rel_f) for rel_f in relative_freq_analytical]
plt.plot(relative_freq_analytical, results_analyt, 'g--', label='Mie Theory', linewidth=2)

plt.legend(fontsize=16)
plt.xlabel(r'Electrical size $2\pi r / \lambda_0$', fontsize=16)
plt.ylabel(r'Normalized RCS $\sigma / (\pi r^2)$', fontsize=16)
# plt.title('RCS Comparison: Numerical Simulation vs Analytical Mie Theory', fontsize=16)

plt.yscale('log')
plt.xscale('log')

plt.axvline(x=30, color='k', linestyle='--', linewidth=1.5)
plt.axvline(x=7e3, color='k', linestyle='--', linewidth=1.5)

# plt.xlim(1, 1e4)
plt.ylim(1.2e-1, 1.2e1)

plt.grid(True, which='both', alpha=0.3)

plt.text(3, 1.5e-1, 'Mie Region', fontsize=16, color='black')
plt.text(150, 1.5e-1, 'Optical Region', fontsize=16, color='black')
plt.text(9e3, 1.5e-1, 'Low Ray\nDensity', fontsize=16, color='black')

# Add annotation about the sphere radius
# plt.text(0.02, 0.98, f'Sphere radius: r = {r} m', 
#          transform=plt.gca().transAxes, fontsize=12,
#          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

folder = "./"
os.makedirs(folder, exist_ok=True)

plt.savefig(os.path.join(folder, "rcs_comparison_sphere_PAPER2.pdf"),
            dpi=300, bbox_inches="tight")


print(f"\nPlot saved to: {os.path.join(folder, 'rcs_comparison_sphere_PAPER2.pdf')}")

# Calculate and print relative errors - CORRECTED
if len(results_norm) == len(relative_freq):
    print("\nRelative Errors (Simulation vs Analytical):")
    print("-" * 50)
    for i, (rel_f, sim_rcs_norm) in enumerate(zip(relative_freq, results_norm)):
        anal_rcs = normalized_rcs(rel_f)
        rel_error = abs(sim_rcs_norm - anal_rcs) / anal_rcs * 100
        print(f"x = {rel_f:.2e}: Sim = {sim_rcs_norm:.4f}, Anal = {anal_rcs:.4f}, Error = {rel_error:.2f}%")



print()
print("standard_deviation", std)

print()
print("standard_deviation_normalized", std_norm)