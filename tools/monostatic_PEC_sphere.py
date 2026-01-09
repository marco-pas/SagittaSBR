import numpy as np
import matplotlib.pyplot as plt
import os
import csv

from scipy.special import spherical_jn, spherical_yn

def normalized_rcs(rel_freq):

    # Analytical results

    if rel_freq <= 0:
        return 0.0

    if rel_freq >= 20:
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

with open("output/results_freq_scans.csv", "r") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        frequencies.append(float(row["Frequency"]))
        results.append(float(row["Avg_RCS"]))

# Convert to numpy arrays
f = np.array(frequencies)
results_norm = np.array(results) / (np.pi * r**2)  # Normalize simulation results

relative_freq = (2 * np.pi * r) * f / c
relative_freq_analytical = np.logspace(np.log10(min(relative_freq)), 
                                       np.log10(max(relative_freq)), 1000)

plt.figure(figsize=(14, 6))

plt.plot(relative_freq, results_norm, '.r', markersize=8, label='Numerical (Simulation)')

results_analyt = [normalized_rcs(rel_f) for rel_f in relative_freq_analytical]
plt.plot(relative_freq_analytical, results_analyt, 'b--', label='Analytical (Mie Theory)')

plt.legend(fontsize=12)
plt.xlabel(r'Relative Frequency $x = 2\pi r / \lambda_0$', fontsize=14)
plt.ylabel(r'Normalized RCS $\sigma / (\pi r^2)$', fontsize=14)
plt.title('RCS Comparison: Numerical Simulation vs Mie Theory', fontsize=16)

plt.yscale('log')
plt.xscale('log')

plt.grid(True, which='both', alpha=0.3)

# Add annotation about the sphere radius
# plt.text(0.02, 0.98, f'Sphere radius: r = {r} m', 
#          transform=plt.gca().transAxes, fontsize=12,
#          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

folder = "tools/benchmark_sphere"
os.makedirs(folder, exist_ok=True)

plt.savefig(os.path.join(folder,"rcs_comparison.png"),
            dpi=300, bbox_inches="tight")

print(f"\nPlot saved to: {os.path.join(folder, 'rcs_comparison.png')}")

# Calculate and print relative errors - CORRECTED
if len(results_norm) == len(relative_freq):
    print("\nRelative Errors (Simulation vs Analytical):")
    print("-" * 50)
    for i, (rel_f, sim_rcs_norm) in enumerate(zip(relative_freq, results_norm)):
        anal_rcs = normalized_rcs(rel_f)
        rel_error = abs(sim_rcs_norm - anal_rcs) / anal_rcs * 100
        print(f"x = {rel_f:.2e}: Sim = {sim_rcs_norm:.4f}, Mie = {anal_rcs:.4f}, Error = {rel_error:.2f}%")