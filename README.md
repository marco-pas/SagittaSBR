
![My Logo](./extra/SBR.svg)

# Monostatic RCS Calculations

**SagittaSBR** is an open-source ray-tracing code written in **C++/CUDA** for calculating the
**monostatic Radar Cross Section (RCS)** of objects in the far field.
It uses the **Shooting-and-Bouncing Rays (SBR)** method, which combines
**Geometrical Optics (GO)** and **Physical Optics (PO)**.
The code tracks ray hits and computes the total scattered field at the receiver.

This code is inspired by the
Accelerated Ray Tracing in One Weekend in CUDA tutorial:
https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/

---

## Running a Simulation

Before compiling, make sure you have a compatible GPU and adapt the `Makefile`
to match your GPU architecture.

### Compilation

make clean
make

The compilation generates object files (.o) in `build/`,
while the executable is placed in the `bin/` directory.

You can see the available Makefile options by running:

make help

---

## Configuration

Modify `config.txt` to set the simulation options.
You can use `#` to comment out lines.
If an option is missing, the code will fall back to hard-coded default values
defined in `src/main`. 

The current version does not yet implement BVH acceleration or mesh support.
To change the positioning of objects (e.g., a sphere), modify the hard-coded values in `src/world_setup.cu`.

---

## Execution

```bash
./bin/SagittaSBR
```

Optional argument:

```bash
./bin/SagittaSBR 1.0e9
```

This directly specifies the frequency (in Hz) without modifying `config.txt`.

---

## Visualization

To plot the RCS as a function of the azimuthal angle $\varphi$,
use the provided Python script.
Make sure all required Python packages are installed;
using a Python virtual environment is recommended.

```bash
python3 tools/plotRCS.py
```

### Optional arguments

Plot Radar Cross Section (RCS) results from output/rcs_results.csv.

Command-line options:

  --unit   : Select plotting unit ('dbsm' or 'm2'), default is 'dbsm'
  
  --plot   : Enable or disable plot saving/display (True/False), default True
  
  --scans  : Enable saving averaged RCS data for parameter scans, default False

Examples:

  `python3 tools/plotRCS.py`
  
  `python3 tools/plotRCS.py --unit=m2`
  
  `python3 tools/plotRCS.py --plot=False --scans=True --unit=m2`

---

## Frequency Scans

To perform frequency sweeps, you can use or adapt the provided shell script.
This script runs simulations at multiple frequencies, saves the data for each run,
and then produces a final Python plot together with analytical results obtained
from Mie scattering for a PEC sphere.

```bash
./tools/run_freq_scan.sh
```

---

## Directory Layout

```bash
.
├── include/              # Header files (C++ and CUDA). Non-inline functions are defined in src/
├── src/                  # Implementation files
├── build/                # Build artifacts (generated)
├── bin/                  # Compiled executable (generated)
├── Makefile              # Build system
├── config.txt            # Simulation parameters
├── tools/                # Plotting scripts and scan utilities
├── output/               # Output CSV files
├── data/                 # More complex hittables (WIP)
└── extra/                # Old or auxiliary files
```


---

## Notes for Developers

- If you change any functions in `src/`, make sure to update the corresponding
  declarations in `include/`.
  
- Always run simulations from the project root directory to avoid issues
  with relative paths.

