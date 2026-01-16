![My Logo](./assets/SBR.svg)

# SBR Project for RCS Calculations

SagittaSBR is an open-source ray tracing code in C++/CUDA for calculating the monostatic Radar Cross Section (RCS) of objects in the far field. It uses the Shooting-and-Bouncing Rays (SBR) method, which is a combination of Geometrical Optics (GO) and Physical Optics (PO). The code tracks ray hits and computes the total scattered field at the receiver.

This code takes inspiration from the [Accelerated Ray Tracing in One Weekend in CUDA tutorial](https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/), which focuses on optical ray tracing.

## Running a Simulation

Before compiling, make sure you have a compatible GPU and adapt the Makefile based on your GPU architecture.

### Compilation

Make sure to have all the necessary modules to compile

```bash
mkdir build
cd build
cmake ..
make -j
```

## Configuration

Modify `config.txt` to set the simulation options.
You can use `#` to comment out lines.
If an option is missing, the code will fall back to hard-coded default values
defined in `src/main`. 

## Execution

```bash
./RT-RCS --model path/to/model.obj
```

Optional arguments: `./SBR 1.0e9` to directly specify the frequency without modifying `config.txt` (frequency may appear anywhere in the argument list).

### Model Loading (CLI)

Use the built binary name (for example, `RT-RCS` from CMake or `SBR` from the Makefile).

```bash
./RT-RCS --model path/to/model.obj
./RT-RCS --model path/to/model.glb --model-scale 0.1
```

Supported extensions: `.obj`, `.gltf`, `.glb`. `--model` is required for simulations.

### BVH Builder Options (Code-Level)

BVH build parameters live in `bvhBuildOptions` (see `include/scene/bvhBuilder.hpp`) and are wired in `src/app/rtRcsEntry.cu`. You can switch between the simple splitter and the SAH splitter by changing `buildOptions.algorithm` in code.

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

  `--unit`   : Select plotting unit ('dbsm' or 'm2'), default is 'dbsm'
  
  `--plot`   : Enable or disable plot saving/display (True/False), default True
  
  `--scans`  : Enable saving averaged RCS data for parameter scans, default False

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
