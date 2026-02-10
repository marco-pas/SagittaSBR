![My Logo](./assets/SBR.svg)

# SBR Project for RCS Calculations

SagittaSBR is an open-source ray tracing code in C++/CUDA for calculating the monostatic Radar Cross Section (RCS) of objects in the far field. It uses the Shooting-and-Bouncing Rays (SBR) method, which is a combination of Geometrical Optics (GO) and Physical Optics (PO). The code tracks ray hits and computes the total scattered field at the receiver.

## Running a Simulation

Before compiling, make sure you have a compatible GPU (NVIDIA or AMD). Make also sure to have a MPI implementation available.

### Compilation

#### NVIDIA GPUs (CUDA)

Make sure you have the CUDA toolkit installed.

Compilation for Nvidia GPUs:
```bash
mkdir build
cd build
cmake ..
make -j
```

#### AMD GPUs (HIP/ROCm)

Make sure you have ROCm installed with HIP, hipBLAS, and hipRAND.

```bash
mkdir build
cd build
cmake -DUSE_HIP=ON ..
make -j
```

You can also specify a target architecture:

```bash
cmake -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES="gfx90a" ..
```

Note: some additional flags may be required to compile, depending on the system. 

### Double Precision

To compile in double precision use `cmake -DDOUBLE_PRECISION=ON ..`


## Configuration

Modify `config.txt` to set the simulation options. You can specifiy the location of the file when launching the simulation. 
You can use `#` to comment out lines.
If an option is missing, the code will fall back to hard-coded default values
defined in `src/main`. 

## Execution

It is necessary to indicate the directory of the hittable object. Supported extensions: `.obj`, `.gltf`, `.glb`.

```bash
./RT-RCS --model path/to/model.obj
```

Optional arguments: 
- `./RT-RCS --model path/to/model.obj 1.0e9` to directly specify the frequency without modifying `config.txt` (frequency may appear anywhere in the argument list).
- `./RT-RCS --model path/to/model.obj --model-scale 0.5` to scale the dimensions of the hittable object.
- `./RT-RCS --model path/to/model.obj --config path/to/config.txt` to specify the location of the configuration file. If no location is specified, then the code tries to look for it in the current folder, then falls back to default values if no such file is present.

### BVH Builder Options (Code-Level)

BVH build parameters live in `bvhBuildOptions` (see `include/scene/bvhBuilder.hpp`) and are wired in `src/app/rtRcsEntry.cu`. You can switch between the simple splitter and the SAH splitter by changing `buildOptions.algorithm` in code.

## Visualization

To plot the RCS as a function of the azimuthal angle $\varphi$, and polar angle $\theta$
use the provided Python scripts in the `util/` folder for plotting in 1D, 2D and 3D.
Make sure all required Python packages are installed. Using a Python virtual environment is recommended.

```bash
python3 tools/plotRCS.py
python3 tools/plotRCS2d.py
python3 tools/plotRCS3d.py
```

Plotting scripts read the `rcs_results.csv` file, saved in the simulation folder. To change the location of these file, modify the scripts.

### Optional arguments

Options for the Radar Cross Section (RCS) plot of the results from `rcs_results.csv`.

Command-line options:
- `--unit`   : Select plotting unit ('dbsm' or 'm2'), default is 'dbsm'
- `--plot`   : Enable or disable plot saving/display (True/False), default True
- `--scans`  : Enable saving averaged RCS data for parameter scans, default False

Examples:
```bash
python3 tools/plotRCS.py
python3 tools/plotRCS.py --unit=m2
python3 tools/plotRCS.py --plot=False --scans=True --unit=m2
```


---

## Frequency Scans

To perform frequency sweeps, you can use or adapt the provided shell script.
This script runs simulations at multiple frequencies, saves the data for each run,
and then produces a final Python plot together with analytical results obtained
from Mie scattering for a PEC sphere.

```bash
./tools/run_freq_scan.sh
```

## Parallelization

OpenMP is used to build the BVH of the object.

MPI is used to perform angle scans.

