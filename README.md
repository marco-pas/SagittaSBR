![My Logo](./SBR.svg)

# SBR Project for RCS Calculations

SagittaSBR is an open-source ray tracing code in C++/CUDA for calculating the monostatic Radar Cross Section (RCS) of objects in the far field. It uses the Shooting-and-Bouncing Rays (SBR) method, which is a combination of Geometrical Optics (GO) and Physical Optics (PO). The code tracks ray hits and computes the total scattered field at the receiver.

This code takes inspiration from the [Accelerated Ray Tracing in One Weekend in CUDA tutorial](https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/), which focuses on optical ray tracing.

## Running a Simulation

Before compiling, make sure you have a compatible GPU and adapt the Makefile based on your GPU architecture.

### Compilation

```bash
make clean
make
```

## Configuration

Modify `config.txt` to set up the simulation options.


## Execution


```bash
./SBR
```

Optional arguments: `./SBR 1.0e9` to directly specify the frequency without modifying config.txt.

## Visualization

```bash
python3 plotRCS.py
```

Optional arguments: `python3 plotRCS.py m2` to display the radar cross section in $m^2$ instead of dBsm.


## Frequency Scans

To perform frequency sweeps, you can use or adapt the provided shell script:

```bash
./run_freq_scan.sh
```

