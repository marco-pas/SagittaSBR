module purge



# module load git/2.41.0
# module load cmake/3.27.7
# # module load nvtop/2.0.2-gcc-11.3.0-why26pe
# # module load cpe-cuda/24.11
# module load openmpi/5.0.8-gcc12.2.0-multi-node
# # module load nvtop
# module load rocm/6.3.3
# module load gcc/12.2.0



# source /cfs/klemming/home/p/pennati/development/raytracingSBR/sagittaSBR.profile
# scracth /cfs/klemming/scratch/p/pennati/gromacs
#module purge --force
module load cmake/4.0.1
module load python/3.12.3

#module load gcc-native/13.2
module load PrgEnv-gnu/8.6.0
#module load cray-libsci/24.11.0
#module load cray-fftw/3.3.10.9
#module load PrgEnv-cray/8.6.0
#module load craype-network-ucx
module load craype-accel-amd-gfx90a
module load rocm/6.3.3
export MPICH_GPU_SUPPORT_ENABLED=1
module load cray-mpich/8.1.31

# source /cfs/klemming/home/p/pennati/development/raytracingSBR/saggitaPy/bin/activate

export MPICH_INCLUDE_GNU="/opt/cray/pe/mpich/8.1.31/ofi/gnu/12.3/include"
export MPICH_LIB_GNU="/opt/cray/pe/mpich/8.1.31/ofi/gnu/12.3/lib/libmpi_gnu_123.so"
export MPICH_LIB_GNU_GPU="/opt/cray/pe/mpich/8.1.31/ofi/gnu/12.3/lib/libmpi_gtl_hsa.so"

# export MPICH_HOME_CRAY="/opt/cray/pe/mpich/8.1.31/ofi/cray/17.0/"
# export MPICH_INCLUDE_CRAY="/opt/cray/pe/mpich/8.1.31/ofi/cray/17.0/include"
# export MPICH_LIB_CRAY="/opt/cray/pe/mpich/8.1.31/ofi/cray/17.0/lib/libmpi_cray.so"
# export MPICH_LIB_GPU="/opt/cray/pe/mpich/8.1.31/gtl/lib/libmpi_gtl_hsa.so"
#export CMAKE_PREFIX_PATH="$MPI_CRAY_HOME:$CMAKE_PREFIX_PATH"

export MPICH_INCLUDE=${MPICH_INCLUDE_GNU}
export MPICH_LIB=${MPICH_LIB_GNU}
export MPICH_LIB_GPU=${MPICH_LIB_GNU_GPU}



if false; then

# use this
  cmake .. \
  -DUSE_HIP=ON \
  -DCMAKE_C_COMPILER=cc \
  -DCMAKE_CXX_COMPILER=CC \
  -DCMAKE_HIP_COMPILER="/opt/rocm-6.3.3/bin/amdclang++" \
  -DAMDGPU_TARGETS="gfx90a" \
  -DCMAKE_HIP_FLAGS="-I${MPICH_INCLUDE} -lxpmem -L${MPICH_LIB} -L${MPICH_LIB_GPU}" \
  > build.out 2>&1 &
  cmake .. \
  -DUSE_HIP=ON \
  -DCMAKE_C_COMPILER=cc \
  -DCMAKE_CXX_COMPILER=CC \
  -DCMAKE_HIP_COMPILER="/opt/rocm-6.3.3/bin/amdclang++" \
  -DAMDGPU_TARGETS="gfx90a" \
  -DCMAKE_HIP_FLAGS="--offload-arch=gfx90a -I${MPICH_INCLUDE} -lxpmem -L${MPICH_LIB} -L${MPICH_LIB_GPU}" \
  > build.out 2>&1 &

fi

# install gromacs
# -DGMX_GPU=HIP -DGMX_HIP_TARGET_ARCH=gfx90a

# salloc
# salloc --nodes=1 -t 1:00:00 -A naiss2024-5-582 -p gpu

# deepModels
# /cfs/klemming/home/p/pennati/development/gromacs/deepmd/deepmodels/dpa1/unke2019_solvatedProtein/dpa1_float32_v2/dpa1_fp32_solvated_protein_v2.pth
# /cfs/klemming/home/p/pennati/development/gromacs/deepmd/deepmodels/dpa1/unke2019_solvatedProtein/dpa1_float32_v3/dpa1_fp32_solvated_protein_v3.pth
# /cfs/klemming/home/p/pennati/development/gromacs/deepmd/deepmodels/dpa2_4_7M/dpa2_4_7M_solvated_protein_frozen.pth

# mpi
# /opt/cray/pe/mpich/8.1.31
# /opt/cray/pe/mpich/8.1.31/ofi/gnu/12.3/include/ù

# TORCH_LIB_DIR="/cfs/klemming/scratch/p/pennati/gromacs/gromacsDeepPy/lib/python3.12/site-packages/torch/lib/"
# mv "$TORCH_LIB_DIR/libnuma.so" "$TORCH_LIB_DIR/libnuma.so.pipbak"
# ln -s /usr/lib64/libnuma.so.1 "$TORCH_LIB_DIR/libnuma.so"
# readelf -d "$TORCH_LIB_DIR"/lib*.so | grep -E 'NEEDED.*numa|RPATH|RUNPATH' || true

# which gmx_mpi ldd $(which gmx_mpi) | egrep -i "libsci|fftw" || echo "No LibSci in dynamic deps"

#  ldd /cfs/klemming/home/p/pennati/development/gromacs/gromacsSrc/gromacs/bin/bin/gmx_mpi | grep -E 'libhsa-runtime64|libamdhip64'
# ldd ${target} | grep -E 'libname'




# this one works


# cmake .. \
#   -DUSE_HIP=ON \
#   -DCMAKE_C_COMPILER=cc \
#   -DCMAKE_CXX_COMPILER=CC \
#   -DCMAKE_HIP_COMPILER="/opt/rocm-6.3.3/bin/amdclang++" \
#   -DAMDGPU_TARGETS="gfx90a" \
#   -DCMAKE_HIP_ARCHITECTURES="gfx90a" \
#   -DCMAKE_HIP_FLAGS="--offload-arch=gfx90a -I${MPICH_INCLUDE} -lxpmem -L/opt/cray/pe/mpich/8.1.31/ofi/gnu/12.3/lib -lmpi_gnu_123 -lmpi_gtl_hsa"