// Starting point

#include "app/rtRcsEntry.hpp"

#if defined(USE_HIP)
#else
#include <nvtx3/nvToolsExt.h>
#endif

#include <mpi.h>

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);  

#ifndef USE_HIP
    nvtxRangePushA("runRcsApp");
#endif

    int rc = runRcsApp(argc, argv);

#ifndef USE_HIP
    nvtxRangePop();
#endif

    MPI_Finalize();
    return rc;
}
