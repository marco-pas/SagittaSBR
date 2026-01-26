// Starting point

#include "app/rtRcsEntry.hpp"

#if defined(USE_HIP)
#else
#include <nvtx3/nvToolsExt.h>
#endif

int main(int argc, char** argv) {


    nvtxRangePushA("runRcsApp");

    return runRcsApp(argc, argv);

    nvtxRangePop();
}
