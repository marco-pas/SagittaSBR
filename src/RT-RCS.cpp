// Starting point

#include "app/rtRcsEntry.hpp"

#if defined(USE_HIP)
#else
#include <nvtx3/nvToolsExt.h>
#endif

int main(int argc, char** argv) {

#ifndef USE_HIP
    nvtxRangePushA("runRcsApp");
#endif

    return runRcsApp(argc, argv);

#ifndef USE_HIP
    nvtxRangePop();
#endif
}
