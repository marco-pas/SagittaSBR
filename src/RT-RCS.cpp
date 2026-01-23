// Starting point

#include "app/rtRcsEntry.hpp"

#include <nvtx3/nvToolsExt.h>


int main(int argc, char** argv) {


    nvtxRangePushA("runRcsApp");

    return runRcsApp(argc, argv);

    nvtxRangePop();
}
