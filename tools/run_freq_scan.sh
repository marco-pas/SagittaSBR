#!/bin/bash

F_START=5e7
F_END=1e11
N=4

for i in $(seq 0 $((N-1))); do
    freq=$(python3 - <<EOF
import numpy as np
f = np.logspace(np.log10($F_START), np.log10($F_END), $N)
f = np.round(f).astype(float)
print(f[$i])
EOF
)
    echo "Running SBR at freq = $freq"
    ./bin/SagittaSBR $freq
    python3 tools/plotRCS.py --unit=m2 --plot=False --scans=True
done

python3 tools/monostatic_PEC_sphere.py
