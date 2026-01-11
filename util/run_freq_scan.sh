#!/bin/bash

F_START=5e7
F_END=1e11
N=100

for i in $(seq 0 $((N-1))); do
    freq=$(python3 - <<EOF
import numpy as np
f = np.logspace(np.log10($F_START), np.log10($F_END), $N)
f = np.round(f).astype(float)
print(f[$i])
EOF
)
    echo "Running SBR at freq = $freq"
    ./SBR $freq
    python3 plotRCS.py m2
done
