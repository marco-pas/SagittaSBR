#!/bin/bash

# Parameters
F_START=5e7
F_END=7.5e11
N=5

# Check for NumPy
python3 - <<'EOF'
try:
    import numpy
except ImportError:
    print("ERROR: NumPy is not available.")
    print("Please activate your virtual environment or install NumPy:")
    print("  python3 -m venv venv")
    print("  source venv/bin/activate")
    print("  pip install numpy")
    raise SystemExit(1)
EOF

# Stop script if NumPy check failed
if [ $? -ne 0 ]; then
    exit 1
fi

rm -f ./output/results_freq_scans.csv

for i in $(seq 0 $((N-1))); do
    freq=$(python3 - <<EOF
import numpy as np

i = $i
N = $N
F_START = $F_START
F_END = $F_END

f = np.logspace(np.log10(F_START), np.log10(F_END), N)
f = np.round(f).astype(float)

print(f[i])
EOF
)

    echo
    echo "---------- Iteration $((i+1)) / $N ----------"
    echo "Frequency: $freq Hz"
    echo 

    ./bin/SagittaSBR "$freq"
    python3 tools/plotRCS.py --unit=m2 --plot=False --scans=True
done


echo
echo "Now plotting results!"
echo

python3 tools/monostatic_PEC_sphere.py