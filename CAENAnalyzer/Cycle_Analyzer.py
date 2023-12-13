#!/usr/bin/env python

import os

thelist=list(range(1,3))
output_files = []

# Run the CAENwaveanalysis2root_update.py script for each index
for j in thelist:
    print("\nwavestest_%01d.bin\n" % (j))
    os.system(f"python3 CAENwaveanalysis2root_update.py {j}")
    
    # Extract the output file name for the current index
    output_files.append(f"rawdata/output_{j}.root")

# Merge the output files into a single file named waveforms.root
merged_output = "rawdata/waveforms.root"
hadd_command = f"hadd -f {merged_output} {' '.join(output_files)}"
os.system(hadd_command)
print("Files merged. Ready to analyze.")

print("Starting the analysis of the waveforms.")
os.system(f"python3 CAENReader_new.py")
