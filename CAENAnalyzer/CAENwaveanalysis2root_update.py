import sys
import os

import ROOT
from array import array
from CAENReader import DataFile
import numpy as np

# Check if the correct number of command-line arguments is provided
if len(sys.argv) != 2:
    print("Usage: python CAENwaveanalysis2root_update.py <file_index>")
    sys.exit(1)
file_index = sys.argv[1]
# Get the DataFile name from the command-line argument
data_file_name = f'wavestest_{file_index}.bin'

# Create an instance of DataFile with the provided .dat file name
data_file = DataFile(data_file_name)

# Extract the file index from the DataFile name
file_index = int(os.path.splitext(os.path.basename(data_file_name))[0].split('_')[-1])

# Create a ROOT file with a name based on the file index
output_root_file_name = os.path.join('rawdata',"output_{}.root".format(file_index))
root_file = ROOT.TFile(output_root_file_name, "RECREATE")

# Define a tree
tree = ROOT.TTree("T", "test tree")

# Prepare arrays for each variable you want to store
event_counter = array('i', [0])
trace_vectors = {}

# Create branches in the tree
tree.Branch("eventCounter", event_counter, "eventCounter/I")

# Counter for iteration
counter = 0

while True:
    trigger = data_file.getNextTrigger()
    if trigger is None:
        break

    # Assuming trigger has attributes: eventCounter and traces
    event_counter[0] = trigger.eventCounter
    traces = trigger.traces  # Assuming trigger.traces returns a dictionary

    for key, trace in traces.items():
        # If we haven't created a branch for this trace yet
        if key not in trace_vectors:
            # Create a std::vector for this trace
            trace_vectors[key] = ROOT.std.vector('double')()

            # Create a branch for this trace
            tree.Branch(key, trace_vectors[key])

        # Clear the vector for this trace and fill it with the current trace data
        trace_vectors[key].clear()
        for val in trace:
            trace_vectors[key].push_back(float(val))

    # Fill the tree
    tree.Fill()

    # Increment and print the counter
    counter += 1
    print("Number of iterations: {}".format(counter))

    # Save data to the ROOT file every 100 events
    if counter % 100 == 0:
        tree.Write("", ROOT.TObject.kOverwrite)

# Write the tree into the ROOT file and close the file
tree.Write()
root_file.Close()
data_file.close()

