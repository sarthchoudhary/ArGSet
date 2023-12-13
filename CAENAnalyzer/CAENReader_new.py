import os
import ctypes
import csv
import math
import random
import ROOT 
from ROOT import TCanvas, TGraph
from ROOT import gROOT
from root_numpy import array2tree
import sys
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import chirp, find_peaks, peak_widths
from itertools import chain
from scipy.integrate import simps
from scipy import optimize, signal
import pylab as plt
import statsmodels.api as sm
from lmfit import models

# Open the ROOT file and get the tree
f = ROOT.TFile.Open('rawdata/waveforms.root')
tree = f.Get('T')

# Check if the tree is a valid TTree
if not isinstance(tree, ROOT.TTree):
    print("Error: The object obtained from the file is not a TTree.")
    f.Close()
    exit()

# Create a new ROOT file for analysis results
f_analysis = ROOT.TFile('waveanalysis.root', 'recreate')

# Create a TTree for analysis results
analysis_tree = ROOT.TTree('AnalysisTree', 'Tree with waveform analysis results')

# Variable to store max_value
peakarea = np.zeros(1, dtype=float)
max_value = np.zeros(1, dtype=float)
event = np.zeros(1, dtype=int)
# Create a branch for max_value in the new tree
analysis_tree.Branch('peakarea', peakarea, 'peakarea/D')
analysis_tree.Branch('max_value', max_value, 'max_value/D')  # 'D' stands for double (float in Python)
analysis_tree.Branch('event', event, 'event/I') 
# Get the number of entries in the tree
nentries = tree.GetEntries()

# Define sample rate
sample_rate = 0.1  # Replace this with your actual sample rate

# 3 dB level
level_3dB = 10  # Replace this with your actual 3 dB level

# Number of samples to use for baseline calculation
baseline_samples = 200

# Loop over the entries in the tree
for i in range(nentries):
    # Get the i-th entry
    tree.GetEntry(i)
    event[0]=i+1
    print("Event number {}: {}".format(i + 1, event[0]))

    # Store the b0tr0 values as a numpy array and ignore first and last 4 elements
    wvfrm = np.array(tree.b0tr0[4:-4])
    time_array = np.arange(len(wvfrm)) * sample_rate
    #print("Length of time_array array for entry {}: {}".format(i + 1, len(time_array)))
    # Calculate the baseline as the average of the first baseline_samples samples
    baseline_samples = min(baseline_samples, len(wvfrm))  # Ensure baseline_samples is within array bounds
    baseline = np.mean(wvfrm[:baseline_samples]) #if baseline_samples > 0 else 0

    # Subtract the baseline from the waveform to remove the offset
    wvfrm = wvfrm - baseline

    # Apply LOWESS smoothing to wvfrm
    lowess = sm.nonparametric.lowess(wvfrm,time_array, frac=0.003)
    fltheight = lowess[:, 1]
    #noisemax= np.max(wvfrm[:baseline_samples])
    #risetime=next(x for x, val in enumerate(wvfrm) if val > 1.5*noisemax)-5 #change to lowess[:,1]
    #print ("This is the rise time of the signal: " +str(risetime))

    #rvwvfrm=lowess[::-1, 1] #wvfrm[::-1] #reverse waveform
    #noisemaxrev =np.max(rvwvfrm[:baseline_samples])
    #falltime=int(len(wvfrm)-next(x for x, val in enumerate(rvwvfrm[2000:]) if val > 2.5*noisemaxrev))+50
    #print ("This is the fall time of the signal: " +str(falltime))

    # Get the maximum value of the waveform and its index
    max_value[0] = np.max(wvfrm[::1]) if len(wvfrm) > 0 else 0
    max_index = np.argmax(lowess[:, 1])
    
    #print("The max value is {}: {}".format(i + 1, max_value)) #len(wvfrm) = 4062
    
    # Find the index of the first peak in the smoothed waveform
    peak_index = np.argmax(lowess[:, 1])

    # Define a window of 1000 samples after the first peak
    window_start = max(peak_index - 500, 0)
    window_end = min(peak_index + 2000, len(fltheight))

    # Find the index of the first value that is below 1.5 times the baseline noise level within the window
    noisemax = np.max(fltheight[:baseline_samples])
    #risetime=next(x for x, val in enumerate(fltheight) if val > 2*noisemax)-5 #change to lowess[:,1]
    

    risetime = next((x for x in range(window_start, peak_index) if fltheight[x] > 1.5 * noisemax), -1)
    falltime = next((x for x in range(peak_index, window_end) if fltheight[x] < 2 * noisemax), -1)

    # Print the rise time (if found)
    #if risetime != -1:
    #    print("This is the rise time of the signal: " + str(risetime))
    #else:
    #    print("Rise time not found within the specified window.")


    # Print the fall time (if found)
    #if falltime != -1:
    #    print("This is the fall time of the signal: " + str(falltime))
    #else:
    #    print("Fall time not found within the specified window.")


    peakarea[0]=np.trapz(fltheight[risetime:falltime],dx=0.001)
    print("Peak area is: " + str(peakarea))

    # Fill the branch with the new max_value
    analysis_tree.Fill()

    if i == 34:
        # Plot original data
        plt.plot(time_array,wvfrm, label='Original Data', alpha=0.5)

        # Plot smoothed data
        plt.plot(time_array,lowess[:, 1], color='red', label='LOWESS Smoothing')
        plt.scatter(time_array[max_index], max_value[0], color='green', marker='x', label='Max Value')

	# Add labels and legend
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.legend()

        # Show the plot
       	plt.show()

# Write the analysis tree to the new ROOT file
f_analysis.Write()
f_analysis.Close()

# Close the original ROOT file
f.Close()

