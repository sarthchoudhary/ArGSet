import numpy as np
import matplotlib.pyplot as plt
from CAENReader import DataFile
from time import perf_counter
from ROOT import TH1F, TSpectrum

hist = TH1F("hist", "hist", 4096, 0, 4096)

ts = TSpectrum(2)

t0 = 2
peak_allowed_lowerlimit = 500.0
peak_allowed_upperlimit = 600.0
pulse_integration_window_begin = 500
pulse_integration_window_end = 700
cntr_1peak_wf = 0
cntr_wf = 0
segment_index_begin = 0
# segment_index_end = 0 # diag
segment_index_end = 4

RecordLength = 4070 #hardcoding
x = np.arange(float(RecordLength-t0))
integration_vector = np.array([])
start_time = perf_counter()
for seg_index in range(segment_index_begin, segment_index_end+1):
    print('segment no.:', seg_index)
    binary_segment_name = f'/work/sarthak/ArgSet/2023_12_05/Run_12/run0_raw_b0_seg{seg_index}.bin' # path on chuck
    open_file = DataFile(binary_segment_name)
    
    while True:
    # for n_exe in range(100): # diag
        trigger_r = open_file.getNextTrigger()
        if trigger_r is None:
            break
        # print('Event number:', trigger_r.eventCounter)
        # trace = trigger_r.traces['b0tr0']
        trace = trigger_r.traces['b0tr2']
        RecordLength = trace.shape[0]
        if RecordLength < 4070: # waveform shorter than 4070 are likely garbage
            continue
        subt_trace = trace[t0:] - np.mean(trace[t0:250])
        hist.Reset()
        # for i in range(RecordLength-t0):
        #     hist.SetBinContent(i+1, subt_trace[i])
        # fill_hist(hist, x, subt_trace)
        hist.FillN(subt_trace.size, x, subt_trace)
        nfound = ts.Search(hist, 45, "", 0.5)
        peakx = ts.GetPositionX()
        cntr_wf += 1 
        if nfound == 1 and (peakx[0] < peak_allowed_upperlimit) and (peakx[0] > peak_allowed_lowerlimit):
            cntr_1peak_wf += 1
            # take the baseline subtracted waveform and integrate the counts in trigger window followed by a histogramming
            pulse_win_integration_ch = np.sum(subt_trace[pulse_integration_window_begin:pulse_integration_window_end])      ## sum or integrate?
            integration_vector = np.append(integration_vector, pulse_win_integration_ch)

print('execution time:', perf_counter() - start_time)

print('Acceptance %age:', 100*cntr_1peak_wf/cntr_wf)

plt.figure(figsize=(8,6))
# plt.hist(integration_vector, color='C0', label='tr0')
# plt.hist(integration_vector, bins=np.arange(0, 15000, 10), color='C0', label='tr0')
plt.hist(integration_vector, bins=np.arange(0, 15000, 10), color='C2', label='tr2')
# plt.hist(integration_vector, bins=np.arange(0, 50000, 100), color='C0', label='tr0')
# plt.plot(np.arange(500, 700), integration_vector_dict_l['tr1'][500:700], color='red', label='pulse integration window')
# plt.yscale('log')
plt.grid(which='both')
plt.legend(loc='upper right')
plt.title('Fingerplot after filtering with TSpectrum & restricting peak position b/w 500 & 600; channel 2 Threshold=0.5')
plt.savefig('../finger_plots/FP_filtered_TSpectrum_advance_ch2_Threshold_0_5_hist10.pdf')
plt.close()
plt.figure(figsize=(8,6))
plt.hist(integration_vector, bins=np.arange(0, 25000, 100), color='C2', label='tr2')
plt.grid(which='both')
plt.legend(loc='upper right')
plt.title('Fingerplot after filtering with TSpectrum & restricting peak position b/w 500 & 600; channel 2 Threshold=0.5')
plt.savefig('../finger_plots/FP_filtered_TSpectrum_advance_ch2_Threshold_0_5_hist100.pdf')